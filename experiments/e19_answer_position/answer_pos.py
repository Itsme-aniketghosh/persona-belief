"""E19: what does the persona do at the answer position? (follows E16/E18: the answer change is carried at the answer
tail, not through the statement-level truth direction).
(a) Component patching at the answer position: in the yes/no question, overwrite the attention output, the MLP output,
    or the residual stream of one layer at the LAST token only with the persona run's values; recovery of the
    persona's effect on logit(True)-logit(False), per cell.
(b) Probes at two sites x three types: site = statement (last statement token, "Consider this statement" format) or
    answer (last token of the yes/no question); type = lr (logistic regression), md (mean difference), persona_lr
    (LR trained on activations under the persona). All trained on Geometry-of-Truth statements, gated at >= 85% on
    the setup's truth-labelled claims without persona. Reported: persona shift per cell at the readout, per-claim
    correlation between probe shift and answer shift, and cos(statement lr, answer lr) by layer.
(c) Causal test per probe: project its direction out at every layer from its first gated layer to the end (all
    positions); reduction of the persona's effect on answers vs 20 random directions (E5 rule), and steering along
    the answer-site lr direction (E15a rule).
Rules (fixed before the run):
- Answer-time truth variable: the answer-site lr probe passes the gate, the persona shifts it selectively (stake cells
  with 95% CI excluding 0, control persona smaller), AND ablating it cuts the persona's effect on every stake cell by
  >= 50% and by more than 95% of random directions, with no-persona accuracy on truth-labelled claims >= 85% under the
  ablation. All three -> consistent with "the persona acts on an answer-time truth variable the answer reads".
- Steering (E15a rule): +-2 gap moves the answer beyond the random q95 both ways and flips >= 25% both ways.
- Component patching is descriptive: which layers' attention / MLP outputs at the answer token carry >= 50%."""
import argparse, json, os, sys, time
from contextlib import contextmanager
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.acts import last_token_acts
from src.got import DS
from src.probes import fit_lr, fit_md, correct
from src import setup as S

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--setups", nargs="+", default=["historical", "partisan"])
p.add_argument("--n_got", type=int, default=400)
p.add_argument("--n_null", type=int, default=20)
p.add_argument("--max_claims", type=int, default=None, help="tiny pass")
p.add_argument("--data", default=os.path.expanduser("~/pb/data/got"))
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
torch.manual_seed(0)
slug = a.model.split("/")[-1]
os.makedirs(a.out, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
dev = model.device
idT, idF = (tok.encode(w, add_special_tokens=False)[0] for w in ("True", "False"))
LAY = model.model.layers
nL = len(LAY)
got = pd.concat([pd.read_csv(f"{a.data}/{d}.csv").sample(frac=1, random_state=0).groupby("label").head(a.n_got // 2) for d in DS])
if a.max_claims:
    got = got.groupby("label").head(40)
yg = torch.tensor(got.label.values, dtype=torch.float32, device=dev)
acts = lambda texts: last_token_acts(model, tok, texts, add_special_tokens=False).cuda()

@contextmanager
def hooked(fns):
    """fns: {layer: f(h) -> h} on block outputs."""
    hs = []
    for k, fn in fns.items():
        def hk(mod, inp, o, fn=fn):
            h = fn(o[0] if isinstance(o, tuple) else o)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        hs.append(LAY[k - 1].register_forward_hook(hk))
    try:
        yield
    finally:
        for h in hs:
            h.remove()

@torch.no_grad()
def tf(texts, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        lg = model(**b).logits[:, -1].float()
        out.append(lg[:, idT] - lg[:, idF])
    return torch.cat(out)

def boot(x, B=1000):
    m = x[torch.randint(len(x), (B, len(x)), device=x.device)].mean(1)
    return [x.mean().item(), m.quantile(0.025).item(), m.quantile(0.975).item()]

t0 = time.time()
res = {"exp": "e19_answer_position", "model": a.model, "seed": 0, "rule": __doc__.split("Rules")[1].strip(), "setups": {}}
for name in a.setups:
    cfg, cl = S.load(name)
    if a.max_claims:
        cl = cl.groupby("cell").head(a.max_claims // 4)
    cl = cl.reset_index(drop=True)
    SYS, CELLS, STAKE = cfg["conditions"], cfg["cells"], cfg["stake_cells"]
    wrap, ask, _ = S.formats(tok, SYS)
    cm = {c: torch.tensor(cl.cell.values == c, device=dev) for c in CELLS}
    has_t = torch.tensor((cl.truth != "").values, device=dev)
    yt = torch.tensor(pd.to_numeric(cl.truth, errors="coerce").fillna(-1).values, dtype=torch.float32, device=dev)
    fmt = {"statement": wrap, "answer": ask}
    T = {c: tf([ask(c, s) for s in cl.statement]) for c in SYS}
    dT = T["hist"] - T["none"]
    R = {"behavior_effect": {c: boot(dT[cm[c]]) for c in CELLS}, "probes": {}, "ablation": {}, "patch_last": {}}

    # (b) probes: two sites x three types
    D = {}
    for site, f in fmt.items():
        Hg = {c: acts([f(c, s) for s in got.statement]) for c in ("none", "hist")}
        Hc = {c: acts([f(c, s) for s in cl.statement]) for c in SYS}
        for kind in ("lr", "md", "persona_lr"):
            X = Hg["hist" if kind == "persona_lr" else "none"]
            fit = fit_md if kind == "md" else fit_lr
            Wb = [fit(X[:, l], yg) for l in range(nL + 1)]
            acc = [correct(Hc["none"][has_t, l], yt[has_t], *Wb[l]).mean().item() for l in range(nL + 1)]
            gl = [l for l in range(nL + 1) if acc[l] >= 0.85]
            P = {"acc": acc, "gate_layers": gl}
            if gl:
                lt = max(gl, key=lambda l: acc[l])
                s = {c: (Hc[c][:, lt] @ Wb[lt][0] + Wb[lt][1]) for c in SYS}
                ds = s["hist"] - s["none"]
                P.update({"readout": lt, "shift_hist": {c: boot(ds[cm[c]]) for c in CELLS},
                          "shift_ctrl": {c: boot((s["ctrl"] - s["none"])[cm[c]]) for c in CELLS},
                          "corr_probe_vs_answer_shift": torch.corrcoef(torch.stack([ds, dT]))[0, 1].item()})
                D[(site, kind)] = (Wb, gl, lt, Hc)
            R["probes"][f"{site}_{kind}"] = P
            print(f"[{slug}/{name}] {site:9s} {kind:10s} gate {gl[:1]}..{gl[-1:]} " + (f"readout {P['readout']} acc {acc[P['readout']]:.2f} shift "
                  + ", ".join(f"{c}:{P['shift_hist'][c][0]:+.2f}" for c in STAKE) + f" r(probe,answer) {P['corr_probe_vs_answer_shift']:+.2f}"
                  if gl else "(fails gate)") + f" | {time.time() - t0:.0f}s", flush=True)
        del Hg
    if ("statement", "lr") in D and ("answer", "lr") in D:
        R["cos_statement_answer_lr"] = [F.cosine_similarity(D[("statement", "lr")][0][l][0], D[("answer", "lr")][0][l][0], dim=0).item()
                                        for l in range(nL + 1)]

    # (c) causal: ablate each gated probe's direction from its first gated layer on
    tn, th = [ask("none", s) for s in cl.statement], [ask("hist", s) for s in cl.statement]
    def summary(fns):
        with hooked(fns):
            n, h = tf(tn), tf(th)
        acc = ((n[has_t] > 0).float() == yt[has_t]).float().mean().item()
        return {c: (h - n)[cm[c]].mean().item() for c in CELLS}, acc, n.std().item()
    base, base_acc, base_sd = summary({})
    proj = lambda U: {l: (lambda h, u=u.to(torch.bfloat16): h - (h @ u)[..., None] * u) for l, u in U.items()}
    red = lambda e: {c: 1 - e[c] / base[c] if abs(base[c]) > 1e-6 else None for c in CELLS}
    for (site, kind), (Wb, gl, lt, Hc) in D.items():
        ls = list(range(min(gl), nL + 1))
        e, acc, sd = summary(proj({l: F.normalize(Wb[l][0], dim=0) for l in ls}))
        nr = [red(summary(proj({l: F.normalize(torch.randn_like(Wb[l][0]), dim=0) for l in ls}))[0]) for _ in range(a.n_null)]
        q95 = {c: torch.tensor([r[c] for r in nr]).quantile(0.95).item() for c in STAKE}
        rd = red(e)
        A = {"layers": ls, "effect": e, "reduction": rd, "null_q95": q95, "acc_none": acc, "acc_none_base": base_acc,
             "logit_sd": sd, "logit_sd_base": base_sd,
             "pass": all(rd[c] >= 0.5 and rd[c] > q95[c] for c in STAKE) and acc >= 0.85}
        R["ablation"][f"{site}_{kind}"] = A
        print(f"[{slug}/{name}] ablate {site}_{kind} from l={ls[0]}: reduction " + ", ".join(f"{c}:{rd[c]:.0%} (q95 {q95[c]:.0%})" for c in STAKE)
              + f" | acc none {base_acc:.2f}->{acc:.2f}, logit sd {base_sd:.1f}->{sd:.1f} | pass {A['pass']} | {time.time() - t0:.0f}s", flush=True)
    if ("answer", "lr") in D:  # steering along the answer-site direction (E15a rule)
        Wb, gl, lt, Hc = D[("answer", "lr")]
        u = F.normalize(Wb[lt][0], dim=0)
        pr = Hc["none"][has_t, lt] @ u
        gap = (pr[yt[has_t] == 1].mean() - pr[yt[has_t] == 0].mean()).item()
        n0 = tf(tn)
        St = {"layer": lt, "gap": gap, "alpha": {}}
        for al in (-2.0, 2.0):
            with hooked({lt: lambda h, v=(al * gap * u).to(torch.bfloat16): h + v}):
                d = tf(tn)
            tgt = n0 < 0 if al > 0 else n0 > 0
            St["alpha"][str(al)] = {"delta": (d - n0).mean().item(), "flip_to_steered": ((d > 0) if al > 0 else (d < 0))[tgt].float().mean().item()}
        nd = []
        for _ in range(a.n_null):
            with hooked({lt: lambda h, r=(2 * abs(gap) * F.normalize(torch.randn_like(u), dim=0)).to(torch.bfloat16): h + r}):
                nd.append((tf(tn) - n0).mean().abs().item())
        q = torch.tensor(nd).quantile(0.95).item()
        up, dn = St["alpha"]["2.0"], St["alpha"]["-2.0"]
        St.update({"null_q95": q, "pass": up["delta"] > q and dn["delta"] < -q and up["flip_to_steered"] >= 0.25 and dn["flip_to_steered"] >= 0.25})
        R["steer_answer_lr"] = St
        print(f"[{slug}/{name}] steer answer_lr l={lt}: Δ {dn['delta']:+.2f}/{up['delta']:+.2f} flips {dn['flip_to_steered']:.0%}/{up['flip_to_steered']:.0%} "
              f"| null q95 {q:.2f} | pass {St['pass']}", flush=True)

    # (a) component patching at the last token, persona -> none
    comps = {"attn": lambda j: LAY[j - 1].self_attn, "mlp": lambda j: LAY[j - 1].mlp, "resid": lambda j: LAY[j - 1]}
    def run_cap(texts):
        caps, hs = {}, []
        for kc, mod in comps.items():
            for j in range(1, nL + 1):
                def hk(m, i, o, key=(kc, j)):
                    caps.setdefault(key, []).append((o[0] if isinstance(o, tuple) else o)[:, -1].clone())
                hs.append(mod(j).register_forward_hook(hk))
        try:
            d = tf(texts)
        finally:
            for h in hs:
                h.remove()
        return d, {k: torch.cat(v) for k, v in caps.items()}
    _, src = run_cap(th)
    full = dT
    PL = {}
    for kc, mod in comps.items():
        PL[kc] = {}
        for j in range(1, nL + 1):
            buf, pos = src[(kc, j)], [0]
            def hk(m, i, o):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                h[:, -1] = buf[pos[0]:pos[0] + h.shape[0]]
                pos[0] += h.shape[0]
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            hh = mod(j).register_forward_hook(hk)
            try:
                d = tf(tn) - T["none"]
            finally:
                hh.remove()
            PL[kc][str(j)] = {c: d[cm[c]].mean().item() / full[cm[c]].mean().item() for c in CELLS}
        top = sorted(PL[kc], key=lambda j: -min(PL[kc][j][c] for c in STAKE))[:3]
        print(f"[{slug}/{name}] last-token patch {kc}: best layers " + ", ".join(f"{j}:" + "/".join(f"{PL[kc][j][c]:.0%}" for c in STAKE) for j in top)
              + f" | {time.time() - t0:.0f}s", flush=True)
    R["patch_last"] = PL
    res["setups"][name] = R
json.dump(res, open(f"{a.out}/e19_answer_position_{slug}.json", "w"), indent=1)
print(f"done in {(time.time() - t0) / 60:.1f} min")
