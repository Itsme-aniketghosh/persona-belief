"""E15: is the truth direction causal for the answer at all? Closes the hole in E5, where ablating the probe direction w
left no-persona answer accuracy at ~99%, so "the persona bypasses w" and "the model never reads w" were not separable.
(a) Steering: add alpha * gap * w_hat at block l (all positions), no persona, yes/no question on the historical claims;
    l = first gate layer k0 and the readout lt; gap = true-minus-false mean of h . w_hat on the wrapped claims.
    20 random unit directions with the same norm as the null.
    Rule (fixed before the run): w steers the answer at layer l if at alpha=+2 logit(True)-logit(False) rises and at
    alpha=-2 it falls, both by more than the 95th percentile of |delta| over random directions, and >= 25% of answers
    that disagreed with the steered label flip to it.
(b) Iterative nullspace (INLP): at every layer >= k0, fit the GoT truth probe, project its direction out, refit, up to
    K times (orthonormal basis Q_l). Ablate span(Q_l[:, :k]) at all layers >= k0 in the none and persona runs.
    k* = smallest k in the grid where no-persona answer accuracy on the historical claims (all truth-labelled) <= 65%.
    Rule (fixed before the run): at k*, the persona's behavioral effect on every stake cell is cut by >= 50% and by more
    than 95% of random k*-dim subspace ablations -> consistent with the effect going through the truth subspace;
    cut < 50% while truth reading is broken -> consistent with a path that bypasses it. No k reaching 65%: inconclusive.
Coherence next to every intervention: mean token NLL of 100 held-out GoT statements (plain text) and p(True)+p(False).
E20 (options, rule fixed before the E20 runs): --site answer trains the probes at the last token of the yes/no question
instead of the statement; --from_layer starts the ablation earlier than the gate; the random-subspace control always runs,
at k* or else at --control_k. Normalized reduction nr = 1 - (effect_k / logit_sd_k) / (effect_0 / logit_sd_0).
Verdict at k_c (= k* if reached, else control_k): "through" if nr >= 50% on every stake cell and above 95% of random
subspaces; "bypass" if no-persona accuracy <= 65% and nr < 50% on every stake cell; otherwise "not_used" (removing the
subspace neither breaks truth answering nor removes the persona effect beyond what general damage does)."""
import argparse, json, os, sys, time
from contextlib import contextmanager
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.acts import last_token_acts
from src.got import DS, heldout_generic
from src.probes import fit_lr, correct
from src import setup as S

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--setups", nargs="+", default=["historical", "physician1850", "partisan", "opposite"])
p.add_argument("--kgrid", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
p.add_argument("--n_null", type=int, default=20)
p.add_argument("--n_null_sub", type=int, default=10)
p.add_argument("--n_got", type=int, default=400)
p.add_argument("--inlp_iters", type=int, default=100)
p.add_argument("--site", default="statement", choices=["statement", "answer"])
p.add_argument("--from_layer", type=int, default=None, help="first ablated layer (default: first gated layer)")
p.add_argument("--control_k", type=int, default=None, help="random-subspace control k when k* is not reached (default max kgrid)")
p.add_argument("--tag", default="", help="output file suffix")
p.add_argument("--max_claims", type=int, default=None, help="tiny pass")
p.add_argument("--data", default=os.path.expanduser("~/pb/data/got"))
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
p.add_argument("--e0_results", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
torch.manual_seed(0)
slug = a.model.split("/")[-1]
e12 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e12_historical")
os.makedirs(a.out, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
dev = model.device
idT, idF = (tok.encode(w, add_special_tokens=False)[0] for w in ("True", "False"))

def claims(name):
    cfg, cl = S.load(name)
    if a.max_claims:
        cl = cl.groupby("cell").head(a.max_claims // 4)
    return cfg, cl.reset_index(drop=True)

cfg_h, cl_h = claims("historical")
wrap, ask_h, _ = S.formats(tok, cfg_h["conditions"])
gate = json.load(open(f"{a.e0_results}/e12a_historical_gate_{slug}.json"))
k0, lt = min(gate["gate_truth_layers"]), gate["truth_readout"]
W = torch.load(f"{e12}/probes_chat_{slug}.pt")["truth"][:, :-1].cuda()
L = W.shape[0] - 1
feat = wrap if a.site == "statement" else ask_h  # where the probes read

@contextmanager
def hooked(fns):
    """fns: {layer: f(h) -> h} applied to block outputs (hidden index = block number)."""
    hs = []
    for k, fn in fns.items():
        def hk(mod, inp, o, fn=fn):
            h = fn(o[0] if isinstance(o, tuple) else o)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        hs.append(model.model.layers[k - 1].register_forward_hook(hk))
    try:
        yield
    finally:
        for h in hs:
            h.remove()

@torch.no_grad()
def tf(texts, bs=16):
    """(log p(True) - log p(False), p(True) + p(False)) at the answer position."""
    d, m = [], []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        lp = model(**b).logits[:, -1].float().log_softmax(-1)
        d.append(lp[:, idT] - lp[:, idF]); m.append(lp[:, idT].exp() + lp[:, idF].exp())
    return torch.cat(d), torch.cat(m)

nll_texts = heldout_generic(a.data, 100, 7).statement.tolist()
@torch.no_grad()
def nll(bs=25):
    tok.padding_side = "right"
    tot, n = 0.0, 0
    try:
        for i in range(0, len(nll_texts), bs):
            b = tok(nll_texts[i:i + bs], return_tensors="pt", padding=True).to(dev)
            lg = model(**b).logits[:, :-1].float()
            msk = b.attention_mask[:, 1:].bool()
            l = F.cross_entropy(lg.transpose(1, 2), b.input_ids[:, 1:], reduction="none")
            tot += l[msk].sum().item(); n += msk.sum().item()
    finally:
        tok.padding_side = "left"
    return tot / n

t0 = time.time()
yh = torch.tensor(pd.to_numeric(cl_h.truth).values, dtype=torch.float32, device=dev)
Hc = last_token_acts(model, tok, [feat("none", s) for s in cl_h.statement], add_special_tokens=False).cuda()
got = pd.concat([pd.read_csv(f"{a.data}/{d}.csv").sample(frac=1, random_state=0).groupby("label").head(a.n_got // 2) for d in DS])
yg = torch.tensor(got.label.values, dtype=torch.float32, device=dev)
Hg = last_token_acts(model, tok, [feat("none", s) for s in got.statement], add_special_tokens=False)
if a.site == "answer":  # probe, gate and readout at the answer position
    Wb = [fit_lr(Hg[:, l].cuda(), yg) for l in range(L + 1)]
    W = torch.stack([w for w, _ in Wb])
    acc_a = [correct(Hc[:, l], yh, *Wb[l]).mean().item() for l in range(L + 1)]
    gl = [l for l in range(L + 1) if acc_a[l] >= 0.85]
    assert gl, "answer-site probe fails the 85% gate at every layer"
    k0, lt = min(gl), max(gl, key=lambda l: acc_a[l])
layers = list(range(a.from_layer or k0, L + 1))
texts_none = [ask_h("none", s) for s in cl_h.statement]
d0, m0 = tf(texts_none)
acc = lambda d: ((d > 0).float() == yh).float().mean().item()
nll0 = nll()
res = {"exp": "e15_truth_causal", "model": a.model, "seed": 0, "site": a.site, "k0": k0, "readout": lt, "ablated_layers": layers, "rule": __doc__,
       "baseline": {"acc": acc(d0), "tf_mass": m0.mean().item(), "nll": nll0, "logit_sd": d0.std().item()}, "steer": {}}
print(f"[{slug}] baseline acc {acc(d0):.3f}, T/F mass {m0.mean():.3f}, NLL {nll0:.3f} | k0={k0} lt={lt}", flush=True)

# (a) steering along w
for l in sorted({k0, lt}):
    u = F.normalize(W[l], dim=0)
    pr = Hc[:, l] @ u
    gap = (pr[yh == 1].mean() - pr[yh == 0].mean()).item()
    R = {"gap": gap, "alpha": {}}
    for al in (-2.0, -1.0, 1.0, 2.0):
        v = (al * gap * u).to(torch.bfloat16)
        with hooked({l: lambda h, v=v: h + v}):
            d, m = tf(texts_none); nl = nll()
        tgt = d0 < 0 if al > 0 else d0 > 0  # answers that disagree with the steered label
        flip = ((d > 0) if al > 0 else (d < 0))[tgt].float().mean().item() if tgt.any() else None
        R["alpha"][str(al)] = {"delta": (d - d0).mean().item(), "flip_to_steered": flip, "acc": acc(d),
                               "tf_mass": m.mean().item(), "nll": nl}
    nd = []
    for _ in range(a.n_null):
        r = (2 * abs(gap) * F.normalize(torch.randn_like(u), dim=0)).to(torch.bfloat16)
        with hooked({l: lambda h, r=r: h + r}):
            nd.append((tf(texts_none)[0] - d0).mean().abs().item())
    R["null_abs_delta"] = nd
    q = torch.tensor(nd).quantile(0.95).item()
    up, dn = R["alpha"]["2.0"], R["alpha"]["-2.0"]
    R["pass"] = bool(up["delta"] > q and dn["delta"] < -q and (up["flip_to_steered"] or 0) >= 0.25 and (dn["flip_to_steered"] or 0) >= 0.25)
    res["steer"][str(l)] = R
    print(f"[{slug}] steer w at l={l} (gap {gap:.2f}): " + ", ".join(f"α={k}: Δ{x['delta']:+.2f} flip {x['flip_to_steered'] or 0:.0%} "
          f"NLL {x['nll']:.2f}" for k, x in R["alpha"].items()) + f" | null |Δ| q95 {q:.2f} | pass {R['pass']} | {time.time() - t0:.0f}s", flush=True)

# (b) INLP basis per layer, from chat-wrapped GoT (the gate's probe training set)
K = max(a.kgrid)
Q, probe_acc, skipped, short = {}, {}, [], {}
for l in layers:
    X = Hg[:, l].cuda()
    basis = []
    for i in range(K):
        Qm = torch.stack(basis, 1) if basis else None
        Xp = X - (X @ Qm) @ Qm.T if basis else X
        w, b = fit_lr(Xp, yg, iters=a.inlp_iters)
        if l == lt and (i in a.kgrid or i == 0):  # held-out check: linear truth info left after i removals
            Hp = Hc[:, l] - (Hc[:, l] @ Qm) @ Qm.T if basis else Hc[:, l]
            probe_acc[i] = correct(Hp, yh, w, b).mean().item()
        u = w - Qm @ (Qm.T @ w) if basis else w
        u = u - Qm @ (Qm.T @ u) if basis else u  # second Gram-Schmidt pass
        if w.norm() == 0 or u.norm() <= 1e-6 * w.norm():
            break  # post-hoc (E20): no linear truth signal left at this layer after i removals (early layers), nothing more to remove
        basis.append(F.normalize(u, dim=0))
    if not basis:
        skipped.append(l)
        continue
    Q[l] = torch.stack(basis, 1)
    if Q[l].shape[1] < K:
        short[l] = Q[l].shape[1]
    err = (Q[l].T @ Q[l] - torch.eye(Q[l].shape[1], device=dev)).abs().max()
    assert err < 1e-3, f"basis not orthonormal at l={l}: err {err:.3g}, zero columns {(Q[l].norm(dim=0) == 0).sum().item()}"
del Hg
res["inlp_heldout_probe_acc_at_readout"] = probe_acc
res["inlp_skipped_layers_no_signal"] = skipped
res["inlp_basis_size_if_short"] = short  # layer -> directions found before the signal ran out
print(f"[{slug}] INLP basis K={K} at {len(Q)} layers (skipped, no signal: {skipped}; shorter bases: {short}); held-out probe acc at l={lt} after i removals: "
      + ", ".join(f"{i}:{x:.2f}" for i, x in probe_acc.items()) + f" | {time.time() - t0:.0f}s", flush=True)

def proj(sub):
    return {l: (lambda h, B=B.to(torch.bfloat16): h - (h @ B) @ B.T) for l, B in sub.items()}

grid = {}
for k in a.kgrid:
    with hooked(proj({l: Q[l][:, :k] for l in Q})):
        d, m = tf(texts_none); nl = nll()
    grid[k] = {"acc": acc(d), "tf_mass": m.mean().item(), "nll": nl, "logit_sd": d.std().item()}
kstar = next((k for k in a.kgrid if grid[k]["acc"] <= 0.65), None)
kc = kstar or a.control_k or K
rand_sub = [{l: torch.linalg.qr(torch.randn(Q[l].shape[0], min(kc, Q[l].shape[1]), device=dev))[0] for l in Q} for _ in range(a.n_null_sub)]
rand_stats = []
for sub in rand_sub:
    with hooked(proj(sub)):
        d, m = tf(texts_none); nl = nll()
    rand_stats.append({"acc": acc(d), "tf_mass": m.mean().item(), "nll": nl})
res.update({"inlp_grid": grid, "kstar": kstar, "k_control": kc, "random_subspace_at_kcontrol": rand_stats, "setups": {}})
print(f"[{slug}] INLP no-persona acc by k: " + ", ".join(f"{k}:{g['acc']:.2f}(NLL {g['nll']:.2f}, mass {g['tf_mass']:.2f})" for k, g in grid.items())
      + f" | k*={kstar} | random k={kc} acc " + ", ".join(f"{r['acc']:.2f}" for r in rand_stats), flush=True)

for name in a.setups:
    cfg, cl = claims(name)
    _, ask, _ = S.formats(tok, cfg["conditions"])
    CELLS, STAKE = cfg["cells"], cfg["stake_cells"]
    cm = {c: torch.tensor(cl.cell.values == c, device=dev) for c in CELLS}
    tn, th = [ask("none", s) for s in cl.statement], [ask("hist", s) for s in cl.statement]
    def effect(fns):
        with hooked(fns):
            dn, dh = tf(tn)[0], tf(th)[0]
        return {c: (dh - dn)[cm[c]].mean().item() for c in CELLS}, dn.std().item()
    base, sd0 = effect({})
    red = lambda e: {c: 1 - e[c] / base[c] if abs(base[c]) > 1e-6 else None for c in CELLS}
    nred = lambda e, sd: {c: 1 - (e[c] / sd) / (base[c] / sd0) if abs(base[c]) > 1e-6 else None for c in CELLS}
    R = {"persona_effect": base, "logit_sd_none": sd0, "by_k": {}}
    for k in sorted(set(a.kgrid) | {kc}):
        e, sd = effect(proj({l: Q[l][:, :k] for l in Q}))
        R["by_k"][str(k)] = {"effect": e, "reduction": red(e), "norm_reduction": nred(e, sd), "logit_sd_none": sd}
    nr = [nred(*effect(proj(sub))) for sub in rand_sub]
    q95 = {c: torch.tensor([r[c] for r in nr]).quantile(0.95).item() for c in STAKE}
    rk = R["by_k"][str(kc)]["norm_reduction"]
    acc_kc = grid[kc]["acc"] if kc in grid else None
    through = all(rk[c] >= 0.5 and rk[c] > q95[c] for c in STAKE)
    bypass = acc_kc is not None and acc_kc <= 0.65 and all(rk[c] < 0.5 for c in STAKE)
    R.update({"null_norm_reduction": nr, "null_q95": q95, "k_control": kc,
              "verdict": "through" if through else "bypass" if bypass else "not_used",
              "through_truth_subspace": through, "bypass": bypass})
    res["setups"][name] = R
    print(f"[{slug}/{name}] persona effect " + ", ".join(f"{c}:{base[c]:+.2f}" for c in STAKE) + " | reduction by k: "
          + "; ".join(f"{k}:" + ",".join(f"{R['by_k'][str(k)]['reduction'][c]:.0%}" for c in STAKE) for k in a.kgrid)
          + f" | at k={kc}: norm reduction " + ",".join(f"{rk[c]:.0%}" for c in STAKE) + " null q95 " + ",".join(f"{q95[c]:.0%}" for c in STAKE)
          + f" -> {R['verdict']} | {time.time() - t0:.0f}s", flush=True)
json.dump(res, open(f"{a.out}/e15_truth_causal_{slug}{a.tag}.json", "w"), indent=1)
print(f"done in {(time.time() - t0) / 60:.1f} min")
