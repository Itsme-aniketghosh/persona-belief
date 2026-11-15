"""E3: steer at layer l, read the truth probe at layers l' > l.
Linear part: w·J·v from one backward pass. G = sum over positions of ds_{l'}/dh_l, so w·J·v = G·v,
direct = w_{l'}·v (residual skip path), mediated = G·v - w_{l'}·v. G does not depend on v, so 1000 random
vectors are scored for free. Finite-α part: actual Δs, 20 norm-matched random vectors, held-out NLL."""
import argparse, json, os, sys, time
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.acts import last_token_acts
from src.steer import steer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--traits", nargs="+", default=["evil", "sycophantic", "hallucinating"])
p.add_argument("--steer_layers", type=int, nargs="+", default=[16, 20])  # Chen et al.: 20 (evil, syc), 16 (halluc)
p.add_argument("--alphas", type=float, nargs="+", default=[-8.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 8.0])
p.add_argument("--n_null", type=int, default=20, help="random vectors per (trait, layer) for finite-α steering")
p.add_argument("--n_null_lin", type=int, default=1000, help="random vectors for the linear (free) null")
p.add_argument("--n", type=int, default=240, help="generic statements (6 GoT datasets)")
p.add_argument("--n_claim", type=int, default=120, help="common_claim statements (uncertain-claims proxy)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default=os.path.expanduser("~/pb/data/got"))
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
p.add_argument("--e0_results", default=os.path.expanduser("~/pb/results"))
p.add_argument("--heldout", action="store_true", help="generic statements the e00 probe never saw")
p.add_argument("--probe", default="lr", choices=["lr", "md"])
p.add_argument("--pv_tag", default="", help="persona vectors from a tagged persona.py rerun")
p.add_argument("--shuffle_acts", default=None, help="dir with pos/neg response acts: label-shuffled persona null")
p.add_argument("--n_shuffle", type=int, default=200)
p.add_argument("--orth", action="store_true", help="E8: remove v's component along the truth direction at layer l")
a = p.parse_args()
torch.manual_seed(a.seed)
slug = a.model.split("/")[-1]
e0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e00_baselines")
os.makedirs(a.out, exist_ok=True)

DS = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "larger_than", "smaller_than"]
if a.heldout:
    from src.got import heldout_generic
    gen = heldout_generic(a.data, a.n, a.seed + 2)
else:
    gen = pd.concat([pd.read_csv(f"{a.data}/{d}.csv").sample(a.n // len(DS), random_state=a.seed + 2) for d in DS])
clm = pd.read_csv(f"{a.data}/common_claim_true_false.csv").sample(a.n_claim, random_state=a.seed + 2)
df = pd.concat([gen.assign(stratum="generic"), clm.assign(stratum="common_claim")], ignore_index=True)
texts, strata = df.statement.tolist(), df.stratum.values
y = torch.tensor(df.label.values, dtype=torch.float32)

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
model.requires_grad_(False)
dev = model.device
P = torch.load(f"{e0}/probes_{slug}.pt")[a.probe].cuda()  # [L+1, d+1], index l = output of block l
W, B = P[:, :-1], P[:, -1]
nL, d = W.shape
acc0 = json.load(open(f"{a.e0_results}/e00_probes_{slug}.json"))["results"][a.probe]["acc"]
PVT = f"_{a.pv_tag}" if a.pv_tag else ""
V = {t: torch.load(f"{e0}/pv_{t}_{slug}{PVT}.pt").cuda() for t in a.traits}

# held-out text for the coherence check: E0's unsteered answers to the eval questions
E0 = json.load(open(f"{a.e0_results}/e00_persona_{slug}.json"))["traits"]
def chat(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
ref = [(chat(s["q"]), s["r"]) for t in E0 for s in E0[t]["steer"]["0.0"]["samples"]]

@torch.no_grad()
def ref_nll(vec, layer, bs=16):
    """Mean NLL per response token of the reference answers, with vec added at layer (None = unsteered)."""
    tok.padding_side = "right"
    tot = []
    for i in range(0, len(ref), bs):
        Pr, R = zip(*ref[i:i + bs])
        b = tok([p + r for p, r in zip(Pr, R)], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        pl = torch.tensor([len(tok(p, add_special_tokens=False).input_ids) for p in Pr], device=dev)
        m = ((torch.arange(b.input_ids.shape[1], device=dev)[None] >= pl[:, None]) & b.attention_mask.bool())[:, 1:].float()
        if vec is None:
            lg = model(**b).logits
        else:
            with steer(model, layer, vec):
                lg = model(**b).logits
        nll = F.cross_entropy(lg[:, :-1].float().transpose(1, 2), b.input_ids[:, 1:], reduction="none")
        tot.append((nll * m).sum(1) / m.sum(1))
    return torch.cat(tot).mean().item()

def readout_grads(l, reads, bs=16):
    """G[i, k] = sum over positions of d s_{reads[k]} / d h_l for statement i: float32 [n, len(reads), d]."""
    tok.padding_side = "left"
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(dev)
        delta = torch.zeros(b.input_ids.shape[0], 1, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
        got = {}

        def grab(k):
            def hook(mod, inp, o):
                got[k] = (o[0] if isinstance(o, tuple) else o)[:, -1].float()
            return hook

        hs = [model.model.layers[k - 1].register_forward_hook(grab(k)) for k in reads]
        try:
            with steer(model, l, delta):  # broadcast over positions, so the grad is the sum over positions
                model.model(**b)
        finally:
            for h in hs:
                h.remove()
        g = [torch.autograd.grad((got[k] @ W[k] + B[k]).sum(), delta, retain_graph=k != reads[-1])[0][:, 0].float()
             for k in reads]
        out.append(torch.stack(g, 1))
    return torch.cat(out)

def boot(x, B_=1000):  # mean and 95% CI over statements
    m = x[torch.randint(len(x), (B_, len(x)), device=x.device)].mean(1)
    return [x.mean().item(), m.quantile(0.025).item(), m.quantile(0.975).item()]

t0 = time.time()
H0 = last_token_acts(model, tok, texts)
s0 = (H0.cuda() * W).sum(-1) + B  # [n, L+1]
probe_acc = {st: [((s0[strata == st, k] > 0).float().cpu() == y[strata == st]).float().mean().item() for k in range(nL)]
             for st in ("generic", "common_claim")}
sep = {st: [(s0[(strata == st) & (y.numpy() == 1), k].mean() - s0[(strata == st) & (y.numpy() == 0), k].mean()).item()
            for k in range(nL)] for st in probe_acc}  # true-false gap: the scale for "how far toward a flip"
print(f"base readout done {time.time() - t0:.0f}s; probe acc l=20 generic {probe_acc['generic'][20]:.3f} "
      f"common_claim {probe_acc['common_claim'][20]:.3f}", flush=True)

res = {"exp": "e03_cross_layer", "model": a.model, "seed": a.seed, "steer_layers": a.steer_layers, "alphas": a.alphas,
       "heldout": a.heldout, "probe_kind": a.probe, "pv_tag": a.pv_tag, "orth": a.orth, "shuffle_null": bool(a.shuffle_acts),
       "n_statements": {st: int((strata == st).sum()) for st in probe_acc}, "probe": "lr from e00 (hook acts)",
       "probe_acc_e00_heldout": acc0, "probe_acc_here": probe_acc, "true_false_gap": sep,
       "coherence": "mean NLL per token of 60 E0 unsteered answers", "nll_unsteered": ref_nll(None, None),
       "null_lin": f"{a.n_null_lin} isotropic random directions, norm matched per trait",
       "null_finite": f"{a.n_null} isotropic random directions per (trait, layer), norm matched", "traits": {}}
R1 = F.normalize(torch.randn(a.n_null_lin, d, device=dev), dim=1)  # unit; scaled per trait below
for l in a.steer_layers:
    reads = [k for k in range(l + 1, nL) if acc0[k] >= 0.85]
    G = readout_grads(l, reads)  # [n, K, d]
    print(f"l={l}: grads for readouts {reads} in {time.time() - t0:.0f}s", flush=True)
    for t in a.traits:
        v = V[t][l]
        if a.orth:  # E8: no direct overlap with the truth direction at the injection layer
            wl = F.normalize(W[l], dim=0)
            v = v - (v @ wl) * wl
        nv = v.norm().item()
        T = res["traits"].setdefault(t, {}).setdefault(str(l), {"v_norm": nv, "readouts": reads, "linear": {}, "finite": {}})
        # linear terms per unit α. Two nulls: isotropic directions, and mean differences of random half-splits of the
        # statement activations at layer l (anisotropic, lives where real activation differences live)
        Rn = R1 * nv
        Hl = H0[:, l].cuda()
        half = torch.rand(a.n_null_lin, len(texts), device=dev)
        half = (half < half.median(1, keepdim=True).values).float()
        Sp = F.normalize(half @ Hl / half.sum(1, keepdim=True) - (1 - half) @ Hl / (1 - half).sum(1, keepdim=True), dim=1) * nv
        Wr = torch.stack([W[k] for k in reads])
        fw = lambda M: [M.min(0).values.quantile(0.025).item(), M.max(0).values.quantile(0.975).item()]  # family-wise over readouts
        Sh = None
        if a.shuffle_acts:  # same pos/neg responses, labels shuffled: a persona-shaped direction with no trait in it
            ac = torch.load(f"{a.shuffle_acts}/acts_{t}_{slug}{PVT}.pt")
            X = torch.cat([ac["pos"][:, l], ac["neg"][:, l]]).float().to(dev)
            npos = len(ac["pos"])
            perm = torch.rand(a.n_shuffle, len(X), device=dev).argsort(1)
            m1 = torch.zeros(a.n_shuffle, len(X), device=dev).scatter_(1, perm[:, :npos], 1.0)
            Sh = F.normalize(m1 @ X / npos - (1 - m1) @ X / (len(X) - npos), dim=1) * nv
        for st in ("generic", "common_claim", "all"):
            msk = torch.ones(len(texts), dtype=torch.bool) if st == "all" else torch.tensor(strata == st)
            Gs = G[msk.to(G.device)]
            tot = Gs @ v  # [n_s, K] per statement
            direct = Wr @ v
            Gbar = Gs.mean(0)  # [K, d]
            med_null = Gbar @ Rn.T - Wr @ Rn.T  # [K, n_null_lin]
            med_split = Gbar @ Sp.T - Wr @ Sp.T
            T.setdefault("fw_null", {})[st] = {"iso": fw(med_null), "split": fw(med_split)}
            if Sh is not None:
                med_sh = Gbar @ Sh.T - Wr @ Sh.T
                T["fw_null"][st]["shuffle"] = fw(med_sh)
                T.setdefault("shuffle_95", {})[st] = [[med_sh[j].quantile(0.025).item(), med_sh[j].quantile(0.975).item()]
                                                      for j in range(len(reads))]
            T["linear"][st] = [{"readout": k, "total": boot(tot[:, j]), "direct": direct[j].item(),
                                "mediated": boot(tot[:, j] - direct[j]),
                                "mediated_null_95": [med_null[j].quantile(0.025).item(), med_null[j].quantile(0.975).item()],
                                "mediated_null_split_95": [med_split[j].quantile(0.025).item(), med_split[j].quantile(0.975).item()],
                                "mediated_over_gap": (tot[:, j].mean() - direct[j]).item() / sep["generic" if st == "all" else st][k]}
                               for j, k in enumerate(reads)]
        # finite α: actual Δs for v and n_null norm-matched random vectors, plus held-out NLL
        rnd = F.normalize(torch.randn(a.n_null, d, device=dev), dim=1) * nv
        for name, vec in [("v", v)] + [(f"r{i}", rnd[i]) for i in range(a.n_null)]:
            for al in a.alphas:
                sv = (al * vec).to(torch.bfloat16)
                with steer(model, l, sv):
                    ds = ((last_token_acts(model, tok, texts).cuda() * W).sum(-1) + B - s0)[:, reads]  # [n, K]
                yt = (y == 1).to(ds.device)  # per label: does the true/false gap shrink, or is it a uniform shift?
                T["finite"].setdefault(name, {})[str(al)] = {
                    "ds_mean": ds.mean(0).tolist(),
                    "ds_true": ds[yt].mean(0).tolist(), "ds_false": ds[~yt].mean(0).tolist(),
                    "ds_ci": [boot(ds[:, j])[1:] for j in range(len(reads))] if name == "v" else None,
                    "nll": ref_nll(sv, l)}
        lin = T["linear"]["all"]
        fv = T["finite"]["v"]
        if "0.5" in fv and "-0.5" in fv:  # self-check: gradient w·J·v vs actual central difference at α=±0.5
            cd = [(p_ - m_) for p_, m_ in zip(fv["0.5"]["ds_mean"], fv["-0.5"]["ds_mean"])]
            T["linear_check"] = [{"readout": x["readout"], "grad": x["total"][0], "central_diff": c} for x, c in zip(lin, cd)]
            print(f"  check grad vs central diff: " + ", ".join(f"{x['readout']}:{x['total'][0]:+.3f}/{c:+.3f}"
                                                              for x, c in zip(lin, cd)), flush=True)
        print(f"[{t} l={l} |v|={nv:.1f}] mediated/unit α by readout: "
              + ", ".join(f"{x['readout']}:{x['mediated'][0]:+.3f} (null {x['mediated_null_95'][0]:+.3f},{x['mediated_null_95'][1]:+.3f})"
                          for x in lin) + f" | fw iso {T['fw_null']['all']['iso']} split {T['fw_null']['all']['split']}"
              + f" | {time.time() - t0:.0f}s", flush=True)

json.dump(res, open(f"{a.out}/e03_cross_layer_{slug}.json", "w"))
print(f"done in {(time.time() - t0) / 60:.1f} min")
