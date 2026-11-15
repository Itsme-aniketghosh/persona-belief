"""E5 causal check: does the persona's effect on behavior go through the truth direction?
Project the chat truth-probe direction w_k out of the residual stream (all positions) at every layer k >= k0, in both
the no-persona and persona runs of the yes/no question, and compare the persona's behavioral effect
(logit(True)-logit(False), persona minus none) with and without ablation, against ablating random directions.
Rule (fixed before the run): ablating w cuts the persona's behavioral effect on every stake cell by >= 50% and by more
than 95% of random-direction ablations. Confound check: no-persona accuracy on shared cells under ablation is reported;
if it falls below 85%, the reduction is not interpretable as "through w"."""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src import setup as S

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--setup", default="historical")
p.add_argument("--n_null", type=int, default=20)
p.add_argument("--max_claims", type=int, default=None)
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
p.add_argument("--e0_results", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
torch.manual_seed(0)
slug = a.model.split("/")[-1]
here = os.path.dirname(os.path.abspath(__file__))
cfg, cl = S.load(a.setup)
if a.max_claims:
    cl = cl.groupby("cell").head(a.max_claims // 4)
CELLS, STAKE, SHARED = cfg["cells"], cfg["stake_cells"], cfg["shared_cells"]
cells = cl.cell.values

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
dev = model.device
_, ask, _ = S.formats(tok, cfg["conditions"])
idT, idF = (tok.encode(w, add_special_tokens=False)[0] for w in ("True", "False"))
PR = torch.load(f"{here}/probes_chat_{S.file_tag(a.setup)}{slug}.pt")
U = F.normalize(PR["truth"][:, :-1].cuda(), dim=1)  # [L+1, d] truth directions
gate = json.load(open(f"{a.e0_results}/e12a_{a.setup}_gate_{slug}.json"))
k0 = min(gate["gate_truth_layers"])
layers = list(range(k0, U.shape[0]))

def behavior(cond, dirs=None, bs=16):
    """logit(True)-logit(False); dirs: {layer: unit vector} projected out of block outputs."""
    hs = []
    if dirs:
        for k, u in dirs.items():
            def hk(mod, inp, o, u=u.to(torch.bfloat16)):
                h = o[0] if isinstance(o, tuple) else o
                h = h - (h @ u)[..., None] * u
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            hs.append(model.model.layers[k - 1].register_forward_hook(hk))
    out = []
    try:
        with torch.no_grad():
            for i in range(0, len(cl), bs):
                b = tok([ask(cond, s) for s in cl.statement.iloc[i:i + bs]], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(dev)
                lg = model(**b).logits[:, -1].float()
                out.append(lg[:, idT] - lg[:, idF])
    finally:
        for h in hs:
            h.remove()
    return torch.cat(out)

t0 = time.time()
cm = {c: torch.tensor(cells == c, device=dev) for c in CELLS}
has_t = torch.tensor((cl.truth != "").values, device=dev)
yt = torch.tensor([float(x) if x != "" else -1.0 for x in cl.truth], device=dev)
def summary(dirs):
    n, h = behavior("none", dirs), behavior("hist", dirs)
    eff = {c: (h - n)[cm[c]].mean().item() for c in CELLS}
    sh = torch.zeros(len(cl), dtype=torch.bool, device=dev)
    for c in SHARED:
        sh |= cm[c]
    sh &= has_t
    acc = ((n[sh] > 0).float() == yt[sh]).float().mean().item() if sh.any() else None
    return eff, acc
base_eff, base_acc = summary(None)
abl_eff, abl_acc = summary({k: U[k] for k in layers})
null = [summary({k: F.normalize(torch.randn(U.shape[1], device=dev), dim=0) for k in layers}) for _ in range(a.n_null)]
red = lambda e: {c: 1 - e[c] / base_eff[c] if abs(base_eff[c]) > 1e-6 else None for c in CELLS}
res = {"exp": "e05_truth_ablation", "setup": a.setup, "model": a.model, "ablated_layers": layers, "rule": __doc__.split("Rule")[1].strip(),
       "persona_effect": base_eff, "persona_effect_ablated": abl_eff, "reduction": red(abl_eff),
       "null_reduction": [red(e) for e, _ in null], "shared_acc_none": base_acc, "shared_acc_none_ablated": abl_acc,
       "shared_acc_none_null": [acc for _, acc in null]}
q95 = {c: torch.tensor([r[c] for r in res["null_reduction"] if r[c] is not None]).quantile(0.95).item() for c in STAKE}
res["pass"] = all((res["reduction"][c] or 0) >= 0.5 and res["reduction"][c] > q95[c] for c in STAKE) and (abl_acc is None or abl_acc >= 0.85)
res["null_reduction_q95"] = q95
print(f"[{slug}/{a.setup}] persona effect " + ", ".join(f"{c}:{base_eff[c]:+.2f}->{abl_eff[c]:+.2f}" for c in CELLS)
      + f" | reduction on stake cells " + ", ".join(f"{c}:{res['reduction'][c]:.0%} (null q95 {q95[c]:.0%})" for c in STAKE)
      + f" | shared acc none {base_acc} -> {abl_acc} | pass {res['pass']} | {time.time() - t0:.0f}s", flush=True)
json.dump(res, open(f"{a.out}/e05_{a.setup}_ablation_{slug}.json", "w"), indent=1)
