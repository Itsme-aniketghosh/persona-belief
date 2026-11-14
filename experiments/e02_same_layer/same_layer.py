"""E2: steer at layer l, read the probe at the same layer l. Measured Δs must equal α(w·v); any mismatch is a hook bug.
Downstream readouts (l' > l) come free from the same forward pass and are saved for E3, not interpreted here."""
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
p.add_argument("--layers", type=int, nargs="+", default=list(range(17, 29)))  # probe >= 85%
p.add_argument("--alphas", type=float, nargs="+", default=[-2.0, -1.0, 1.0, 2.0])
p.add_argument("--n_null", type=int, default=20)
p.add_argument("--n", type=int, default=128, help="statements")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default=os.path.expanduser("~/pb/data/got"))
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
torch.manual_seed(a.seed)
slug = a.model.split("/")[-1]
e0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e00_baselines")
DS = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "larger_than", "smaller_than"]
texts = pd.concat([pd.read_csv(f"{a.data}/{d}.csv").sample(a.n // len(DS), random_state=a.seed + 1)
                   for d in DS]).statement.tolist()

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
P = torch.load(f"{e0}/probes_{slug}.pt")["lr"].cuda()  # [L+1, d+1]
w, b = P[:, :-1], P[:, -1]
V = {t: torch.load(f"{e0}/pv_{t}_{slug}.pt").cuda() for t in a.traits}
d = w.shape[1]
mean_norm = torch.stack([v.norm(dim=1) for v in V.values()]).mean(0)  # null norm per layer
V.update({f"rand{i}": F.normalize(torch.randn(*w.shape, device="cuda"), dim=1) * mean_norm[:, None]
          for i in range(a.n_null)})

t0 = time.time()
s_of = lambda H: (H.cuda() * w).sum(-1) + b  # [n, L+1] probe reading at every layer
H0 = last_token_acts(model, tok, texts)
s0 = s_of(H0)
rows = []
for name, v in V.items():
    for l in a.layers:
        for al in a.alphas:
            vec = (al * v[l]).to(torch.bfloat16)
            with steer(model, l, vec):
                ds = s_of(last_token_acts(model, tok, texts)) - s0  # [n, L+1]
            pred = (w[l] @ vec.float()).item()  # α(w·v), exact for the bf16 vector actually added
            meas = ds[:, l]
            gap, se = abs(meas.mean().item() - pred), (meas.std() / len(meas) ** 0.5).item()
            ok = gap <= 0.02 * abs(pred) + 3 * se  # se absorbs bf16 rounding of h + αv
            rows.append({"vec": name, "layer": l, "alpha": al, "pred": pred, "meas_mean": meas.mean().item(),
                         "meas_sd": meas.std().item(), "rel_gap": gap / max(abs(pred), 1e-6), "ok": ok,
                         "ds_mean_by_readout_layer": ds.mean(0).tolist()})
    if not name.startswith("rand"):
        r = [x for x in rows if x["vec"] == name]
        r20 = [x for x in r if x["layer"] == 20]
        print(f"{name}: ok {sum(x['ok'] for x in r)}/{len(r)}, max rel gap {max(x['rel_gap'] for x in r):.4f} | "
              f"l=20 Δs/α meas {[round(x['meas_mean'] / x['alpha'], 3) for x in r20]} pred {r20[0]['pred'] / r20[0]['alpha']:.3f}",
              flush=True)
n_ok = sum(x["ok"] for x in rows)
rr = [x for x in rows if x["vec"].startswith("rand")]
print(f"all: ok {n_ok}/{len(rows)}; random: ok {sum(x['ok'] for x in rr)}/{len(rr)}; {time.time() - t0:.0f}s")
json.dump({"exp": "e02_same_layer", "model": a.model, "layers": a.layers, "alphas": a.alphas, "n_statements": len(texts),
           "n_null": a.n_null, "seed": a.seed, "probe": "lr from e00", "null_norm": "mean persona norm per layer",
           "pass_rule": "|mean Δs - α(w·v)| <= 2% of prediction + 3 SE across statements, every (vector, layer, alpha)",
           "n_ok": n_ok, "n_total": len(rows), "pass": n_ok == len(rows), "rows": rows},
          open(f"{a.out}/e02_same_layer_{slug}.json", "w"), indent=1)
