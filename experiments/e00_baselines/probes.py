"""E0a: truth probes at every layer, held out by dataset (leave-one-dataset-out)."""
import argparse, json, os, sys, time
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.acts import last_token_acts
from src.probes import fit_lr, fit_md, correct

DATASETS = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "larger_than", "smaller_than"]
p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--n", type=int, default=400, help="max rows per dataset, label-balanced")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default=os.path.expanduser("~/pb/data/got"))
p.add_argument("--cache", default=f"/scratch/{os.environ['USER']}/persona-belief/acts")
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
torch.manual_seed(a.seed)
slug = a.model.split("/")[-1]
os.makedirs(a.cache, exist_ok=True); os.makedirs(a.out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))

df = pd.concat([pd.read_csv(f"{a.data}/{d}.csv").sample(frac=1, random_state=a.seed)
                .groupby("label").head(a.n // 2).assign(ds=d) for d in DATASETS], ignore_index=True)
cache = f"{a.cache}/got_{slug}_n{a.n}_s{a.seed}_hooks.pt"  # _hooks: read via src.acts hooks, last layer pre-norm
t0 = time.time()
if os.path.exists(cache):
    X = torch.load(cache)
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    X = last_token_acts(model, tok, df.statement.tolist())
    torch.save(X.half(), cache)  # reused by E1/E2
    del model; torch.cuda.empty_cache()
X = X.float().cuda()
y = torch.tensor(df.label.values, dtype=torch.float32, device="cuda")
ds = df.ds.values
print(f"acts {tuple(X.shape)} in {time.time() - t0:.0f}s", flush=True)

def boot_ci(c, B=1000):
    idx = torch.randint(len(c), (B, len(c)), device=c.device)
    m = c[idx].mean(1)
    return [m.quantile(0.025).item(), m.quantile(0.975).item()]

res = {m: {"acc": [], "ci": [], "fold": []} for m in ("lr", "md")}
W = {m: [] for m in ("lr", "md")}
for l in range(X.shape[1]):
    Xl = X[:, l]
    for m, fit in (("lr", fit_lr), ("md", fit_md)):
        cs, folds = [], {}
        for d in DATASETS:
            te = torch.tensor(ds == d, device="cuda")
            c = correct(Xl[te], y[te], *fit(Xl[~te], y[~te]))
            cs.append(c); folds[d] = c.mean().item()
        c = torch.cat(cs)
        res[m]["acc"].append(c.mean().item()); res[m]["ci"].append(boot_ci(c)); res[m]["fold"].append(folds)
        w, b = fit(Xl, y)  # final readout: trained on all datasets
        W[m].append(torch.cat([w, b[None]]).cpu())
    print(f"L{l:2d} lr {res['lr']['acc'][-1]:.3f} md {res['md']['acc'][-1]:.3f} "
          f"min-fold lr {min(res['lr']['fold'][-1].values()):.3f}", flush=True)

torch.save({m: torch.stack(v) for m, v in W.items()}, f"{here}/probes_{slug}.pt")  # [L+1, d+1], last col = bias
good = {m: [l for l, x in enumerate(res[m]["acc"]) if x >= 0.85] for m in res}
json.dump({"exp": "e00_probes", "model": a.model, "n_layers": X.shape[1] - 1, "d_model": X.shape[2],
           "n_rows": len(df), "rows_per_dataset": df.ds.value_counts().to_dict(), "seed": a.seed,
           "holdout": "leave-one-dataset-out, pooled held-out accuracy, 95% bootstrap CI over rows",
           "layers_ge_85": good, "results": res, "minutes": (time.time() - t0) / 60},
          open(f"{a.out}/e00_probes_{slug}.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.6), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
L = range(X.shape[1])
for m, col, name in (("lr", "#2a78d6", "Logistic regression"), ("md", "#eb6834", "Mean difference")):
    lo, hi = zip(*res[m]["ci"])
    ax.fill_between(L, lo, hi, color=col, alpha=0.18, lw=0)
    ax.plot(L, res[m]["acc"], color=col, lw=2, label=name)
ax.axhline(0.85, color="#52514e", lw=1, ls="--"); ax.text(0, 0.87, "85% gate", color="#52514e", fontsize=8)
ax.text(0.5, 0.52, "chance", color="#52514e", fontsize=8)
ax.axhline(0.5, color="#c3c2b7", lw=1)
ax.set(xlabel="Layer (0 = embeddings)", ylabel="Held-out accuracy", ylim=(0, 1.0),
       title=f"Truth probe accuracy by layer, {slug} (leave-one-dataset-out)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e8e7e3", lw=0.8); ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{a.out}/e00_probe_acc_{slug}.png", dpi=150)
print("layers >= 0.85:", good)
