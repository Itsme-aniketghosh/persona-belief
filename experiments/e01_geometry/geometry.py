"""E1: cos(w_l, v_l) per trait and layer vs. cos(w_l, random) for 1000 isotropic random directions."""
import argparse, json, os
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--traits", nargs="+", default=["evil", "sycophantic", "hallucinating"])
p.add_argument("--n_null", type=int, default=1000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
torch.manual_seed(a.seed)
slug = a.model.split("/")[-1]
e0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e00_baselines")
P = torch.load(f"{e0}/probes_{slug}.pt")  # {"lr","md"}: [L+1, d+1], last col = bias
V = {t: torch.load(f"{e0}/pv_{t}_{slug}.pt") for t in a.traits}  # [L+1, d]
acc = json.load(open(f"{a.out}/e00_probes_{slug}.json"))["results"]
nL, d = V[a.traits[0]].shape
R = F.normalize(torch.randn(a.n_null, d), dim=1)

res = {"exp": "e01_geometry", "model": a.model, "seed": a.seed, "n_null": a.n_null, "d_model": d,
       "null": "isotropic Gaussian directions; ignores activation-space anisotropy", "probe": {}}
for m in ("lr", "md"):
    W = F.normalize(P[m][:, :-1], dim=1)
    null = W @ R.T  # [L+1, n_null]
    lo, hi = null.quantile(0.025, dim=1), null.quantile(0.975, dim=1)
    r = {"null_lo": lo.tolist(), "null_hi": hi.tolist(), "probe_acc": acc[m]["acc"], "cos": {}, "outside_null": {}}
    for t in a.traits:
        c = (W * F.normalize(V[t], dim=1)).sum(1)
        r["cos"][t] = c.tolist()
        r["outside_null"][t] = [l for l in range(nL) if (c[l] < lo[l] or c[l] > hi[l]) and acc[m]["acc"][l] >= 0.85]
        print(f"{m} {t}: max|cos| over probe-good layers = "
              f"{max(abs(c[l].item()) for l in range(nL) if acc[m]['acc'][l] >= 0.85):.3f}, "
              f"null 95% ~ [{lo.mean():.3f}, {hi.mean():.3f}], outside null at layers {r['outside_null'][t]}")
    res["probe"][m] = r
json.dump(res, open(f"{a.out}/e01_geometry_{slug}.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
cols = {"evil": "#2a78d6", "sycophantic": "#eb6834", "hallucinating": "#1baf7a"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, facecolor="#fcfcfb")
for ax, (m, name) in zip(axes, (("lr", "Logistic-regression probe"), ("md", "Mean-difference probe"))):
    r = res["probe"][m]
    ax.set_facecolor("#fcfcfb")
    ax.fill_between(range(nL), r["null_lo"], r["null_hi"], color="#c3c2b7", alpha=0.5, lw=0, label="Random 95%")
    good = [l for l in range(nL) if r["probe_acc"][l] >= 0.85]
    ax.axvspan(good[0] - 0.5, good[-1] + 0.5, color="#e8e7e3", alpha=0.5, lw=0)
    for t in a.traits:
        ax.plot(range(nL), r["cos"][t], color=cols.get(t, "#6250d6"), lw=2, label=t)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set(xlabel="Layer (shaded: probe >= 85%)", title=name)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("cos(w, v)")
axes[1].legend(frameon=False, fontsize=8)
fig.suptitle(f"Truth direction vs. persona vectors, {slug}", fontsize=11)
fig.tight_layout(); fig.savefig(f"{a.out}/e01_geometry_{slug}.png", dpi=150)
