"""Post figures from saved E12a/E12c results (run locally: uv run --with matplotlib python experiments/e12_historical/figures.py)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "..", "results")
MODELS = [("Qwen2.5-7B-Instruct", "Qwen2.5-7B"), ("Llama-3.1-8B-Instruct", "Llama-3.1-8B")]
CELLS = [("era_belief", "Era-believed falsehoods", "#2a78d6"), ("modern_truth", "Modern truths", "#eb6834"),
         ("shared_truth", "Shared truths", "#1baf7a"), ("shared_false", "Shared falsehoods", "#eda100")]
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"

def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)

# Figure A: prompt effect per cell (E12a), 1500 scholar vs 2020 control, 95% CI
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, facecolor=SURF)
for ax, (slug, name) in zip(axes, MODELS):
    r = json.load(open(f"{R}/e12a_historical_gate_{slug}.json"))
    l = str(max(r["gate_truth_layers"], key=lambda k: r["truth_acc"][k]))
    for i, (c, lab, col) in enumerate(CELLS):
        for j, (key, alpha) in enumerate((("d_hist", 1.0), ("d_ctrl", 0.35))):
            m, lo, hi = r["cells"][c]["probe"][l][key]
            ax.bar(i + (j - 0.5) * 0.38, m, 0.36, color=col, alpha=alpha, edgecolor=SURF, lw=2)
            ax.errorbar(i + (j - 0.5) * 0.38, m, [[m - lo], [hi - m]], color=INK, lw=1, capsize=2)
    ax.set_xticks(range(4), [lab.replace(" ", "\n", 1) for _, lab, _ in CELLS], fontsize=8)
    ax.set_title(f"{name} (truth probe, layer {l})", fontsize=10)
    style(ax)
axes[0].set_ylabel("Truth-probe shift vs no prompt (logits)")
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(color=MUTED, label="1500-scholar prompt (solid)"),
                        Patch(color=MUTED, alpha=0.35, label="2020-scholar control (faded)")],
               frameon=False, fontsize=8, loc="lower left")
fig.suptitle("A persona prompt moves the truth readout only on claims the persona has a stake in", fontsize=11)
fig.tight_layout(); fig.savefig(f"{R}/e12a_prompt_effect.png", dpi=150)

# Figure B: patching statement tokens layer by layer (E12c)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURF)
for ax, (slug, name) in zip(axes, MODELS):
    r = json.load(open(f"{R}/e12c_historical_patch_{slug}.json"))
    lt = r["truth_readout"]
    L = list(range(1, lt + 1))
    for c, lab, col in CELLS:
        ax.plot(L, [r["patch"]["hist"][str(l)][c]["d_truth"][lt] for l in L], color=col, lw=2, label=lab)
        if c in ("era_belief", "modern_truth"):
            ax.plot(L, [r["patch"]["ctrl"][str(l)][c]["d_truth"][lt] for l in L], color=col, lw=1.2, ls="--", alpha=0.6)
    ax.set_xlabel(f"Layer patched (statement tokens only); readout at layer {lt}")
    ax.set_title(name, fontsize=10)
    style(ax)
axes[0].set_ylabel("Truth-probe shift at readout (logits)")
axes[1].plot([], [], color=MUTED, ls="--", lw=1.2, label="2020-control patch (era cells)")
axes[1].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("Patching 1500-scholar activations into the statement tokens reproduces the effect from a mid layer on", fontsize=11)
fig.tight_layout(); fig.savefig(f"{R}/e12c_patching.png", dpi=150)
print("saved e12a_prompt_effect.png, e12c_patching.png")
