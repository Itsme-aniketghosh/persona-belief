"""Figures for REPORT.md, read straight from results/*.json.
Run: uv run --with matplotlib python figures/make_figs.py   -> figures/*.png"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
OUT = HERE
J = lambda f: json.load(open(os.path.join(R, f)))

# reference palette (dataviz skill), light mode; categorical slots in fixed order
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titlecolor": INK,
                     "legend.frameon": False, "lines.linewidth": 2})
M4 = [("Qwen2.5-7B-Instruct", "Qwen-7B"), ("Llama-3.1-8B-Instruct", "Llama-8B"), ("gemma-2-9b-it", "Gemma-9B"),
      ("OLMo-2-1124-7B-Instruct", "OLMo-7B")]
M6 = M4 + [("Qwen2.5-14B-Instruct", "Qwen-14B"), ("Qwen2.5-32B-Instruct", "Qwen-32B")]
STAKE = {"historical": ["era_belief", "modern_truth"], "partisan": ["lib_contested", "con_contested"]}
mean = lambda xs: sum(xs) / len(xs)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

# ---------- Fig 0: the two routes (schematic) ----------
def fig0():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
    def box(x, y, w, h, title, sub, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=edge, lw=1.6))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=11, color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=8.5, color=INK2)
    box(0.2, 1.6, 2.0, 1.0, "Persona", "\"You are a European\nscholar in 1500\"", INK2)
    box(3.4, 2.7, 2.6, 1.0, "Statement tokens", "\"The Sun travels around\nthe Earth once a day\"", BLUE)
    box(3.4, 0.4, 2.6, 1.0, "Answer position", "\"...true or false?\"\n+ answer slot", ORANGE)
    box(7.6, 1.6, 2.0, 1.0, "Answer", "logit(True) − logit(False)", INK2)
    def arrow(p, q, color, lw, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}"))
    arrow((2.2, 2.35), (3.4, 3.1), BLUE, 2)
    arrow((2.2, 1.85), (3.4, 0.95), ORANGE, 3.2)
    arrow((6.0, 2.9), (7.6, 2.35), GRAY, 1.4, ls=(0, (3, 3)))
    arrow((6.0, 0.95), (7.6, 1.85), ORANGE, 3.2)
    arrow((4.7, 2.7), (4.7, 1.4), GRAY, 1.4, ls=(0, (3, 3)))
    ax.text(0.2, 3.25, "rewrites the truth readout,\nclaim by claim (probe sees this)", fontsize=8.5, color=BLUE, ha="left")
    ax.text(6.25, 3.2, "weak: ablate 0–13%,\nsteer flips 6–11%", fontsize=8.5, color=INK2)
    ax.text(4.85, 2.0, "statement patch\nrecovers < 60%\nof the answer change", fontsize=8.5, color=INK2, va="center")
    ax.text(6.35, 0.35, "carries the persona's effect\n(answer-tail patching ≥ 50%)", fontsize=8.5, color=ORANGE)
    ax.text(0.2, 0.55, "persona read again\nat answer time", fontsize=8.5, color=ORANGE)
    save(fig, "fig0_two_routes.png")

# ---------- Fig 1: the persona moves the truth probe, claim by claim ----------
def fig1():
    cells = [("era_belief", "Era-believed falsehoods"), ("modern_truth", "Modern truths"), ("shared_truth", "Shared truths")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (c, title) in zip(axs, cells):
        for i, (m, lab) in enumerate(M6):
            g = J(f"e12a_historical_gate_{m}.json")
            p = g["cells"][c]["probe"][str(g["truth_readout"])]
            for key, col, dx, fill in (("d_hist", BLUE, -0.12, True), ("d_ctrl", GRAY, 0.12, False)):
                v, lo, hi = p[key]
                ax.plot([i + dx] * 2, [lo, hi], color=col, lw=2, solid_capstyle="round")
                ax.plot(i + dx, v, "o", ms=8, color=col, mfc=col if fill else SURF, mew=2)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(len(M6))); ax.set_xticklabels([l for _, l in M6], rotation=40, ha="right")
        ax.set_title(title); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("truth-probe shift vs no persona\n(logit units, 95% CI)")
    axs[0].plot([], [], "o", color=BLUE, ms=8, label="1500 scholar"); axs[0].plot([], [], "o", color=GRAY, mfc=SURF, mew=2, ms=8, label="2020 scholar (control)")
    axs[0].legend(loc="lower left", fontsize=9)
    save(fig, "fig1_probe_shift.png")

# ---------- Fig 2: where it enters: statement vs answer-tail patching ----------
def rec(f, src="hist"):
    r = J(f)
    st = STAKE[r["setup"]]
    xs = sorted(int(l) for l in r["patch"][src])
    return xs, [mean([r["patch"][src][str(l)][c]["recovery"] for c in st]) for l in xs]

def fig2():
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True)
    for ax, (m, lab) in zip(axs, M4):
        for f, col, name in ((f"e12c_historical_patch_probe_{m}.json", BLUE, "statement patch → probe"),
                             (f"e12c_historical_patch_ask_{m}.json", GRAY, "statement patch → answer"),
                             (f"e12c_historical_patch_ask_suffix_{m}.json", ORANGE, "answer-tail patch → answer")):
            xs, ys = rec(f)
            ax.plot(xs, ys, color=col, label=name, ls=(0, (4, 2)) if col == GRAY else "-")
        ax.axhline(0.5, color=INK2, lw=1, ls=":")
        ax.set_title(lab); ax.set_xlabel("patched layer")
        ax.set_ylim(-0.15, 1.25)
    axs[0].set_ylabel("share of persona effect recovered\n(mean of the two stake cells)")
    axs[0].text(1, 0.53, "50% rule", fontsize=8, color=INK2)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    save(fig, "fig2_patching.png")

# ---------- Fig 4: faithful but unused ----------
def fig4():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(0.5, 1.15, color=AQUA, alpha=0.08, lw=0)
    ax.text(0.02, 1.08, "pre-registered: \"the answer uses it\" (ablation removes ≥ 50%)", fontsize=8.5, color=INK2)
    series = (("statement_lr", BLUE, "o", "truth probe at statement (LR)"), ("answer_lr", ORANGE, "s", "truth probe at answer position (LR)"),
              ("statement_md", AQUA, "^", "truth probe at statement (mean diff.)"))
    for key, col, mk, name in series:
        xs, ys = [], []
        for m, lab in M4:
            for st, cells in STAKE.items():
                s = J(f"e19_answer_position_{m}.json")["setups"][st]
                xs.append(s["probes"][key]["corr_probe_vs_answer_shift"])
                ys.append(mean([s["ablation"][key]["reduction"][c] for c in cells]))
        ax.scatter(xs, ys, s=64, color=col, marker=mk, edgecolor=SURF, linewidth=1.5, label=name, zorder=3)
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xlabel("how well the probe tracks the answer\n(correlation of probe shift with answer shift, per claim)")
    ax.set_ylabel("share of persona effect on the answer\nremoved by ablating the probe direction")
    ax.set_xlim(-0.35, 1.02); ax.set_ylim(-0.4, 1.15)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.annotate("tracks the answer at r ≈ 0.9,\nremoving it changes almost nothing", xy=(0.9, 0.03), xytext=(-0.32, 0.3),
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    save(fig, "fig4_faithful_unused.png")

# ---------- Fig 3: INLP: the probe breaks, the answer doesn't ----------
def fig3():
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True)
    for ax, (m, lab) in zip(axs, M4):
        r = J(f"e15_truth_causal_{m}.json")
        pa = {int(k): v for k, v in r["inlp_heldout_probe_acc_at_readout"].items()}
        ks = sorted(int(k) for k in r["inlp_grid"])
        sd0 = r["baseline"]["logit_sd"] if "logit_sd" in r["baseline"] else r["inlp_grid"][str(ks[0])]["logit_sd"]
        bk = r["setups"]["historical"]["by_k"]
        rem = [1 - mean([bk[str(k)]["reduction"][c] for c in STAKE["historical"]]) for k in ks]
        ax.plot(sorted(pa), [pa[k] for k in sorted(pa)], color=BLUE, marker="o", ms=5, label="held-out truth-probe accuracy")
        ax.plot(ks, [r["inlp_grid"][str(k)]["acc"] for k in ks], color=ORANGE, marker="s", ms=5, label="model's answer accuracy (no persona)")
        ax.plot(ks, rem, color=AQUA, marker="^", ms=5, label="persona effect remaining")
        ax.plot(ks, [r["inlp_grid"][str(k)]["logit_sd"] / sd0 for k in ks], color=GRAY, ls=(0, (4, 2)), label="answer confidence remaining (logit sd)")
        ax.axhline(0.5, color=INK2, lw=1, ls=":")
        ax.set_xscale("symlog", linthresh=1); ax.set_xticks([0, 1, 4, 16, 64]); ax.set_xticklabels(["0", "1", "4", "16", "64"])
        ax.set_title(lab); ax.set_ylim(0, 1.3); ax.set_xlim(-0.15, 90)
    axs[0].set_ylabel("proportion")
    fig.supxlabel("truth directions removed per layer (INLP, from the first gated layer)", fontsize=10, color=INK2, y=-0.04)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.16))
    save(fig, "fig3_inlp.png")

# ---------- Fig 5: free-text answers ----------
def fig5():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars = (("recovery_stmt_patch", GRAY, "statement patch recovers"), ("recovery_tail_patch", ORANGE, "answer-tail patch recovers"),
            ("ablation_reduction", BLUE, "truth ablation removes"))
    w = 0.26
    for j, (key, col, name) in enumerate(bars):
        ys = []
        for m, _ in M4:
            C = J(f"e22_generation_historical_{m}.json")["cells"]
            ys.append(mean([C[c][key] or 0 for c in STAKE["historical"]]))
        ax.bar([i + (j - 1) * w for i in range(len(M4))], ys, width=w - 0.03, color=col, label=name)
    ax.axhline(0.5, color=INK2, lw=1, ls=":"); ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(M4))); ax.set_xticklabels([l for _, l in M4]); ax.grid(axis="x", visible=False)
    ax.set_ylabel("share of the persona's change in\n\"true\" verdicts (free text)")
    ax.legend(loc="upper right", fontsize=8.5, ncol=1)
    save(fig, "fig5_free_text.png")

for f in (fig0, fig1, fig2, fig3, fig4, fig5):
    f()
