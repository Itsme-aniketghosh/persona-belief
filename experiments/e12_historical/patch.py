"""E12c patching: where does the persona's effect on the truth readout enter the statement tokens?
For each layer l: run the no-prompt text, but overwrite the residual stream (output of block l) at the statement
tokens with the 1500-scholar run's values (or the 2020-scholar run's, as control). Read the truth and endorsement
probes at the last token. Recovery = (patched - none) / (prompt - none), per claim cell.
Rule (fixed before the run): some layer at or before the truth readout recovers >= 50% of the prompt's effect on both
era cells (era_belief, modern_truth), while the shared cells move less than half as much as the era cells."""
import argparse, csv, json, os, sys, time
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--max_claims", type=int, default=None, help="tiny pass")
p.add_argument("--claims", default=os.path.expanduser("~/pb/data/historical/claims.csv"))
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
p.add_argument("--e0_results", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
slug = a.model.split("/")[-1]
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(a.out, exist_ok=True)

SYS = {"none": None,  # same prompts as gate.py (E12a)
       "hist": "You are a learned European scholar living in the year 1500. You hold the beliefs that educated people "
               "of your time hold, and you know nothing of discoveries made after 1500.",
       "ctrl": "You are a learned European scholar living in the year 2020. You hold the beliefs that educated people "
               "of your time hold, and you know of all discoveries made up to 2020."}
cl = pd.DataFrame(list(csv.DictReader(open(a.claims, encoding="utf-8"))))
if a.max_claims:
    cl = cl.groupby("cell").head(a.max_claims // 4)
cells = cl.cell.values
CELLS = ["era_belief", "modern_truth", "shared_truth", "shared_false"]

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"  # statement tokens sit at the end of every row, aligned across conditions
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
dev = model.device
MARK = "\x00"
SUF = "Consider this statement: "

def wrap(cond, s):
    msgs = ([{"role": "system", "content": SYS[cond]}] if SYS[cond] else []) + [{"role": "user", "content": MARK}]
    return tok.apply_chat_template(msgs, tokenize=False).split(MARK)[0] + SUF + s

PR = torch.load(f"{here}/probes_chat_{slug}.pt")
WT, BT = PR["truth"][:, :-1].cuda(), PR["truth"][:, -1].cuda()
WE, BE = PR["endorse"][:, :-1].cuda(), PR["endorse"][:, -1].cuda()
gate = json.load(open(f"{a.e0_results}/e12a_historical_gate_{slug}.json"))
lt = max(gate["gate_truth_layers"], key=lambda l: gate["truth_acc"][l])  # best truth readout layer
nL = WT.shape[0]
K = [len(tok(SUF + s, add_special_tokens=False).input_ids) for s in cl.statement]  # statement-token span per claim

def run(cond, idx, patch=None):
    """Last-token truth/endorse scores at every layer [B, L+1]; optionally capture block-l output at the last Kmax
    tokens (patch=("cap", l)) or overwrite it (patch=("put", l, src, mask))."""
    b = tok([wrap(cond, cl.statement.iloc[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    hs, got, cap = [], {}, {}
    mods = [model.model.embed_tokens] + list(model.model.layers)

    def grab(j):
        def hook(mod, inp, o):
            got[j] = (o[0] if isinstance(o, tuple) else o)[:, -1].float()
        return hook

    for j, m in enumerate(mods):
        if patch and patch[1] == j and j > 0:
            if patch[0] == "cap":
                def hk(mod, inp, o, Km=patch[2]):
                    cap["h"] = (o[0] if isinstance(o, tuple) else o)[:, -Km:].clone()
            else:
                def hk(mod, inp, o, src=patch[2], mask=patch[3]):
                    h = (o[0] if isinstance(o, tuple) else o).clone()
                    Km = src.shape[1]
                    h[:, -Km:] = torch.where(mask[..., None], src, h[:, -Km:])
                    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            hs.append(m.register_forward_hook(hk))
        hs.append(m.register_forward_hook(grab(j)))  # registered after the patch hook, so it reads patched values
    try:
        with torch.no_grad():
            model.model(**b)
    finally:
        for h in hs:
            h.remove()
    H = torch.stack([got[j] for j in range(len(mods))], 1)
    return (H * WT).sum(-1) + BT, (H * WE).sum(-1) + BE, cap.get("h"), b.input_ids

t0 = time.time()
bs = 32
batches = [list(range(i, min(i + bs, len(cl)))) for i in range(0, len(cl), bs)]
base = {c: [run(c, ix) for ix in batches] for c in SYS}
for ix, (_, _, _, idn), (_, _, _, idh) in zip(batches, base["none"], base["hist"]):  # self-check: identical statement tokens
    for r, i in enumerate(ix):
        assert torch.equal(idn[r, -K[i]:], idh[r, -K[i]:]), f"statement tokens differ for claim {i}"
S = {c: (torch.cat([x[0] for x in base[c]]), torch.cat([x[1] for x in base[c]])) for c in SYS}
res = {"exp": "e12c_historical_patch", "model": a.model, "truth_readout": lt, "rule": __doc__.split("Rule")[1].strip(),
       "prompt_effect": {}, "patch": {"hist": {}, "ctrl": {}}}
cm = {c: torch.tensor(cells == c, device=dev) for c in CELLS}
for src_cond in ("hist", "ctrl"):
    for c in CELLS:
        res["prompt_effect"].setdefault(src_cond, {})[c] = {
            "truth": (S[src_cond][0] - S["none"][0])[cm[c]].mean(0).tolist(),
            "endorse": (S[src_cond][1] - S["none"][1])[cm[c]].mean(0).tolist()}
for l in range(1, nL):
    for src_cond in ("hist", "ctrl"):
        Pt, Pe = [], []
        for ix in batches:
            Km = max(K[i] for i in ix)
            _, _, src, _ = run(src_cond, ix, ("cap", l, Km))
            mask = torch.stack([torch.arange(Km, device=dev) >= Km - K[i] for i in ix])  # this claim's statement tokens
            st, se, _, _ = run("none", ix, ("put", l, src, mask))
            Pt.append(st); Pe.append(se)
        Pt, Pe = torch.cat(Pt), torch.cat(Pe)
        ent = {}
        for c in CELLS:
            m = cm[c]
            dT = (Pt - S["none"][0])[m].mean(0)
            dE = (Pe - S["none"][1])[m].mean(0)
            full = (S[src_cond][0] - S["none"][0])[m].mean(0)
            ent[c] = {"d_truth": dT.tolist(), "d_endorse": dE.tolist(),
                      "recovery_truth_at_readout": (dT[lt] / full[lt]).item() if full[lt].abs() > 1e-6 else None}
        if l > lt:  # self-check: a patch after the readout layer cannot change the readout
            assert all(abs(ent[c]["d_truth"][lt]) < 1e-2 for c in CELLS), f"patch at {l} changed readout {lt}"  # bf16 run-to-run slack
        res["patch"][src_cond][str(l)] = ent
    e = res["patch"]["hist"][str(l)]
    if l <= lt:
        print(f"[{slug}] patch hist@{l:2d} -> truth@{lt}: " + ", ".join(
            f"{c}:{e[c]['d_truth'][lt]:+.2f}({e[c]['recovery_truth_at_readout'] or 0:.0%})" for c in CELLS)
            + f" | {time.time() - t0:.0f}s", flush=True)
json.dump(res, open(f"{a.out}/e12c_historical_patch_{slug}.json", "w"))
print(f"done in {(time.time() - t0) / 60:.1f} min")
