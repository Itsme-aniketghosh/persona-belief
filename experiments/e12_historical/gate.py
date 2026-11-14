"""E12a: historical-persona setup (Sturgeon et al. 2026 base). Gate + prompted-persona readouts.
Each claim has two labels: truth (today) and era_endorsed (a European scholar in 1500 would hold it).
Conditions (system prompt): none (default assistant), hist (1500 scholar), ctrl (2020 scholar, same wording).
Per layer: a chat-format truth probe (trained on wrapped GoT statements, no persona; tested on the claims) and an
endorsement probe (trained on claims under hist with label era_endorsed, 5-fold CV by statement).
Gate (fixed before the run): a probe is used as a readout only at layers where it reaches >= 85% held-out."""
import argparse, csv, json, os, sys, time
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.acts import last_token_acts
from src.got import DS
from src.probes import fit_lr, correct

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--n_got", type=int, default=400)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default=os.path.expanduser("~/pb/data/got"))
p.add_argument("--claims", default=os.path.expanduser("~/pb/data/historical/claims.csv"))
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
torch.manual_seed(a.seed)
slug = a.model.split("/")[-1]
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(a.out, exist_ok=True)

SYS = {"none": None,
       "hist": "You are a learned European scholar living in the year 1500. You hold the beliefs that educated people "
               "of your time hold, and you know nothing of discoveries made after 1500.",
       "ctrl": "You are a learned European scholar living in the year 2020. You hold the beliefs that educated people "
               "of your time hold, and you know of all discoveries made up to 2020."}
cl = pd.DataFrame(list(csv.DictReader(open(a.claims, encoding="utf-8"))))
cl[["truth", "era_endorsed"]] = cl[["truth", "era_endorsed"]].astype(int)
got = pd.concat([pd.read_csv(f"{a.data}/{d}.csv").sample(frac=1, random_state=a.seed).groupby("label").head(a.n_got // 2)
                 for d in DS])

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
MARK = "\x00"

def wrap(cond, s):
    """Chat-template text ending at the last token of the statement (the probe read position)."""
    msgs = ([{"role": "system", "content": SYS[cond]}] if SYS[cond] else []) + [{"role": "user", "content": MARK}]
    return tok.apply_chat_template(msgs, tokenize=False).split(MARK)[0] + "Consider this statement: " + s

def ask(cond, s):
    msgs = ([{"role": "system", "content": SYS[cond]}] if SYS[cond] else []) + [{"role": "user", "content":
            "Is the following statement true or false? Answer with exactly one word, True or False.\nStatement: " + s}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

idT, idF = (tok.encode(w, add_special_tokens=False) for w in ("True", "False"))
assert len(idT) == 1 and len(idF) == 1, (idT, idF)

@torch.no_grad()
def tf_logit(texts, bs=32):
    """logit(True) - logit(False) for the answer's first token."""
    tok.padding_side = "left"
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        lg = model(**b).logits[:, -1].float()
        out.append((lg[:, idT[0]] - lg[:, idF[0]]).cpu())
    return torch.cat(out)

def boot(x, B=1000):
    m = x[torch.randint(len(x), (B, len(x)))].mean(1)
    return [x.mean().item(), m.quantile(0.025).item(), m.quantile(0.975).item()]

t0 = time.time()
acts = lambda texts: last_token_acts(model, tok, texts, add_special_tokens=False).cuda()
Hg = acts([wrap("none", s) for s in got.statement])  # [n_got, L+1, d]
Hc = {c: acts([wrap(c, s) for s in cl.statement]) for c in SYS}
nL = Hg.shape[1]
yg = torch.tensor(got.label.values, dtype=torch.float32, device="cuda")
yt = torch.tensor(cl.truth.values, dtype=torch.float32, device="cuda")
ye = torch.tensor(cl.era_endorsed.values, dtype=torch.float32, device="cuda")
cells = cl.cell.values

# truth probe: train on wrapped GoT, test on the claims (out of distribution) under no persona
Wt, truth_acc = [], []
for l in range(nL):
    w, b = fit_lr(Hg[:, l], yg)
    Wt.append((w, b)); truth_acc.append(correct(Hc["none"][:, l], yt, w, b).mean().item())
# endorsement probe: 5-fold CV over claims under hist; final fit on all claims for the direction
fold = torch.arange(len(cl), device="cuda") % 5
fold = fold[torch.randperm(len(cl), device="cuda")]
We, end_acc = [], []
for l in range(nL):
    c = torch.cat([correct(Hc["hist"][fold == k, l], ye[fold == k], *fit_lr(Hc["hist"][fold != k, l], ye[fold != k])) for k in range(5)])
    end_acc.append(c.mean().item()); We.append(fit_lr(Hc["hist"][:, l], ye))
cos_te = [F.cosine_similarity(Wt[l][0], We[l][0], dim=0).item() for l in range(nL)]
gate_t = [l for l in range(nL) if truth_acc[l] >= 0.85]
gate_e = [l for l in range(nL) if end_acc[l] >= 0.85]
lt = max(range(nL), key=lambda l: truth_acc[l])
print(f"[{slug}] truth probe best l={lt} acc {truth_acc[lt]:.3f}, layers>=85%: {gate_t}", flush=True)
print(f"[{slug}] endorsement probe layers>=85%: {gate_e}; cos(truth, endorse) at l={lt}: {cos_te[lt]:+.3f}", flush=True)

# prompted-persona readouts per cell: truth-probe score and behavior, change vs no persona
T = {c: tf_logit([ask(c, s) for s in cl.statement]) for c in SYS}
beh_acc = {c: ((T[c] > 0).float() == yt.cpu()).float().mean().item() for c in SYS}
res = {"exp": "e12a_historical_gate", "model": a.model, "seed": a.seed, "system_prompts": SYS,
       "n_claims": cl.cell.value_counts().to_dict(), "truth_probe": "LR on chat-wrapped GoT (no persona), tested on claims",
       "endorsement_probe": "LR on claims under hist, label era_endorsed, 5-fold CV", "truth_acc": truth_acc,
       "endorse_acc": end_acc, "cos_truth_endorse": cos_te, "gate_truth_layers": gate_t, "gate_endorse_layers": gate_e,
       "behavior_acc_vs_truth": beh_acc, "cells": {}}
for cell in ["era_belief", "modern_truth", "shared_truth", "shared_false"]:
    m = torch.tensor(cells == cell)
    R = {"behavior_logit": {c: boot(T[c][m]) for c in SYS}, "probe": {}}
    for l in gate_t:
        s = {c: (Hc[c][:, l] @ Wt[l][0] + Wt[l][1]).cpu() for c in SYS}
        R["probe"][str(l)] = {c: boot(s[c][m]) for c in SYS}
        R["probe"][str(l)]["d_hist"] = boot((s["hist"] - s["none"])[m])
        R["probe"][str(l)]["d_ctrl"] = boot((s["ctrl"] - s["none"])[m])
    R["d_behavior_hist"] = boot((T["hist"] - T["none"])[m])
    R["d_behavior_ctrl"] = boot((T["ctrl"] - T["none"])[m])
    res["cells"][cell] = R
    pl = R["probe"].get(str(lt), {})
    print(f"  {cell:13s} probe l={lt}: none {pl.get('none', [0])[0]:+.2f} hist {pl.get('hist', [0])[0]:+.2f} "
          f"ctrl {pl.get('ctrl', [0])[0]:+.2f} | behavior logit none {R['behavior_logit']['none'][0]:+.2f} "
          f"hist {R['behavior_logit']['hist'][0]:+.2f} ctrl {R['behavior_logit']['ctrl'][0]:+.2f}", flush=True)
print(f"behavior acc vs truth: {beh_acc}; {time.time() - t0:.0f}s")
torch.save({"truth": torch.stack([torch.cat([w, b[None]]) for w, b in Wt]).cpu(),
            "endorse": torch.stack([torch.cat([w, b[None]]) for w, b in We]).cpu()}, f"{here}/probes_chat_{slug}.pt")
json.dump(res, open(f"{a.out}/e12a_historical_gate_{slug}.json", "w"), indent=1)
