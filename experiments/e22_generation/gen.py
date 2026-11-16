"""E22: do the routing results hold for free-text answers, not just logit(True)-logit(False)?
Greedy generation (24 tokens) for "Is the following statement true or false? Answer in one short sentence." with the
no-persona and persona prompts; the first of true/false/correct/incorrect/accurate/inaccurate in the reply is the verdict.
Interventions (prompt-processing pass only, so generated tokens are untouched except by the ablation):
  statement patch: persona run's residuals written into the no-persona run at the statement tokens, at the layer where
    statement patching reproduced the probe shift (first rule-pass layer of the E12c probe run, else the readout);
  tail patch: same at the tokens after the statement, at the first rule-pass layer of the E16a tail run, else readout;
  truth ablation: the chat truth direction projected out at every layer from the first gated layer, all positions,
    during the whole generation, in both runs.
Rules (fixed before the run): (1) the persona moves the share of "true" verdicts up on era-believed falsehoods and down
on modern truths, 95% CIs excluding 0; (2) the statement patch recovers < 50% and the tail patch >= 50% of that change
on both stake cells; (3) the truth ablation removes < 50%. All three -> consistent with the logit-based routing."""
import argparse, json, os, re, sys, time
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src import setup as S

ASK_GEN = "Is the following statement true or false? Answer in one short sentence.\nStatement: "
VERD = {"true": 1, "correct": 1, "accurate": 1, "false": 0, "incorrect": 0, "inaccurate": 0}
WORD = re.compile(r"\b(true|false|correct|incorrect|accurate|inaccurate)\b")

def verdict(text):
    t = text.lower()
    m = WORD.search(t)
    if not m:
        return -1
    v = VERD[m.group(1)]
    pre = t[max(0, m.start() - 12):m.start()]
    return 1 - v if re.search(r"\b(not|n't)\s*$", pre) else v

assert [verdict(x) for x in ("False. The Earth orbits the Sun.", "This is not true.", "It is correct that...", "Hmm.")] == [0, 0, 1, -1]

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--setup", default="historical")
p.add_argument("--max_new", type=int, default=24)
p.add_argument("--max_claims", type=int, default=None, help="tiny pass")
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
torch.manual_seed(0)
slug = a.model.split("/")[-1]
here = os.path.dirname(os.path.abspath(__file__))
e12 = os.path.join(here, "..", "e12_historical")
R0 = os.path.expanduser("~/pb/results")
cfg, cl = S.load(a.setup)
if a.max_claims:
    cl = cl.groupby("cell").head(a.max_claims // 4)
cl = cl.reset_index(drop=True)
CELLS, STAKE = cfg["cells"], cfg["stake_cells"]
tag = S.file_tag(a.setup)

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
dev = model.device
LAY = model.model.layers
_, ask, span = S.formats(tok, cfg["conditions"], ask_text=ASK_GEN)
gate = json.load(open(f"{R0}/e12a_{a.setup}_gate_{slug}.json"))
lt, k0 = gate["truth_readout"], min(gate["gate_truth_layers"])
def first_pass(f, default):
    try:
        pl = json.load(open(f"{R0}/{f}"))["rule_pass_layers"]["hist"]
        return pl[0] if pl else default
    except FileNotFoundError:
        return default
l_stmt = first_pass(f"e12c_{a.setup}_patch_probe_{slug}.json", lt)
l_tail = first_pass(f"e12c_{a.setup}_patch_ask_suffix_{slug}.json", lt)
U = torch.nn.functional.normalize(torch.load(f"{e12}/probes_chat_{tag}{slug}.pt")["truth"][:, :-1].cuda(), dim=1)

@torch.no_grad()
def gen(cond, patch=None, ablate=False, bs=16):
    """Verdicts per claim. patch: (layer, region) with region in {statement, tail}; sources from the persona run."""
    out = []
    for i in range(0, len(cl), bs):
        stm = cl.statement.iloc[i:i + bs].tolist()
        texts = [ask(cond, s) for s in stm]
        b = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        hs = []
        if ablate:
            for k in range(k0, len(LAY) + 1):
                def hk(m, inp, o, u=U[k].to(torch.bfloat16)):
                    h = o[0] if isinstance(o, tuple) else o
                    h = h - (h @ u)[..., None] * u
                    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
                hs.append(LAY[k - 1].register_forward_hook(hk))
        if patch:
            l, region = patch
            sp = [span(t, "Statement: " + s) for t, s in zip(texts, stm)]
            Km = max(e1 for _, e1 in sp)
            bh = tok([ask("hist", s) for s in stm], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            cap = {}
            hc = LAY[l - 1].register_forward_hook(lambda m, inp, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[:, -Km:].clone()))
            try:
                model(**bh)
            finally:
                hc.remove()
            pos = torch.arange(Km, 0, -1, device=dev)
            mask = torch.stack([((pos > e0) & (pos <= e1)) if region == "statement" else (pos <= e0) for e0, e1 in sp])
            def hp(m, inp, o, src=cap["h"], mask=mask):
                h = o[0] if isinstance(o, tuple) else o
                if h.shape[1] < Km:  # decoding step: leave generated tokens alone
                    return None
                h = h.clone()
                h[:, -Km:] = torch.where(mask[..., None], src, h[:, -Km:])
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            hs.append(LAY[l - 1].register_forward_hook(hp))
        try:
            g = model.generate(**b, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        finally:
            for h in hs:
                h.remove()
        out += [verdict(x) for x in tok.batch_decode(g[:, b.input_ids.shape[1]:], skip_special_tokens=True)]
    return torch.tensor(out, dtype=torch.float32, device=dev)

def boot(x, B=1000):
    m = x[torch.randint(len(x), (B, len(x)), device=x.device)].mean(1)
    return [x.mean().item(), m.quantile(0.025).item(), m.quantile(0.975).item()]

t0 = time.time()
cm = {c: torch.tensor(cl.cell.values == c, device=dev) for c in CELLS}
runs = {"none": gen("none"), "hist": gen("hist"), "stmt_patch": gen("none", (l_stmt, "statement")),
        "tail_patch": gen("none", (l_tail, "tail")), "none_ablate": gen("none", ablate=True), "hist_ablate": gen("hist", ablate=True)}
true_rate = lambda v, m: (v[m & (v >= 0)] == 1).float()
res = {"exp": "e22_generation", "model": a.model, "setup": a.setup, "prompt": ASK_GEN, "patch_layers": {"statement": l_stmt, "tail": l_tail},
       "ablated_from": k0, "rule": __doc__.split("Rules")[1].strip(),
       "unparsed": {k: (v < 0).float().mean().item() for k, v in runs.items()}, "cells": {}}
for c in CELLS:
    m = cm[c]
    full = true_rate(runs["hist"], m).mean() - true_rate(runs["none"], m).mean()
    E = {"true_rate": {k: boot(true_rate(v, m)) for k, v in runs.items()}, "persona_effect": full.item()}
    for k in ("stmt_patch", "tail_patch"):
        E[f"recovery_{k}"] = ((true_rate(runs[k], m).mean() - true_rate(runs["none"], m).mean()) / full).item() if abs(full) > 1e-6 else None
    ab = true_rate(runs["hist_ablate"], m).mean() - true_rate(runs["none_ablate"], m).mean()
    E["ablation_reduction"] = (1 - ab / full).item() if abs(full) > 1e-6 else None
    res["cells"][c] = E
    print(f"[{slug}/{a.setup}] {c:13s} true-rate none {E['true_rate']['none'][0]:.2f} persona {E['true_rate']['hist'][0]:.2f} | "
          f"recovery stmt {E['recovery_stmt_patch'] or 0:.0%} tail {E['recovery_tail_patch'] or 0:.0%} | ablation removes {E['ablation_reduction'] or 0:.0%}", flush=True)
C = res["cells"]
res["rule1"] = bool(C[STAKE[0]]["persona_effect"] > 0 and C[STAKE[1]]["persona_effect"] < 0
                    and C[STAKE[0]]["true_rate"]["hist"][1] > C[STAKE[0]]["true_rate"]["none"][2]
                    and C[STAKE[1]]["true_rate"]["hist"][2] < C[STAKE[1]]["true_rate"]["none"][1])  # shortcut: non-overlapping CIs, stricter than a paired CI
res["rule2"] = all((C[c]["recovery_stmt_patch"] or 0) < 0.5 and (C[c]["recovery_tail_patch"] or 0) >= 0.5 for c in STAKE)
res["rule3"] = all((C[c]["ablation_reduction"] or 0) < 0.5 for c in STAKE)
print(f"[{slug}/{a.setup}] unparsed {res['unparsed']} | rules {res['rule1']} {res['rule2']} {res['rule3']} | layers stmt {l_stmt} tail {l_tail} | {time.time() - t0:.0f}s")
json.dump(res, open(f"{a.out}/e22_generation_{a.setup}_{slug}.json", "w"), indent=1)
