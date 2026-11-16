"""E12a gate + prompted readouts for one prior-paper setup (data/<setup>/setup.json).
Conditions: none (default assistant), hist (persona), ctrl (matched control persona).
Truth probe: LR on chat-wrapped GoT statements (no persona), tested on the setup's claims that have a truth label.
Endorsement probe (if the setup has an endorse column): LR on claims under the persona, 5-fold CV by statement.
Gate (fixed before the run): a probe is a readout only at layers where it reaches >= 85% held-out.
Also reports flip rates (E6): share of claims whose truth-probe sign or answer flips under the persona.
E18: --adapters HIST CTRL replaces the persona/control system prompts with LoRAs merged into the no-prompt model."""
import argparse, json, os, sys, time
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.acts import last_token_acts
from src.got import DS
from src.probes import fit_lr, correct
from src import setup as S
from src.lora import merge

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--setup", default="historical")
p.add_argument("--n_got", type=int, default=400)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default=os.path.expanduser("~/pb/data/got"))
p.add_argument("--adapters", nargs=2, default=None, help="E18: persona and control LoRA dirs")
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
torch.manual_seed(a.seed)
slug = a.model.split("/")[-1]
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(a.out, exist_ok=True)

cfg, cl = S.load(a.setup)
SYS, CELLS, ecol = cfg["conditions"], cfg["cells"], cfg["endorse_col"]
got = pd.concat([pd.read_csv(f"{a.data}/{d}.csv").sample(frac=1, random_state=a.seed).groupby("label").head(a.n_got // 2)
                 for d in DS])

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
wrap, ask, _ = S.formats(tok, SYS)
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
Hg = acts([wrap("none", s) for s in got.statement])
AD = dict(zip(("hist", "ctrl"), a.adapters)) if a.adapters else {}
def under(c, f):
    """f(text condition) under condition c: its system prompt, or (E18) the no-prompt text with c's LoRA merged."""
    if c not in AD:
        return f(c)
    restore = merge(model, AD[c])
    try:
        return f("none")
    finally:
        restore()
Hc = {c: under(c, lambda k: acts([wrap(k, s) for s in cl.statement])) for c in SYS}
nL = Hg.shape[1]
yg = torch.tensor(got.label.values, dtype=torch.float32, device="cuda")
has_t = torch.tensor((cl.truth != "").values, device="cuda")
yt = torch.tensor(pd.to_numeric(cl.truth, errors="coerce").fillna(-1).values, dtype=torch.float32, device="cuda")
cells = cl.cell.values

Wt, truth_acc = [], []
for l in range(nL):
    w, b = fit_lr(Hg[:, l], yg)
    Wt.append((w, b)); truth_acc.append(correct(Hc["none"][has_t, l], yt[has_t], w, b).mean().item())
gate_t = [l for l in range(nL) if truth_acc[l] >= 0.85]
lt = max(range(nL), key=lambda l: truth_acc[l])
res = {"exp": "e12a_gate", "setup": a.setup, "paper": cfg["paper"], "model": a.model, "seed": a.seed, "system_prompts": SYS,
       "adapters": AD or None, "n_claims": cl.cell.value_counts().to_dict(), "truth_acc": truth_acc, "gate_truth_layers": gate_t, "truth_readout": lt}
We = None
if ecol:
    has_e = torch.tensor((cl[ecol] != "").values, device="cuda")
    ye = torch.tensor(pd.to_numeric(cl[ecol], errors="coerce").fillna(-1).values, dtype=torch.float32, device="cuda")
    He, yee = Hc["hist"][has_e], ye[has_e]
    fold = (torch.arange(len(yee), device="cuda") % 5)[torch.randperm(len(yee), device="cuda")]
    We, end_acc = [], []
    for l in range(nL):
        c = torch.cat([correct(He[fold == k, l], yee[fold == k], *fit_lr(He[fold != k, l], yee[fold != k])) for k in range(5)])
        end_acc.append(c.mean().item()); We.append(fit_lr(He[:, l], yee))
    res.update({"endorse_acc": end_acc, "gate_endorse_layers": [l for l in range(nL) if end_acc[l] >= 0.85],
                "cos_truth_endorse": [F.cosine_similarity(Wt[l][0], We[l][0], dim=0).item() for l in range(nL)]})
print(f"[{slug}/{a.setup}] truth probe best l={lt} acc {truth_acc[lt]:.3f}, layers>=85%: {gate_t}", flush=True)
if ecol:
    print(f"[{slug}/{a.setup}] endorsement layers>=85%: {res['gate_endorse_layers']}; cos(truth, endorse) l={lt}: "
          f"{res['cos_truth_endorse'][lt]:+.3f}", flush=True)

T = {c: under(c, lambda k: tf_logit([ask(k, s) for s in cl.statement])) for c in SYS}
res["behavior_acc_vs_truth"] = {c: ((T[c][has_t.cpu()] > 0).float() == yt[has_t].cpu()).float().mean().item() for c in SYS}
res["cells"] = {}
for cell in CELLS:
    m = torch.tensor(cells == cell)
    R = {"behavior_logit": {c: boot(T[c][m]) for c in SYS}, "probe": {},
         "d_behavior_hist": boot((T["hist"] - T["none"])[m]), "d_behavior_ctrl": boot((T["ctrl"] - T["none"])[m]),
         "d_behavior_hist_ctrl": boot((T["hist"] - T["ctrl"])[m]),  # paired persona-vs-control (E17)
         "flip_behavior_hist": ((T["hist"] > 0) != (T["none"] > 0))[m].float().mean().item(),
         "flip_behavior_ctrl": ((T["ctrl"] > 0) != (T["none"] > 0))[m].float().mean().item()}
    for l in gate_t:
        s = {c: (Hc[c][:, l] @ Wt[l][0] + Wt[l][1]).cpu() for c in SYS}
        R["probe"][str(l)] = {c: boot(s[c][m]) for c in SYS}
        R["probe"][str(l)]["d_hist"] = boot((s["hist"] - s["none"])[m])
        R["probe"][str(l)]["d_ctrl"] = boot((s["ctrl"] - s["none"])[m])
        R["probe"][str(l)]["d_hist_ctrl"] = boot((s["hist"] - s["ctrl"])[m])
        R["probe"][str(l)]["flip_hist"] = ((s["hist"] > 0) != (s["none"] > 0))[m].float().mean().item()
        R["probe"][str(l)]["flip_ctrl"] = ((s["ctrl"] > 0) != (s["none"] > 0))[m].float().mean().item()
    res["cells"][cell] = R
    pl = R["probe"].get(str(lt), {})
    print(f"  {cell:13s} truth-probe shift l={lt}: persona {pl.get('d_hist', [0])[0]:+.2f} control {pl.get('d_ctrl', [0])[0]:+.2f} "
          f"(flip {pl.get('flip_hist', 0):.0%} vs {pl.get('flip_ctrl', 0):.0%}) | behavior shift persona "
          f"{R['d_behavior_hist'][0]:+.2f} control {R['d_behavior_ctrl'][0]:+.2f} (flip {R['flip_behavior_hist']:.0%})", flush=True)
print(f"behavior acc vs truth: {res['behavior_acc_vs_truth']}; {time.time() - t0:.0f}s")
tag = S.file_tag(a.setup)
pr = {"truth": torch.stack([torch.cat([w, b[None]]) for w, b in Wt]).cpu()}
if We:
    pr["endorse"] = torch.stack([torch.cat([w, b[None]]) for w, b in We]).cpu()
if not AD:  # the truth probe is trained without persona either way; LoRA runs reuse the prompt run's file
    torch.save(pr, f"{here}/probes_chat_{tag}{slug}.pt")
json.dump(res, open(f"{a.out}/e12a_{a.setup}_gate_{'lora_' if AD else ''}{slug}.json", "w"), indent=1)
