"""E16b: which path carries the persona's effect on the answer? Attention blocking in the yes/no question.
In both the no-persona and persona runs, stop query positions in region Q from attending to the prefix (every token
before the statement except the first real token, kept as an attention sink), at all layers or at layers >= k.
Effect remaining = persona minus none under the same block (so the block's own damage cancels).
  Q = tail (tokens after the statement): what remains flows through the statement tokens (the truth-probe site);
  Q = statement: what remains reaches the answer without passing through the statement tokens;
  Q = both: sanity check, should leave ~0.
E16a (patch.py --format ask --region suffix|both) is the patching counterpart.
Rule (fixed before the run): "direct path" if blocking the statement keeps >= 50% of the effect on every stake cell;
"statement path" if blocking the tail keeps >= 50%. Interpretable only if no-persona accuracy on truth-labelled claims
under that block stays >= 85% and p(True)+p(False) >= 0.5."""
import argparse, json, os, sys, time
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src import setup as S

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--setups", nargs="+", default=["historical", "physician1850", "partisan", "opposite"])
p.add_argument("--max_claims", type=int, default=None, help="tiny pass")
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
slug = a.model.split("/")[-1]
os.makedirs(a.out, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
dev = model.device
idT, idF = (tok.encode(w, add_special_tokens=False)[0] for w in ("True", "False"))
nL = len(model.model.layers)
BLK = {}  # current block: {"m": [B, T, T] bool, "from": first block index (1-based)}

def pre(j):
    def hk(mod, args, kwargs):
        if not BLK or j < BLK["from"]:
            return None
        m = kwargs.get("attention_mask")
        assert m is not None and m.dim() == 4 and m.shape[-1] == BLK["m"].shape[-1], "need a materialized 4D mask (eager)"
        bm = BLK["m"][:, None, -m.shape[-2]:, :]
        kwargs["attention_mask"] = (m & ~bm) if m.dtype == torch.bool else m.masked_fill(bm, torch.finfo(m.dtype).min)
        return args, kwargs
    return hk
for j, lay in enumerate(model.model.layers, start=1):
    lay.register_forward_pre_hook(pre(j), with_kwargs=True)

@torch.no_grad()
def run(texts, pieces, span, region=None, k=1, bs=16):
    """(log p(True) - log p(False), p(True) + p(False)); region in {None, tail, statement, both}, blocked at blocks >= k."""
    d, ms = [], []
    for i in range(0, len(texts), bs):
        tx = texts[i:i + bs]
        b = tok(tx, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        T = b.input_ids.shape[1]
        if region:
            bm = torch.zeros(len(tx), T, T, dtype=torch.bool, device=dev)
            for r, (t, pc) in enumerate(zip(tx, pieces[i:i + bs])):
                after, to_end = span(t, pc)
                first = T - int(b.attention_mask[r].sum())
                s0, s1 = T - to_end, T - after  # statement = [s0, s1), tail = [s1, T)
                q0, q1 = {"tail": (s1, T), "statement": (s0, s1), "both": (s0, T)}[region]
                bm[r, q0:q1, first + 1:s0] = True
            BLK.update(m=bm, **{"from": k})
        try:
            lp = model(**b, use_cache=False).logits[:, -1].float().log_softmax(-1)
        finally:
            BLK.clear()
        d.append(lp[:, idT] - lp[:, idF]); ms.append(lp[:, idT].exp() + lp[:, idF].exp())
    return torch.cat(d), torch.cat(ms)

t0 = time.time()
ks = sorted({1} | {max(1, round(nL * f)) for f in (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)})
res = {"exp": "e16_block", "model": a.model, "rule": __doc__.split("Rule")[1].strip(), "tail_from_k": ks, "setups": {}}
for name in a.setups:
    cfg, cl = S.load(name)
    if a.max_claims:
        cl = cl.groupby("cell").head(a.max_claims // 4)
    cl = cl.reset_index(drop=True)
    _, ask, span = S.formats(tok, cfg["conditions"])
    CELLS, STAKE = cfg["cells"], cfg["stake_cells"]
    cm = {c: torch.tensor(cl.cell.values == c, device=dev) for c in CELLS}
    has_t = torch.tensor((cl.truth != "").values, device=dev)
    yt = torch.tensor(pd.to_numeric(cl.truth, errors="coerce").fillna(-1).values, dtype=torch.float32, device=dev)
    pcs = ["Statement: " + s for s in cl.statement]
    tn, th = [ask("none", s) for s in cl.statement], [ask("hist", s) for s in cl.statement]
    def cond(region, k=1):
        dn, mn = run(tn, pcs, span, region, k)
        dh, _ = run(th, pcs, span, region, k)
        return {"effect": {c: (dh - dn)[cm[c]].mean().item() for c in CELLS},
                "acc_none": ((dn[has_t] > 0).float() == yt[has_t]).float().mean().item(), "tf_mass_none": mn.mean().item()}
    base = cond(None)
    if name == a.setups[0]:  # self-check: an empty block changes nothing; a full block changes something
        dn0, _ = run(tn[:8], pcs[:8], span)
        BLK_test = run(tn[:8], pcs[:8], span, "both")[0]
        assert not torch.allclose(dn0, BLK_test), "blocking had no effect: mask edit not reaching attention"
    R = {"none": base}
    for region in ("tail", "statement", "both"):
        R[region] = cond(region)
    R["tail_from_k"] = {str(k): cond("tail", k) for k in ks if k > 1}
    kept = lambda reg: {c: R[reg]["effect"][c] / base["effect"][c] if abs(base["effect"][c]) > 1e-6 else None for c in CELLS}
    ok = lambda reg: R[reg]["acc_none"] >= 0.85 and R[reg]["tf_mass_none"] >= 0.5
    R["kept"] = {reg: kept(reg) for reg in ("tail", "statement", "both")}
    R["direct_path"] = all((R["kept"]["statement"][c] or 0) >= 0.5 for c in STAKE) if ok("statement") else None
    R["statement_path"] = all((R["kept"]["tail"][c] or 0) >= 0.5 for c in STAKE) if ok("tail") else None
    res["setups"][name] = R
    print(f"[{slug}/{name}] effect " + ", ".join(f"{c}:{base['effect'][c]:+.2f}" for c in STAKE)
          + " | kept: " + "; ".join(f"{reg} " + ",".join(f"{R['kept'][reg][c]:.0%}" for c in STAKE)
                                    + f" (acc {R[reg]['acc_none']:.2f}, mass {R[reg]['tf_mass_none']:.2f})" for reg in ("tail", "statement", "both"))
          + f" | direct {R['direct_path']} statement {R['statement_path']} | tail from k: "
          + ", ".join(f"{k}:" + ",".join(f"{v['effect'][c] / base['effect'][c]:.0%}" for c in STAKE) for k, v in R["tail_from_k"].items())
          + f" | {time.time() - t0:.0f}s", flush=True)
json.dump(res, open(f"{a.out}/e16_block_{slug}.json", "w"), indent=1)
print(f"done in {(time.time() - t0) / 60:.1f} min")
