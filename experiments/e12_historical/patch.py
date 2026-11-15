"""E12c patching for one setup: where does the persona's effect enter the statement tokens?
For each layer l: run the no-persona text, but overwrite the residual stream (output of block l) at the statement
tokens with the source run's values. Source = a condition (hist = persona, ctrl = control) or a LoRA adapter applied to
the no-persona text (emergent misalignment). Format "probe": read truth/endorsement probes at the statement's last
token. Format "ask": the yes/no question; read logit(True) - logit(False) (behavior under patching).
Rule (fixed before the run): some layer at or before the readout recovers >= 50% of the source's effect on every
stake cell, while shared cells move less than half as much as the smallest stake cell."""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src import setup as S

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--setup", default="historical")
p.add_argument("--format", default="probe", choices=["probe", "ask"])
p.add_argument("--adapter", default=None, help="LoRA source instead of the persona/control conditions")
p.add_argument("--max_claims", type=int, default=None, help="tiny pass")
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
p.add_argument("--e0_results", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
slug = a.model.split("/")[-1]
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(a.out, exist_ok=True)

cfg, cl = S.load(a.setup)
if a.max_claims:
    cl = cl.groupby("cell").head(a.max_claims // 4)
CELLS, STAKE, SHARED = cfg["cells"], cfg["stake_cells"], cfg["shared_cells"]
cells = cl.cell.values

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
dev = model.device
wrap, ask, span = S.formats(tok, cfg["conditions"])
fmt = (lambda c, s: wrap(c, s)) if a.format == "probe" else (lambda c, s: ask(c, s))
piece = (lambda s: S.SUF + s) if a.format == "probe" else (lambda s: "Statement: " + s)
tag = S.file_tag(a.setup)
PR = torch.load(f"{here}/probes_chat_{tag}{slug}.pt")
WT, BT = PR["truth"][:, :-1].cuda(), PR["truth"][:, -1].cuda()
gate = json.load(open(f"{a.e0_results}/e12a_{a.setup}_gate_{slug}.json"))
lt = gate["truth_readout"] if "truth_readout" in gate else max(gate["gate_truth_layers"], key=lambda l: gate["truth_acc"][l])
nL = WT.shape[0]
idT, idF = (tok.encode(w, add_special_tokens=False)[0] for w in ("True", "False"))
mods = [model.model.embed_tokens] + list(model.model.layers)

def batch(cond, idx):
    texts = [fmt(cond, cl.statement.iloc[i]) for i in idx]
    sp = [span(t, piece(cl.statement.iloc[i])) for t, i in zip(texts, idx)]  # (after, to_end) per row
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev), sp

def forward(b, caps=None, put=None):
    """Readout per row: truth-probe score at layer lt (probe) or logit(True)-logit(False) (ask).
    caps: dict to fill with block outputs at the last Kmax tokens for every layer; put: (layer, src, mask) overwrite."""
    hs, got = [], {}
    Km = caps.pop("Km") if caps is not None else None
    for j, m in enumerate(mods):
        if j == 0:
            continue
        if caps is not None:
            def hk(mod, inp, o, j=j):
                caps[j] = (o[0] if isinstance(o, tuple) else o)[:, -Km:].clone()
            hs.append(m.register_forward_hook(hk))
        if put is not None and put[0] == j:
            def hk2(mod, inp, o, src=put[1], mask=put[2]):
                h = (o[0] if isinstance(o, tuple) else o).clone()
                k = src.shape[1]
                h[:, -k:] = torch.where(mask[..., None], src, h[:, -k:])
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            hs.append(m.register_forward_hook(hk2))
        if a.format == "probe" and j == lt:
            def hk3(mod, inp, o):
                got["h"] = (o[0] if isinstance(o, tuple) else o)[:, -1].float()
            hs.append(m.register_forward_hook(hk3))  # after the patch hook, so it reads patched values
    try:
        with torch.no_grad():
            if a.format == "probe":
                model.model(**b)
                return got["h"] @ WT[lt] + BT[lt]
            lg = model(**b).logits[:, -1].float()
            return lg[:, idT] - lg[:, idF]
    finally:
        for h in hs:
            h.remove()

t0 = time.time()
bs = 16
batches = [list(range(i, min(i + bs, len(cl)))) for i in range(0, len(cl), bs)]
sources = [a.adapter] if a.adapter else ["hist", "ctrl"]
res = {"exp": "e12c_patch", "setup": a.setup, "paper": cfg["paper"], "model": a.model, "format": a.format,
       "truth_readout": lt if a.format == "probe" else "logit(True)-logit(False)", "rule": __doc__.split("Rule")[1].strip(),
       "sources": sources, "full_effect": {}, "patch": {}}
none_out, src_out, src_caps, meta = [], {s: [] for s in sources}, {s: [] for s in sources}, []
for ix in batches:
    bn, spn = batch("none", ix)
    none_out.append(forward(bn))
    Km = max(e1 for _, e1 in spn)
    for s in sources:
        if a.adapter:
            from src.lora import merge
            restore = merge(model, s)
            bs_, sps = bn, spn
        else:
            bs_, sps = batch(s, ix)
        for r, ((e0, e1), (f0, f1)) in enumerate(zip(spn, sps)):  # self-check: same span and same tokens
            assert (e0, e1) == (f0, f1) and torch.equal(bn.input_ids[r, -e1:bn.input_ids.shape[1] - e0],
                                                        bs_.input_ids[r, -f1:bs_.input_ids.shape[1] - f0]), f"span mismatch row {ix[r]}"
        caps = {"Km": Km}
        src_out[s].append(forward(bs_, caps=caps))
        src_caps[s].append({j: caps[j] for j in caps})
        if a.adapter:
            restore()
    mask = torch.stack([(torch.arange(Km, 0, -1, device=dev) > e0) & (torch.arange(Km, 0, -1, device=dev) <= e1) for e0, e1 in spn])
    meta.append((bn, mask))
none_all = torch.cat(none_out)
cm = {c: torch.tensor(cells == c, device=dev) for c in CELLS}
for s in sources:
    full = torch.cat(src_out[s]) - none_all
    res["full_effect"][s] = {c: full[cm[c]].mean().item() for c in CELLS}
    res["patch"][s] = {}
    last = lt if a.format == "probe" else nL - 1
    for l in range(1, last + 1):
        d = torch.cat([forward(bn, put=(l, src_caps[s][k][l], mask)) for k, (bn, mask) in enumerate(meta)]) - none_all
        flip = ((d + none_all > 0) != (none_all > 0))
        res["patch"][s][str(l)] = {c: {"d": d[cm[c]].mean().item(),
                                       "recovery": d[cm[c]].mean().item() / res["full_effect"][s][c] if abs(res["full_effect"][s][c]) > 1e-6 else None,
                                       "flip": flip[cm[c]].float().mean().item()} for c in CELLS}
    ok = []
    for l in range(1, last + 1):
        e = res["patch"][s][str(l)]
        rec_ok = all((e[c]["recovery"] or 0) >= 0.5 for c in STAKE)
        sh_ok = all(abs(e[c]["d"]) < min(abs(e[x]["d"]) for x in STAKE) / 2 for c in SHARED)
        if rec_ok and sh_ok:
            ok.append(l)
    res["rule_pass_layers"] = res.get("rule_pass_layers", {}) | {s: ok}
    print(f"[{slug}/{a.setup}/{a.format}] source {s}: full effect " + ", ".join(f"{c}:{res['full_effect'][s][c]:+.2f}" for c in CELLS)
          + f" | rule passes at layers {ok} | {time.time() - t0:.0f}s", flush=True)
    for l in range(1, last + 1, max(1, last // 8)):
        e = res["patch"][s][str(l)]
        print(f"   patch@{l:2d}: " + ", ".join(f"{c}:{e[c]['d']:+.2f}(flip {e[c]['flip']:.0%})" for c in CELLS), flush=True)
name = f"e12c_{a.setup}_patch_{a.format}_" + (a.adapter.split("/")[-1] + "_" if a.adapter else "") + f"{slug}.json"
json.dump(res, open(f"{a.out}/{name}", "w"))
print(f"done in {(time.time() - t0) / 60:.1f} min")
