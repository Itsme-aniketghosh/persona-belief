"""E12b: steer the no-persona model along the historical-persona direction and split the truth/endorsement readout
shift into direct and mediated parts, per claim cell.
Persona vector v = mean(h_hist - h_none) over claims at layer l, cross-fitted: built on one half of the claims and
applied to the other half. The direct term w·v is one constant for every claim, so any selectivity across cells
must come from the mediated term (G·v - w·v, G = sum over positions of ds/dh_l from one backward pass).
Success rule (fixed before the run): for at least one model, the mediated truth-probe term on era-believed falsehoods
(toward true) and on modern truths (toward false) (1) beats the family-wise isotropic null and the 2020-control
direction, (2) is larger in magnitude than on shared facts, and (3) α=1 steering moves the truth readout the same way
the prompt did, with held-out NLL <= 1.5x unsteered."""
import argparse, csv, json, os, sys, time
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.acts import last_token_acts
from src.steer import steer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--steer_layers", type=int, nargs="+", default=None, help="default: Qwen 12 16, Llama 10 14")
p.add_argument("--alphas", type=float, nargs="+", default=[-1.0, -0.5, 0.5, 1.0, 2.0])
p.add_argument("--n_null", type=int, default=20)
p.add_argument("--n_null_lin", type=int, default=1000)
p.add_argument("--max_claims", type=int, default=None, help="tiny pass")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--claims", default=os.path.expanduser("~/pb/data/historical/claims.csv"))
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
p.add_argument("--e0_results", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
torch.manual_seed(a.seed)
slug = a.model.split("/")[-1]
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(a.out, exist_ok=True)
steer_layers = a.steer_layers or ([10, 14] if "Llama" in slug else [12, 16])

SYS = {"none": None,  # same prompts as gate.py (E12a)
       "hist": "You are a learned European scholar living in the year 1500. You hold the beliefs that educated people "
               "of your time hold, and you know nothing of discoveries made after 1500.",
       "ctrl": "You are a learned European scholar living in the year 2020. You hold the beliefs that educated people "
               "of your time hold, and you know of all discoveries made up to 2020."}
cl = pd.DataFrame(list(csv.DictReader(open(a.claims, encoding="utf-8"))))
if a.max_claims:
    cl = cl.groupby("cell").head(a.max_claims // 4)
cl[["truth", "era_endorsed"]] = cl[["truth", "era_endorsed"]].astype(int)
cells = cl.cell.values
CELLS = ["era_belief", "modern_truth", "shared_truth", "shared_false"]
fold = torch.randperm(len(cl)) % 2  # cross-fitting halves

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
model.requires_grad_(False)
dev = model.device
MARK = "\x00"

def wrap(cond, s):
    msgs = ([{"role": "system", "content": SYS[cond]}] if SYS[cond] else []) + [{"role": "user", "content": MARK}]
    return tok.apply_chat_template(msgs, tokenize=False).split(MARK)[0] + "Consider this statement: " + s

def ask(s):
    return tok.apply_chat_template([{"role": "user", "content": "Is the following statement true or false? Answer with "
                                     "exactly one word, True or False.\nStatement: " + s}], tokenize=False, add_generation_prompt=True)

idT, idF = (tok.encode(w, add_special_tokens=False)[0] for w in ("True", "False"))
PR = torch.load(f"{here}/probes_chat_{slug}.pt")  # from gate.py: {"truth","endorse"}: [L+1, d+1]
WT, BT = PR["truth"][:, :-1].cuda(), PR["truth"][:, -1].cuda()
WE, BE = PR["endorse"][:, :-1].cuda(), PR["endorse"][:, -1].cuda()
gate = json.load(open(f"{a.e0_results}/e12a_historical_gate_{slug}.json"))
nL, d = WT.shape
texts = [wrap("none", s) for s in cl.statement]
asks = [ask(s) for s in cl.statement]

E0 = json.load(open(f"{a.e0_results}/e00_persona_{slug}.json"))["traits"]  # held-out text: E0 unsteered answers
ref = [(tok.apply_chat_template([{"role": "user", "content": s["q"]}], tokenize=False, add_generation_prompt=True), s["r"])
       for t in E0 for s in E0[t]["steer"]["0.0"]["samples"]]

def rows(vec, idx):  # per-row steering vectors (cross-fitting gives each claim the other half's vector)
    return vec[idx] if vec.dim() == 3 else vec

@torch.no_grad()
def ref_nll(vec, l, bs=16):
    tok.padding_side = "right"
    tot = []
    for i in range(0, len(ref), bs):
        Pr, R = zip(*ref[i:i + bs])
        b = tok([p + r for p, r in zip(Pr, R)], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        pl = torch.tensor([len(tok(p, add_special_tokens=False).input_ids) for p in Pr], device=dev)
        m = ((torch.arange(b.input_ids.shape[1], device=dev)[None] >= pl[:, None]) & b.attention_mask.bool())[:, 1:].float()
        if vec is None:
            lg = model(**b).logits
        else:
            with steer(model, l, vec):
                lg = model(**b).logits
        nll = F.cross_entropy(lg[:, :-1].float().transpose(1, 2), b.input_ids[:, 1:], reduction="none")
        tot.append((nll * m).sum(1) / m.sum(1))
    return torch.cat(tot).mean().item()

@torch.no_grad()
def readouts(vec, l, bs=32):
    """Truth- and endorsement-probe scores [n, L+1] and behavior logit(True)-logit(False) [n] under steering.
    vec: None, [d], or [n, 1, d] (one vector per claim)."""
    S_t, S_e, beh = [], [], []
    for i in range(0, len(texts), bs):
        idx = torch.arange(i, min(i + bs, len(texts)))
        v = None if vec is None else rows(vec, idx)
        if v is None:
            H = last_token_acts(model, tok, texts[i:i + bs], add_special_tokens=False).cuda()
        else:
            with steer(model, l, v):
                H = last_token_acts(model, tok, texts[i:i + bs], add_special_tokens=False).cuda()
        S_t.append((H * WT).sum(-1) + BT); S_e.append((H * WE).sum(-1) + BE)
        tok.padding_side = "left"
        b = tok(asks[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        if v is None:
            lg = model(**b).logits[:, -1].float()
        else:
            with steer(model, l, v):
                lg = model(**b).logits[:, -1].float()
        beh.append(lg[:, idT] - lg[:, idF])
    return torch.cat(S_t), torch.cat(S_e), torch.cat(beh)

def readout_grads(l, reads, Wp, Bp, bs=16):
    """G[i, k] = sum over positions of d s_{reads[k]} / d h_l, for probe (Wp, Bp): float32 [n, K, d]."""
    tok.padding_side = "left"
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        delta = torch.zeros(b.input_ids.shape[0], 1, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
        got = {}

        def grab(k):
            def hook(mod, inp, o):
                got[k] = (o[0] if isinstance(o, tuple) else o)[:, -1].float()
            return hook

        hs = [model.model.layers[k - 1].register_forward_hook(grab(k)) for k in reads]
        try:
            with steer(model, l, delta):
                model.model(**b)
        finally:
            for h in hs:
                h.remove()
        out.append(torch.stack([torch.autograd.grad((got[k] @ Wp[k] + Bp[k]).sum(), delta, retain_graph=k != reads[-1])[0][:, 0].float()
                                for k in reads], 1))
    return torch.cat(out)

def boot(x, B=1000):
    m = x[torch.randint(len(x), (B, len(x)), device=x.device)].mean(1)
    return [x.mean().item(), m.quantile(0.025).item(), m.quantile(0.975).item()]

t0 = time.time()
A = {c: last_token_acts(model, tok, [wrap(c, s) for s in cl.statement], add_special_tokens=False) for c in SYS}  # CPU [n, L+1, d]
nll0 = ref_nll(None, None)
s0_t, s0_e, beh0 = readouts(None, None)
res = {"exp": "e12b_historical_steer", "model": a.model, "seed": a.seed, "steer_layers": steer_layers, "alphas": a.alphas,
       "rule": __doc__.split("Success rule")[1].strip(), "nll_unsteered": nll0, "n_claims": cl.cell.value_counts().to_dict(),
       "layers": {}}
R1 = F.normalize(torch.randn(a.n_null_lin, d, device=dev), dim=1)
cellmask = {c: torch.tensor(cells == c, device=dev) for c in CELLS}
for l in steer_layers:
    reads = {"truth": [k for k in gate["gate_truth_layers"] if k > l], "endorse": [k for k in gate["gate_endorse_layers"] if k > l]}
    # cross-fitted vectors: claim i gets the vector built on the other half
    dv = {c: (A[c][:, l] - A["none"][:, l]).cuda() for c in ("hist", "ctrl")}
    half = {c: torch.stack([dv[c][fold.cuda() == f].mean(0) for f in (0, 1)]) for c in dv}  # [2, d]
    v_of = {c: half[c][1 - fold.cuda()] for c in dv}  # [n, d]
    nv = half["hist"].norm(dim=1).mean().item()
    v_of["ctrl"] = F.normalize(v_of["ctrl"], dim=1) * v_of["hist"].norm(dim=1, keepdim=True)  # norm-matched control
    L = {"v_norm": nv, "cos_hist_ctrl": F.cosine_similarity(half["hist"].mean(0), half["ctrl"].mean(0), dim=0).item(),
         "readouts": reads, "linear": {}, "finite": {}}
    Hl = A["none"][:, l].cuda()
    hm = torch.rand(a.n_null_lin, len(cl), device=dev)
    hm = (hm < hm.median(1, keepdim=True).values).float()
    Sp = F.normalize(hm @ Hl / hm.sum(1, keepdim=True) - (1 - hm) @ Hl / (1 - hm).sum(1, keepdim=True), dim=1) * nv
    fw = lambda M: [M.min(0).values.quantile(0.025).item(), M.max(0).values.quantile(0.975).item()]
    for probe, (Wp, Bp) in {"truth": (WT, BT), "endorse": (WE, BE)}.items():
        ks = reads[probe]
        if not ks:
            continue
        G = readout_grads(l, ks, Wp, Bp)  # [n, K, d]
        Wr = torch.stack([Wp[k] for k in ks])
        L["linear"][probe] = {}
        for c in CELLS:
            m = cellmask[c]
            out = {}
            for vn in ("hist", "ctrl"):
                tot = (G[m] * v_of[vn][m][:, None]).sum(-1)  # [n_c, K]
                direct = (v_of[vn][m] @ Wr.T)  # [n_c, K], same value up to the fold's vector
                out[vn] = [{"readout": k, "total": boot(tot[:, j]), "direct": direct[:, j].mean().item(),
                            "mediated": boot(tot[:, j] - direct[:, j])} for j, k in enumerate(ks)]
            Gbar = G[m].mean(0)
            iso = Gbar @ (R1 * nv).T - Wr @ (R1 * nv).T
            spl = Gbar @ Sp.T - Wr @ Sp.T
            out["null_iso_95"] = [[iso[j].quantile(0.025).item(), iso[j].quantile(0.975).item()] for j in range(len(ks))]
            out["fw_iso"], out["fw_split"] = fw(iso), fw(spl)
            L["linear"][probe][c] = out
        if probe == "truth":
            for c in CELLS:
                o = L["linear"]["truth"][c]
                j = max(range(len(ks)), key=lambda i: abs(o["hist"][i]["mediated"][0]))
                print(f"[{slug} l={l}] truth {c:13s} readout {ks[j]}: mediated hist {o['hist'][j]['mediated'][0]:+.3f} "
                      f"(direct {o['hist'][j]['direct']:+.3f}) ctrl {o['ctrl'][j]['mediated'][0]:+.3f} | fw iso {o['fw_iso'][0]:+.2f},{o['fw_iso'][1]:+.2f}", flush=True)
    # finite α: v_hist and v_ctrl (cross-fitted, per claim) and n_null isotropic random vectors
    rnd = F.normalize(torch.randn(a.n_null, d, device=dev), dim=1) * nv
    vecs = [("hist", v_of["hist"][:, None].to(torch.bfloat16)), ("ctrl", v_of["ctrl"][:, None].to(torch.bfloat16))] + \
           [(f"r{i}", rnd[i].to(torch.bfloat16)) for i in range(a.n_null)]
    for name, vec in vecs:
        for al in a.alphas:
            st, se, bh = readouts(al * vec, l)
            ent = {}
            for c in CELLS:
                m = cellmask[c]
                ent[c] = {"d_truth": (st - s0_t)[m].mean(0).tolist(), "d_endorse": (se - s0_e)[m].mean(0).tolist(),
                          "d_behavior": (bh - beh0)[m].mean().item()}
            if name in ("hist", "ctrl"):
                ent["nll"] = ref_nll((al * vec).mean(0), l)  # shortcut: NLL uses the fold-average vector, not per-claim ones
            L["finite"].setdefault(name, {})[str(al)] = ent
    fv = L["finite"]["hist"]
    if "0.5" in fv and "-0.5" in fv and reads["truth"]:  # self-check: gradient vs central difference, era_belief cell
        k = reads["truth"][-1]
        cd = fv["0.5"]["era_belief"]["d_truth"][k] - fv["-0.5"]["era_belief"]["d_truth"][k]
        gr = L["linear"]["truth"]["era_belief"]["hist"][-1]["total"][0]
        L["grad_check"] = {"readout": k, "grad": gr, "central_diff": cd}
        print(f"  grad check readout {k}: grad {gr:+.3f} vs central diff {cd:+.3f}", flush=True)
    print(f"  α=1 truth shift at last readout by cell: " + ", ".join(f"{c}:{fv['1.0'][c]['d_truth'][reads['truth'][-1]]:+.2f}"
          for c in CELLS) + f" | NLL {fv['1.0']['nll']:.3f} vs {nll0:.3f} | {time.time() - t0:.0f}s", flush=True)
    res["layers"][str(l)] = L
json.dump(res, open(f"{a.out}/e12b_historical_steer_{slug}.json", "w"))
print(f"done in {(time.time() - t0) / 60:.1f} min")
