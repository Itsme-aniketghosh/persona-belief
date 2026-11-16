"""E14: Zhong & Li 2026 steer-then-knock-out design, with the truth probe in place of refusal.
Steer with a Chen persona vector v at layer l (all positions), then project v-hat out of the residual stream at layer k,
and read the E0 truth probe at the last readout layers. Retained = [s(steer + KO) - s(KO alone)] / [s(steer) - s0]:
the KO-alone term removes the shift that projecting out v-hat's natural component causes by itself (v1 omitted it).
k = l is the same-layer control (cancels the injection by construction, the flaw in the original design), k = l-2 is a
before-injection control (should change nothing), k = l+2 and l+4 are the informative downstream knockouts, and 20
random directions knocked out at l+2 are the null.
Reading (fixed before the run): retained < 50% at k = l+2 with random KO retaining > 90% means the effect is carried
along v itself downstream; retained >= 50% means it has been converted into other features by l+2."""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.acts import last_token_acts
from src.got import heldout_generic
from src.steer import steer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--layer", type=int, default=None, help="default: Qwen 20, Llama 16")
p.add_argument("--traits", nargs="+", default=["evil", "sycophantic", "hallucinating"])
p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0])
p.add_argument("--n_null", type=int, default=20)
p.add_argument("--n", type=int, default=240)
p.add_argument("--data", default=os.path.expanduser("~/pb/data/got"))
p.add_argument("--tag", default="", help="output file suffix (E21: larger alphas on Qwen)")
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
p.add_argument("--e0_results", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
torch.manual_seed(0)
slug = a.model.split("/")[-1]
l = a.layer or (16 if "Llama" in slug else 20)
e0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e00_baselines")
texts = heldout_generic(a.data, a.n, 5).statement.tolist()

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
P = torch.load(f"{e0}/probes_{slug}.pt")["lr"].cuda()
W, B = P[:, :-1], P[:, -1]
nL, d = W.shape
acc0 = json.load(open(f"{a.e0_results}/e00_probes_{slug}.json"))["results"]["lr"]["acc"]
reads = [k for k in range(nL - 3, nL) if acc0[k] >= 0.85]  # last readout layers

def readout(hooks):
    """Mean truth-probe score at the readout layers under the given (layer, fn) hooks."""
    hs = []
    for k, fn in hooks:
        hs.append(model.model.layers[k - 1].register_forward_hook(fn))
    try:
        H = last_token_acts(model, tok, texts).cuda()
    finally:
        for h in hs:
            h.remove()
    return ((H * W).sum(-1) + B)[:, reads]  # [n, R]

nll_texts = heldout_generic(a.data, 100, 7).statement.tolist()
@torch.no_grad()
def nll(hooks, bs=25):
    """Coherence: mean token NLL of held-out plain-text statements under the given hooks."""
    hs = [model.model.layers[k - 1].register_forward_hook(fn) for k, fn in hooks]
    tok.padding_side = "right"
    tot, n = 0.0, 0
    try:
        for i in range(0, len(nll_texts), bs):
            b = tok(nll_texts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
            lg = model(**b).logits[:, :-1].float()
            m = b.attention_mask[:, 1:].bool()
            tot += F.cross_entropy(lg.transpose(1, 2), b.input_ids[:, 1:], reduction="none")[m].sum().item(); n += m.sum().item()
    finally:
        tok.padding_side = "left"
        for h in hs:
            h.remove()
    return tot / n

def add(vec):
    def hk(mod, inp, o):
        return (o[0] + vec,) + tuple(o[1:]) if isinstance(o, tuple) else o + vec
    return hk

def ko(u):
    u = u.to(torch.bfloat16)
    def hk(mod, inp, o):
        h = o[0] if isinstance(o, tuple) else o
        h = h - (h @ u)[..., None] * u
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    return hk

t0 = time.time()
s0 = readout([])
res = {"exp": "e14_knockout", "model": a.model, "steer_layer": l, "readouts": reads, "alphas": a.alphas,
       "reading": __doc__.split("Reading")[1].strip(), "traits": {}}
for t in a.traits:
    v = torch.load(f"{e0}/pv_{t}_{slug}.pt")[l].cuda()
    u = F.normalize(v, dim=0)
    T = {"v_norm": v.norm().item()}
    for al in a.alphas:
        sv = (al * v).to(torch.bfloat16)
        base = (readout([(l, add(sv))]) - s0).mean(0)  # [R]
        out = {"d_steer": base.tolist(), "nll_steer": nll([(l, add(sv))]), "nll_base": nll([])}
        for name, k in (("ko_same", l), ("ko_before", l - 2), ("ko_plus2", l + 2), ("ko_plus4", l + 4)):
            hooks = [(l, add(sv)), (k, ko(u))] if k != l else [(l, add(sv)), (l, ko(u))]  # same layer: hooks run in order
            k0 = (readout([(k, ko(u))]) - s0).mean(0)  # KO alone, no steering
            dk = (readout(hooks) - s0).mean(0) - k0
            out[name] = {"d": dk.tolist(), "ko_alone": k0.tolist(), "retained": (dk / base).tolist()}
        rk = []
        for _ in range(a.n_null):
            r = F.normalize(torch.randn(d, device="cuda"), dim=0)
            rk.append(((readout([(l, add(sv)), (l + 2, ko(r))]) - readout([(l + 2, ko(r))])).mean(0) / base).tolist())
        out["ko_random_plus2_retained"] = rk
        T[str(al)] = out
        print(f"[{slug} {t} l={l} α={al}] Δs steer {base[-1]:+.2f} | retained at readout {reads[-1]}: same {out['ko_same']['retained'][-1]:.0%} "
              f"before {out['ko_before']['retained'][-1]:.0%} +2 {out['ko_plus2']['retained'][-1]:.0%} +4 {out['ko_plus4']['retained'][-1]:.0%} "
              f"| random +2 min {min(x[-1] for x in rk):.0%} | {time.time() - t0:.0f}s", flush=True)
    res["traits"][t] = T
json.dump(res, open(f"{a.out}/e14_knockout_{slug}{a.tag}.json", "w"), indent=1)
