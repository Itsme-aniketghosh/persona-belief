"""E0b: persona vectors (Chen et al. 2025 response-avg mean diff; no LLM-judge filter yet) + steering sanity check.
Trait score without a judge: project each response, re-read by the UNSTEERED model, onto v_hat at the steering layer."""
import argparse, json, os, re, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.steer import steer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--traits", nargs="+", default=["evil", "sycophantic", "hallucinating"])
p.add_argument("--layer", type=int, default=20)  # Chen et al. steer Qwen2.5-7B at layer 20
p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0])  # floats: they become JSON keys
p.add_argument("--n_null", type=int, default=20)
p.add_argument("--max_new", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default=os.path.expanduser("~/pb/data/pv"))
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
torch.manual_seed(a.seed)
slug = a.model.split("/")[-1]
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(a.out, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
dev = model.device

def chat(system, q):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def rows(vec, i, bs):  # per-row steering vectors get sliced with the batch
    return vec[i:i + bs] if vec is not None and vec.dim() == 3 else vec

@torch.no_grad()
def generate(prompts, vec=None, sample=False, bs=64):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        kw = dict(do_sample=True, temperature=1.0, top_p=1.0) if sample else dict(do_sample=False)
        if vec is None:
            g = model.generate(**b, max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id, **kw)
        else:
            with steer(model, a.layer, rows(vec, i, bs)):
                g = model.generate(**b, max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id, **kw)
        out += tok.batch_decode(g[:, b.input_ids.shape[1]:], skip_special_tokens=True)
    return out

@torch.no_grad()
def read(prompts, responses, vec=None, bs=16):
    """Teacher-forced pass over prompt+response. Returns (response-avg acts [n, L+1, d], response mean NLL [n])."""
    tok.padding_side = "right"
    acts, nll = [], []
    for i in range(0, len(prompts), bs):
        P, R = prompts[i:i + bs], responses[i:i + bs]
        b = tok([p + r for p, r in zip(P, R)], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        pl = torch.tensor([len(tok(p, add_special_tokens=False).input_ids) for p in P], device=dev)
        pos = torch.arange(b.input_ids.shape[1], device=dev)
        m = (pos[None] >= pl[:, None]) & b.attention_mask.bool()  # response tokens
        if vec is None:
            o = model(**b, output_hidden_states=True)
        else:
            with steer(model, a.layer, rows(vec, i, bs)):
                o = model(**b, output_hidden_states=True)
        mf = m[..., None].float()
        acts.append(torch.stack([(h * mf).sum(1) / mf.sum(1).clamp(min=1) for h in o.hidden_states], 1).float().cpu())
        lp = -F.cross_entropy(o.logits[:, :-1].float().transpose(1, 2), b.input_ids[:, 1:], reduction="none")
        mt = m[:, 1:].float()
        nll.append((-(lp * mt).sum(1) / mt.sum(1).clamp(min=1)).cpu())
    return torch.cat(acts), torch.cat(nll)

REFUSAL = re.compile(r"\s*(I'm sorry|I am sorry|I apologize|I can(no|')t|I won't|I will not|As an AI)", re.I)

def ci(x, B=1000):
    m = x[torch.randint(len(x), (B, len(x)))].mean(1)
    return [x.mean().item(), m.quantile(0.025).item(), m.quantile(0.975).item()]

t0 = time.time()
summary = {"exp": "e00_persona", "model": a.model, "layer": a.layer, "alphas": a.alphas, "n_null": a.n_null,
           "seed": a.seed, "max_new": a.max_new, "vector": "response-avg mean diff pos-neg; pos refusals dropped by prefix regex, no judge", "traits": {}}
for trait in a.traits:
    E = json.load(open(f"{a.data}/extract/{trait}.json"))
    pos = [chat(ins["pos"], q) for ins in E["instruction"] for q in E["questions"]]
    neg = [chat(ins["neg"], q) for ins in E["instruction"] for q in E["questions"]]
    rp, rn = generate(pos, sample=True), generate(neg, sample=True)
    # shortcut: refusal-prefix filter stands in for the paper's GPT judge (refusal only, no trait score); upgrade with a local judge if vectors look weak
    keep = [i for i, r in enumerate(rp) if not REFUSAL.match(r)]
    print(f"[{trait}] dropped {len(rp) - len(keep)}/{len(rp)} refusing pos responses", flush=True)
    pos, rp = [pos[i] for i in keep], [rp[i] for i in keep]
    Ap, _ = read(pos, rp)
    An, _ = read(neg, rn)
    v = Ap.mean(0) - An.mean(0)  # [L+1, d]
    torch.save(v, f"{here}/pv_{trait}_{slug}.pt")
    vl = v[a.layer].to(dev, torch.bfloat16)
    vhat = F.normalize(v[a.layer], dim=0)
    print(f"[{trait}] vector done, |v_{a.layer}|={v[a.layer].norm():.2f}, {time.time() - t0:.0f}s", flush=True)

    Q = json.load(open(f"{a.data}/eval/{trait}.json"))["questions"]
    base = [chat(None, q) for q in Q]
    gen = {0.0: generate(base)}
    for al in a.alphas:
        if al:
            gen[al] = generate(base, al * vl)
    ref = gen[0.0]  # held-out text for the coherence check: the unsteered answers
    T = {"v_norm_by_layer": v.norm(dim=1).tolist(), "n_pos_kept": len(keep), "n_neg": len(rn), "steer": {}, "null": {}}
    for al, R in gen.items():
        A, _ = read(base, R)
        _, nll = read(base, ref, None if al == 0 else al * vl)
        T["steer"][al] = {"proj": ci(A[:, a.layer] @ vhat), "ref_nll": ci(nll),
                          "samples": [{"q": q, "r": r} for q, r in zip(Q, R)]}
    # null: n_null random directions with the same norm as v at this layer, same questions, each alpha
    rnd = F.normalize(torch.randn(a.n_null, v.shape[1]), dim=1) * v[a.layer].norm()
    bq = [b for _ in range(a.n_null) for b in base]
    rq = [r for _ in range(a.n_null) for r in ref]
    rv = rnd.repeat_interleave(len(base), 0)[:, None].to(dev, torch.bfloat16)
    for al in a.alphas:
        if al:
            R = generate(bq, al * rv)
            A, _ = read(bq, R)
            _, nll = read(bq, rq, al * rv)
            per_vec = (A[:, a.layer] @ vhat).view(a.n_null, -1).mean(1)  # one score per random vector
            T["null"][al] = {"proj_per_vec": per_vec.tolist(), "proj_q95": per_vec.quantile(0.95).item(),
                             "ref_nll": ci(nll), "samples": [{"q": bq[i], "r": R[i]} for i in range(0, len(R), len(base))][:5]}
    summary["traits"][trait] = T
    s = T["steer"]
    print(f"[{trait}] proj by alpha: " + ", ".join(f"{al}:{s[al]['proj'][0]:.2f}" for al in s)
          + " | null q95: " + ", ".join(f"{al}:{T['null'][al]['proj_q95']:.2f}" for al in T["null"])
          + " | ref NLL: " + ", ".join(f"{al}:{s[al]['ref_nll'][0]:.2f}" for al in s), flush=True)
    json.dump({"pos": rp, "neg": rn}, open(f"{a.out}/e00_pv_responses_{trait}_{slug}.json", "w"))

summary["minutes"] = (time.time() - t0) / 60
json.dump(summary, open(f"{a.out}/e00_persona_{slug}.json", "w"), indent=1)
