"""E9: read the e00 truth probe on emergent-misalignment (EM) LoRA fine-tunes of Qwen2.5-7B-Instruct vs the base
model and a benign control fine-tune. Then ask whether the activation shift (fine-tune minus base) lines up with
the persona vectors beyond random-direction nulls.
Success rule (fixed before the run): for at least 2 of 3 EM models, (a) the truth gap shrinks more than for the
control fine-tune at most probe-good layers, and (b) cos(shift, v_evil) beats the 97.5% random null and the control."""
import argparse, json, os, sys, time
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.acts import last_token_acts
from src.got import heldout_generic

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--em", nargs="+", default=[f"ModelOrganismsForEM/Qwen2.5-7B-Instruct_{x}"
                                           for x in ("bad-medical-advice", "extreme-sports", "risky-financial-advice")])
p.add_argument("--control", default="eekay/Qwen2.5-7B-Instruct-owl-numbers-ft")  # benign numbers fine-tune
p.add_argument("--traits", nargs="+", default=["evil", "sycophantic", "hallucinating"])
p.add_argument("--n", type=int, default=300)
p.add_argument("--n_claim", type=int, default=150)
p.add_argument("--n_null", type=int, default=1000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default=os.path.expanduser("~/pb/data/got"))
p.add_argument("--out", default=os.path.expanduser("~/pb/results"))
p.add_argument("--e0_results", default=os.path.expanduser("~/pb/results"))
a = p.parse_args()
torch.manual_seed(a.seed)
slug = a.model.split("/")[-1]
e0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e00_baselines")
os.makedirs(a.out, exist_ok=True)

gen = heldout_generic(a.data, a.n, a.seed + 3)
clm = pd.read_csv(f"{a.data}/common_claim_true_false.csv").sample(a.n_claim, random_state=a.seed + 3)
df = pd.concat([gen.assign(stratum="generic"), clm.assign(stratum="common_claim")], ignore_index=True)
texts, strata, y = df.statement.tolist(), df.stratum.values, torch.tensor(df.label.values, dtype=torch.float32)

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
tok = AutoTokenizer.from_pretrained(a.model)
P = torch.load(f"{e0}/probes_{slug}.pt")["lr"]
W, B = P[:, :-1], P[:, -1]
acc0 = json.load(open(f"{a.e0_results}/e00_probes_{slug}.json"))["results"]["lr"]["acc"]
good = [l for l, x in enumerate(acc0) if x >= 0.85]
V = {t: torch.load(f"{e0}/pv_{t}_{slug}.pt") for t in a.traits}

def acts(adapter=None):
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if adapter:  # merge LoRA by hand: W += scale * B @ A (no peft in the env)
        path = snapshot_download(adapter)
        cfg = json.load(open(f"{path}/adapter_config.json"))
        r, alpha = cfg["r"], cfg["lora_alpha"]
        scale = alpha / r ** 0.5 if cfg.get("use_rslora") else alpha / r
        sd = load_file(f"{path}/adapter_model.safetensors")
        params = dict(model.named_parameters())
        n = 0
        for k in sd:
            if "lora_A" not in k:
                continue
            name = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
            Bm = sd[k.replace("lora_A", "lora_B")].cuda().float()
            with torch.no_grad():
                params[name] += (scale * Bm @ sd[k].cuda().float()).to(params[name].dtype)
            n += 1
        assert n > 0 and n * 2 == len([k for k in sd if "lora_" in k]), f"merged {n} pairs of {len(sd)} tensors"
        print(f"{adapter}: merged {n} LoRA pairs, scale {scale:.3f}", flush=True)
    H = last_token_acts(model, tok, texts)
    del model; torch.cuda.empty_cache()
    return H

def summarize(H):
    s = (H * W).sum(-1) + B  # [n, L+1]
    out = {}
    for st in ("generic", "common_claim"):
        m = torch.tensor(strata == st)
        yt = y[m]
        out[st] = {"acc": [((s[m, l] > 0).float() == yt).float().mean().item() for l in range(len(W))],
                   "gap": [(s[m & (y == 1), l].mean() - s[m & (y == 0), l].mean()).item() for l in range(len(W))],
                   "s_mean": [s[m, l].mean().item() for l in range(len(W))]}
    return out

t0 = time.time()
H0 = acts()
res = {"exp": "e09_em_organisms", "model": a.model, "seed": a.seed, "probe": "lr from e00", "probe_good_layers": good,
       "n_statements": {st: int((strata == st).sum()) for st in ("generic", "common_claim")},
       "rule": __doc__.split("Success rule")[1].strip(), "base": summarize(H0), "finetunes": {}}
d = W.shape[1]
R = F.normalize(torch.randn(a.n_null, d), dim=1)
for name in a.em + [a.control]:
    H = acts(name)
    shift = (H - H0).mean(0)  # [L+1, d] mean activation shift
    null = (F.normalize(shift, dim=1) @ R.T)  # [L+1, n_null] cos with random directions
    S = summarize(H)
    S["shift_norm"] = shift.norm(dim=1).tolist()
    S["cos_null_95"] = [[null[l].quantile(0.025).item(), null[l].quantile(0.975).item()] for l in range(len(W))]
    S["cos_shift_v"] = {t: F.cosine_similarity(shift, V[t], dim=1).tolist() for t in a.traits}
    S["gap_change"] = {st: [S[st]["gap"][l] - res["base"][st]["gap"][l] for l in range(len(W))] for st in ("generic", "common_claim")}
    res["finetunes"][name] = S
    print(f"{name} ({time.time() - t0:.0f}s): generic gap change at l=20 {S['gap_change']['generic'][20]:+.2f}, "
          f"acc l=20 {S['generic']['acc'][20]:.3f} (base {res['base']['generic']['acc'][20]:.3f}); "
          f"cos(shift, v) l=20 " + ", ".join(f"{t}:{S['cos_shift_v'][t][20]:+.3f}" for t in a.traits)
          + f" null95 {S['cos_null_95'][20][1]:.3f}", flush=True)
json.dump(res, open(f"{a.out}/e09_em_organisms_{slug}.json", "w"), indent=1)
print(f"done in {(time.time() - t0) / 60:.1f} min")
