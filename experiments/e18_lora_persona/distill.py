"""E18 step 1: bake a prompted persona into a LoRA by context distillation (no peft in the env: LoRA by hand).
The model answers dolly-15k open questions under the persona system prompt (cond = hist: 1500 scholar; ctrl: 2020
scholar), then a LoRA is trained on (no system prompt, same question) -> that answer, loss on answer tokens only.
The 2020-scholar LoRA is the control for "fine-tuned on its own scholar-style text".
Output: PEFT-format adapter dir (adapter_config.json + adapter_model.safetensors) readable by src/lora.merge, plus
the generated training data for inspection. Self-check: merged weights reproduce the hooked LoRA's logits."""
import argparse, json, os, random, sys, time
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src import setup as S
from src.lora import merge

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--setup", default="historical", help="which setup's persona/control prompts (E21: physician1850)")
p.add_argument("--cond", default="hist", choices=["hist", "ctrl"])
p.add_argument("--n", type=int, default=1000)
p.add_argument("--max_new", type=int, default=160)
p.add_argument("--r", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--dolly", default="/scratch/ghosh.anik/h/hub/datasets--databricks--databricks-dolly-15k/snapshots/"
               "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl")
p.add_argument("--out", default=None, help="adapter dir (default /scratch/$USER/persona-belief/lora/<cond>_<model>)")
a = p.parse_args()
random.seed(0); torch.manual_seed(0)
slug = a.model.split("/")[-1]
tag = S.file_tag(a.setup)
out = a.out or f"/scratch/{os.environ['USER']}/persona-belief/lora/{tag}{a.cond}_{slug}"
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out, exist_ok=True)

cfg, cl = S.load(a.setup)
persona = cfg["conditions"][a.cond]
rows = [json.loads(l) for l in open(a.dolly, encoding="utf-8")]
rows = [r for r in rows if r["category"] in ("open_qa", "general_qa", "brainstorming") and not r["context"] and len(r["instruction"]) < 300]
claims = set(cl.statement.str.lower())
rows = [r for r in rows if r["instruction"].lower() not in claims]  # no eval claim used as a training prompt
qs = [r["instruction"] for r in random.sample(rows, a.n)]

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
dev = model.device
try:
    tok.apply_chat_template([{"role": "system", "content": "x"}, {"role": "user", "content": "y"}], tokenize=False)
    sysmsg = lambda sp, q: [{"role": "system", "content": sp}, {"role": "user", "content": q}]
except Exception:  # no system role (Gemma-2): persona opens the user turn, as in src/setup.formats
    sysmsg = lambda sp, q: [{"role": "user", "content": sp + "\n\n" + q}]
prompt = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# 1. generate in-persona answers
t0 = time.time()
tok.padding_side = "left"
ans = []
with torch.no_grad():
    for i in range(0, len(qs), 32):
        b = tok([prompt(sysmsg(persona, q)) for q in qs[i:i + 32]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        g = model.generate(**b, max_new_tokens=a.max_new, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id)
        ans += tok.batch_decode(g[:, b.input_ids.shape[1]:], skip_special_tokens=True)
data = [{"q": q, "a": x.strip()} for q, x in zip(qs, ans) if x.strip()]
with open(f"{here}/data_{tag}{a.cond}_{slug}.jsonl", "w", encoding="utf-8") as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"[{slug}/{a.cond}] generated {len(data)} answers in {time.time() - t0:.0f}s; e.g. Q: {data[0]['q']!r} A: {data[0]['a'][:200]!r}", flush=True)

# 2. tokenize (no system prompt), labels on the answer only
ex = []
for d in data:
    pr = prompt([{"role": "user", "content": d["q"]}])
    full = tok.apply_chat_template([{"role": "user", "content": d["q"]}, {"role": "assistant", "content": d["a"]}], tokenize=False)
    assert full.startswith(pr), "chat template: prompt is not a prefix of the full conversation"
    ip, ifull = tok(pr, add_special_tokens=False).input_ids, tok(full, add_special_tokens=False).input_ids[:512]
    ex.append((ifull, len(ip)))

# 3. LoRA by hand on every attention and MLP projection
targets = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in
           ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")]
scale = a.alpha / a.r
lora, hooks = {}, []
for n, m in targets:
    A = torch.nn.Parameter(torch.randn(a.r, m.in_features, device=dev) / m.in_features ** 0.5)
    B = torch.nn.Parameter(torch.zeros(m.out_features, a.r, device=dev))
    lora[n] = (A, B)
    hooks.append(m.register_forward_hook(lambda mod, inp, o, A=A, B=B: o + (scale * (inp[0].float() @ A.T) @ B.T).to(o.dtype)))
for q in model.parameters():
    q.requires_grad_(False)
params = [x for ab in lora.values() for x in ab]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()  # checkpointing needs a grad path through the frozen embeddings
steps = a.epochs * ((len(ex) + a.bs - 1) // a.bs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
model.train()
step, losses = 0, []
for ep in range(a.epochs):
    random.shuffle(ex)
    for i in range(0, len(ex), a.bs):
        bt = ex[i:i + a.bs]
        T = max(len(x) for x, _ in bt)
        ids = torch.full((len(bt), T), tok.pad_token_id, device=dev)
        lab = torch.full((len(bt), T), -100, device=dev)
        att = torch.zeros((len(bt), T), dtype=torch.long, device=dev)
        for r, (x, npr) in enumerate(bt):  # right padding for training
            ids[r, :len(x)] = torch.tensor(x); att[r, :len(x)] = 1
            lab[r, npr:len(x)] = torch.tensor(x[npr:])
        lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        losses.append(loss.item()); step += 1
        if step % 25 == 0 or step == 1:
            print(f"  step {step}/{steps} loss {sum(losses[-25:]) / len(losses[-25:]):.3f} | {time.time() - t0:.0f}s", flush=True)
model.eval()

# 4. save in PEFT layout, then check that merging reproduces the hooked model
sd = {}
for n, (A, B) in lora.items():
    sd[f"base_model.model.{n}.lora_A.weight"] = A.detach().cpu().contiguous()
    sd[f"base_model.model.{n}.lora_B.weight"] = B.detach().cpu().contiguous()
save_file(sd, f"{out}/adapter_model.safetensors")
json.dump({"r": a.r, "lora_alpha": a.alpha, "use_rslora": False, "persona": persona, "cond": a.cond, "n_train": len(ex),
           "epochs": a.epochs, "lr": a.lr, "final_loss": sum(losses[-25:]) / len(losses[-25:]), "base_model": a.model},
          open(f"{out}/adapter_config.json", "w"), indent=1)
chk = tok([prompt([{"role": "user", "content": q}]) for q in qs[:4]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
with torch.no_grad():
    l_hook = model(**chk).logits[:, -1].float()
    for h in hooks:
        h.remove()
    l_base = model(**chk).logits[:, -1].float()
    restore = merge(model, out)
    l_merge = model(**chk).logits[:, -1].float()
    restore()
err, moved = (l_merge - l_hook).abs().max().item(), (l_hook - l_base).abs().max().item()
# bf16 rounding of the merged weights is ~0.3 logits, so the check only means something once the LoRA moves logits more
assert moved < 2 or err < 0.25 * moved, f"merged adapter does not reproduce the hooked LoRA (err {err:.3f}, LoRA moved logits {moved:.3f})"
print(f"[{slug}/{a.cond}] saved {out}; merge check err {err:.3f} vs LoRA shift {moved:.3f}; {time.time() - t0:.0f}s", flush=True)
