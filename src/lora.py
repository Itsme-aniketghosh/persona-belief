import json, os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def merge(model, adapter):
    """Merge a LoRA adapter into model weights in place (W += scale * B @ A; no peft in the env).
    Returns restore(), which puts the original weights back exactly (from a CPU copy)."""
    path = adapter if os.path.isdir(adapter) else snapshot_download(adapter)  # local dir (E18) or hub id
    cfg = json.load(open(f"{path}/adapter_config.json"))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / r ** 0.5 if cfg.get("use_rslora") else alpha / r
    sd = load_file(f"{path}/adapter_model.safetensors")
    params = dict(model.named_parameters())
    saved = {}
    for k in sd:
        if "lora_A" not in k:
            continue
        name = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
        saved[name] = params[name].detach().cpu().clone()
        with torch.no_grad():
            params[name] += (scale * sd[k.replace("lora_A", "lora_B")].cuda().float() @ sd[k].cuda().float()).to(params[name].dtype)
    assert saved and len(saved) * 2 == len([k for k in sd if "lora_" in k]), f"merged {len(saved)} of {len(sd)} tensors"

    def restore():
        with torch.no_grad():
            for name, w in saved.items():
                params[name].copy_(w.to(params[name].device))

    return restore
