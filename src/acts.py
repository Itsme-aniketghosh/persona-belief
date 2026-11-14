import torch


@torch.no_grad()
def last_token_acts(model, tok, texts, bs=32, add_special_tokens=True):  # False for chat-templated text (BOS already in)
    """Residual stream at the last token: float32 [n, n_layers+1, d] on CPU.
    Index 0 = embeddings, index l = output of block l (pre final norm at every layer, including the last).
    Read with our own hooks, not output_hidden_states: HF 4.57 records block outputs with hooks that run
    before a steering hook, so hidden_states[l] misses steering applied at layer l (caught by E2)."""
    tok.padding_side = "left"  # last real token sits at position -1
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mods = [model.model.embed_tokens] + list(model.model.layers)
    out = []
    for i in range(0, len(texts), bs):
        got = [None] * len(mods)

        def grab(j):
            def hook(mod, inp, o):
                got[j] = (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu()
            return hook

        hs = [m.register_forward_hook(grab(j)) for j, m in enumerate(mods)]  # registered after any steer hook
        try:
            b = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=add_special_tokens).to(model.device)
            model.model(**b)  # base model only: no LM head needed
        finally:
            for h in hs:
                h.remove()
        out.append(torch.stack(got, 1))
    return torch.cat(out)
