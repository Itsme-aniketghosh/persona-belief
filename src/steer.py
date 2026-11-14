from contextlib import contextmanager


@contextmanager
def steer(model, layer, vec):
    """h_layer <- h_layer + vec at every position, where h_layer = hidden_states[layer] (output of block layer-1).
    vec: [d], or [B, 1, d] for a different vector per batch row."""
    def hook(mod, inp, out):
        if isinstance(out, tuple):  # older transformers return (hidden, ...)
            return (out[0] + vec,) + out[1:]
        return out + vec

    h = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()
