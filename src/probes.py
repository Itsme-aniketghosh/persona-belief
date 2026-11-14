import torch
import torch.nn.functional as F


def fit_lr(X, y, l2=1e-3, iters=300):
    """L2 logistic regression on standardized X. Returns (w, b) in raw-activation space: s = X @ w + b."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros((), device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Z @ w + b, y) + l2 * w.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    w_raw = (w / sd).detach()
    return w_raw, (b - mu @ w_raw).detach()


def fit_md(X, y):
    """Mean-difference probe: w = mean(true) - mean(false), threshold at the midpoint."""
    m1, m0 = X[y == 1].mean(0), X[y == 0].mean(0)
    w = m1 - m0
    return w, -w @ (m1 + m0) / 2


def correct(X, y, w, b):
    return ((X @ w + b > 0).float() == y).float()


if __name__ == "__main__":  # self-check on separable synthetic data
    torch.manual_seed(0)
    d, n = 512, 400
    u = F.normalize(torch.randn(d), dim=0)
    y = (torch.rand(n) > 0.5).float()
    X = torch.randn(n, d) + 3 * (2 * y[:, None] - 1) * u + 5  # offset checks the bias term
    for fit in (fit_lr, fit_md):
        acc = correct(X, y, *fit(X, y)).mean().item()
        assert acc > 0.95, (fit.__name__, acc)
    print("probes self-check OK")
