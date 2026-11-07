import numpy as np
from scipy.stats import binomtest


def mcnemar_exact(y, pred_a, pred_b):
    """Exact two-sided McNemar. Returns (n01, n10, p): n01 = A right & B wrong."""
    y, a, b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    a_ok, b_ok = a == y, b == y
    n01 = int((a_ok & ~b_ok).sum())
    n10 = int((~a_ok & b_ok).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    return n01, n10, float(binomtest(min(n01, n10), n, 0.5).pvalue)


def holm(pvals):
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
