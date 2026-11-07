import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score


def prf(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def fpr(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    neg = (y == 0).sum()
    return float(((y == 0) & (pred == 1)).sum() / neg) if neg else 0.0


def brier(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def auroc(y, p):
    return float(roc_auc_score(y, p))


def ece(y, p, bins=15):
    """Expected calibration error, equal-width bins on P(y=1)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def bootstrap_ci(y, pred, metric, groups, n=1000, seed=0, alpha=0.05):
    """Percentile CI that resamples whole groups (sites/templates), not rows."""
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    stats = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        stats.append(metric(y[idx], pred[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def apply_temperature(logit, T):
    return 1 / (1 + np.exp(-np.asarray(logit, float) / T))


def fit_temperature(logit, y):
    """T > 0 minimising binary NLL of sigmoid(logit / T). Fit on held-out data only."""
    logit, y = np.asarray(logit, float), np.asarray(y, float)

    def nll(T):
        p = np.clip(apply_temperature(logit, T), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
