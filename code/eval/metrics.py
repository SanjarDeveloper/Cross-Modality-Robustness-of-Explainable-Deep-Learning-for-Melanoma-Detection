"""Diagnostic metrics, calibration, and the paired statistical tests.

Operating-point policy (matters for every sensitivity number reported):
the decision threshold is chosen ONCE on the dermoscopic validation split
and then FROZEN. Re-tuning the threshold per condition would hide exactly
the degradation the study is trying to measure.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


# ------------------------------------------------------------------ basics
def sensitivity(y, p, thr):
    y, p = np.asarray(y), np.asarray(p)
    pos = y == 1
    return float((p[pos] >= thr).mean()) if pos.any() else float("nan")


def specificity(y, p, thr):
    y, p = np.asarray(y), np.asarray(p)
    neg = y == 0
    return float((p[neg] < thr).mean()) if neg.any() else float("nan")


def auroc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def threshold_at_sensitivity(y, p, target=0.95):
    """Pick the operating point on the BASELINE (dermoscopy) validation set.

    Clinical triage is sensitivity-first: melanomas missed are the costly
    error, so the threshold targets a fixed sensitivity rather than
    maximising accuracy on an 11%-prevalence set.
    """
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return float(thr[ok[0]]) if len(ok) else float(thr[np.argmax(tpr)])


def ece(y, p, bins=15):
    """Expected Calibration Error, equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= 0) & (p <= edges[1])
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


def all_metrics(y, p, thr):
    return {
        "n": int(len(y)),
        "n_mel": int(np.sum(y)),
        "sensitivity": sensitivity(y, p, thr),
        "specificity": specificity(y, p, thr),
        "auroc": auroc(y, p),
        "ece": ece(y, p),
    }


# ------------------------------------------------------------- paired tests
def mcnemar(y, p_a, p_b, thr):
    """Exact McNemar on paired classifications of the SAME lesions.

    Returns (b, c, p_value) where b = correct under A but wrong under B.
    Exact binomial is used because discordant counts here are small
    (Derm7pt has 252 melanomas), where the chi-square approximation is poor.
    """
    from scipy.stats import binomtest
    y = np.asarray(y)
    ca = (np.asarray(p_a) >= thr).astype(int) == y
    cb = (np.asarray(p_b) >= thr).astype(int) == y
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


def delong(y, p_a, p_b):
    """DeLong test for two correlated ROC curves (same cases, two readings).

    Implements the fast midrank algorithm (Sun & Xu 2014). Returns
    (auc_a, auc_b, p_value).
    """
    from scipy import stats
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p_a, float))  # any consistent order
    y = y[order]
    preds = np.vstack([np.asarray(p_a, float)[order], np.asarray(p_b, float)[order]])
    m = int(np.sum(y == 1))
    n = int(np.sum(y == 0))
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")
    pos = preds[:, y == 1]
    neg = preds[:, y == 0]
    k = preds.shape[0]

    def midrank(x):
        j = np.argsort(x)
        z = x[j]
        N = len(x)
        t = np.zeros(N, float)
        i = 0
        while i < N:
            j2 = i
            while j2 < N - 1 and z[j2 + 1] == z[i]:
                j2 += 1
            t[i:j2 + 1] = 0.5 * (i + j2) + 1
            i = j2 + 1
        out = np.empty(N, float)
        out[j] = t
        return out

    tx = np.array([midrank(pos[r]) for r in range(k)])
    ty = np.array([midrank(neg[r]) for r in range(k)])
    tz = np.array([midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = (tz[:, :m].sum(1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    s = np.atleast_2d(s)
    L = np.array([[1, -1]], float)
    var = float(L @ s @ L.T)
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


def bootstrap_ci(fn, *arrays, n_boot=2000, alpha=0.05, paired=True, seed=0):
    """Percentile bootstrap CI. Resamples CASES (not images) so paired
    dermoscopy/clinical readings move together -- required for Derm7pt."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    stats_ = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = fn(*[a[idx] for a in arrays])
        except Exception:
            continue
        if np.isfinite(v):
            stats_.append(v)
    if not stats_:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cohens_d_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 0 else float("nan")
