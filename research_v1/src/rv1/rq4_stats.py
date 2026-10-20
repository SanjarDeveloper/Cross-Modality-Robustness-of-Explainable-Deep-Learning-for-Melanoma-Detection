"""RQ4 statistics helpers (Stage 7B): sample median, one-sample Hodges-Lehmann pseudomedian (median
of Walsh averages, i <= j), two-sided Wilcoxon signed-rank with frozen settings, Holm adjustment,
paired case bootstrap with shared draws.
"""
import math

import numpy as np

WILCOXON = {"zero_method": "wilcox", "correction": False, "method": "approx", "alternative": "two-sided"}


def hodges_lehmann(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    i, j = np.triu_indices(x.size)
    return float(np.median((x[i] + x[j]) / 2.0))


def sample_median(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(x)) if x.size else None


def wilcoxon_test(x):
    """Frozen settings: zeros dropped (zero_method='wilcox'); normal approximation with scipy's tie
    correction of the variance, no continuity correction; two-sided. NOT an exact test."""
    from scipy.stats import wilcoxon
    x = np.asarray(x, dtype=float)
    nz = x[x != 0]
    out = {"n": int(x.size), "n_nonzero": int(nz.size), "n_zero": int(x.size - nz.size),
           "n_tied_abs_values": int(nz.size - np.unique(np.abs(nz)).size), "settings": dict(WILCOXON)}
    if nz.size == 0:
        out.update(statistic=None, p_value=None, testable=False, reason="all differences are zero")
        return out
    r = wilcoxon(x, **WILCOXON)
    out.update(statistic=float(r.statistic), p_value=float(r.pvalue), testable=True, reason=None,
               method_label="normal approximation, tie-corrected variance, zeros dropped (not exact)")
    return out


def holm(pvals):
    """Holm step-down adjusted p-values. None (untestable) is carried as p=1 so the family size is
    never reduced; the returned list marks those entries."""
    m = len(pvals)
    p = [1.0 if v is None else float(v) for v in pvals]
    order = sorted(range(m), key=lambda k: p[k])
    adj = [0.0] * m
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[k]))
        adj[k] = running
    return adj


def bootstrap_stats(values_by_stat, B, seed, alpha=0.05):
    """values_by_stat: {name: 1-D array over the SAME K pairs}. One draw of K pair indices per replicate
    is applied to every statistic (pairs stay together; multiplicity kept). Returns per-stat CIs for
    the sample median and HL."""
    names = list(values_by_stat)
    K = len(next(iter(values_by_stat.values())))
    if any(len(v) != K for v in values_by_stat.values()):
        raise ValueError("all statistics must be over the same pairs")
    rng = np.random.Generator(np.random.PCG64(seed))
    arrs = {n: np.asarray(v, dtype=float) for n, v in values_by_stat.items()}
    reps = {n: {"median": [], "hl": []} for n in names}
    undefined = 0
    for _ in range(B):
        idx = rng.integers(0, K, size=K)
        if K == 0:
            undefined += 1
            continue
        for n in names:
            s = arrs[n][idx]
            reps[n]["median"].append(float(np.median(s)))
            reps[n]["hl"].append(hodges_lehmann(s))
    out = {}
    for n in names:
        out[n] = {}
        for k in ("median", "hl"):
            a = np.asarray(reps[n][k])
            out[n][k] = {"ci95": np.quantile(a, [alpha / 2, 1 - alpha / 2], method="linear").tolist() if a.size else None,
                         "replicates": a}
    return out, {"B": B, "seed": seed, "n_undefined": undefined, "unreliable": undefined / B > 0.05,
                 "rng": "numpy Generator(PCG64(seed)); integers(0,K,K) per replicate, shared across statistics",
                 "quantiles": "numpy.quantile(method='linear')"}


def gate(adj_p_dermoscopy, hl_dermoscopy, alpha=0.05):
    """Prespecified gate: positive Holm-supported dermoscopy evidence (adjusted p < alpha and HL > 0)."""
    return adj_p_dermoscopy is not None and adj_p_dermoscopy < alpha and hl_dermoscopy is not None and hl_dermoscopy > 0


def finite(v):
    return v is not None and v != "" and math.isfinite(float(v))
