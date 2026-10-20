"""Paired (correlated) AUROC comparison — DeLong, DeLong & Clarke-Pearson (1988), Biometrics 44:837-845,
computed with the midrank formulation of Sun & Xu (2014), IEEE Signal Process. Lett. 21:1389-1393.
Own implementation (no code reused). AUROC = Mann-Whitney statistic; tied positive/negative pairs
count 1/2.

For modality r with positive scores X (m) and negative scores Y (n):
  V10_i = (1/n) sum_j psi(X_i, Y_j),  V01_j = (1/m) sum_i psi(X_i, Y_j),  psi = 1[x>y] + 0.5*1[x=y]
  midrank form: V10_i = (R_i - Rx_i)/n, V01_j = 1 - (R_{m+j} - Ry_j)/m, with R the midrank in the
  pooled sample and Rx / Ry the midranks within positives / negatives.
  S10 = sample covariance (ddof=1) of V10 across the 2 modalities; S01 likewise.
  Var(AUC_d - AUC_c) = c' (S10/m + S01/n) c, c = (1, -1).  z = diff / sqrt(Var); p = 2*Phi(-|z|).
Zero variance: z and p are reported as undefined (None) with a reason; never replaced.
"""
import math

import numpy as np


def midrank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    r = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        r[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = r
    return out


def placements(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    m, n = len(pos), len(neg)
    R = midrank(np.concatenate([pos, neg]))
    v10 = (R[:m] - midrank(pos)) / n
    v01 = 1.0 - (R[m:] - midrank(neg)) / m
    return v10, v01


def auc(pos, neg):
    v10, _ = placements(pos, neg)
    return float(v10.mean())


def delong_paired(scores_a, scores_b, labels):
    """scores_a/b: per-case scores for the same cases (aligned); labels 0/1."""
    y = np.asarray(labels)
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    if not (len(a) == len(b) == len(y)):
        raise ValueError("length mismatch")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite scores")
    m, n = int((y == 1).sum()), int((y == 0).sum())
    if m < 2 or n < 2:
        return {"status": "undefined", "reason": "fewer than 2 cases in a class", "m": m, "n": n}
    v10, v01 = [], []
    for s in (a, b):
        p, q = placements(s[y == 1], s[y == 0])
        v10.append(p)
        v01.append(q)
    aucs = [float(v.mean()) for v in v10]
    S10 = np.cov(np.vstack(v10), ddof=1)
    S01 = np.cov(np.vstack(v01), ddof=1)
    S = S10 / m + S01 / n
    c = np.array([1.0, -1.0])
    var = float(c @ S @ c)
    diff = aucs[0] - aucs[1]
    out = {"status": "ok", "m": m, "n": n, "auc_a": aucs[0], "auc_b": aucs[1], "diff": diff,
           "var_a": float(S[0, 0]), "var_b": float(S[1, 1]), "cov_ab": float(S[0, 1]),
           "corr_ab": float(S[0, 1] / math.sqrt(S[0, 0] * S[1, 1])) if S[0, 0] > 0 and S[1, 1] > 0 else None,
           "var_diff": var, "se_diff": math.sqrt(var) if var > 0 else 0.0}
    if var <= 0.0:
        out.update(z=None, p_value=None, inference="undefined: zero variance of the paired contrast")
        return out
    z = diff / math.sqrt(var)
    from scipy.stats import norm
    p = float(2 * norm.sf(abs(z)))
    out.update(z=z, p_value=p, p_value_sci=f"{p:.3e}", log10_p=float((math.log(2) + norm.logsf(abs(z))) / math.log(10)),
               inference="two-sided normal approximation (DeLong 1988)")
    return out
