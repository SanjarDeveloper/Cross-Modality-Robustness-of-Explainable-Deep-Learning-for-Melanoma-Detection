"""Stage 4B3 descriptive statistics: paired correctness (E1.4), positive-class ECE with reliability
bins, and lesion-cluster bootstrap helpers. Pure Python/numpy.

ECE convention ece_pos_v1 (newly specified in Stage 4B3; retrospective implementation decision):
  p = melanoma probability, y = melanoma indicator; 15 equal-width bins on [0,1]; bin b covers
  [b/15, (b+1)/15) except the last, which is [14/15, 1]; ECE = sum_b (n_b/N) |mean(p_b) - mean(y_b)|.
  Empty bins are reported with n=0 and contribute 0. Positive-class (not top-label) calibration.
"""
import math

import numpy as np

N_BINS = 15


def _check(p, y):
    p, y = list(p), list(y)
    if len(p) != len(y) or not p:
        raise ValueError("p and y must be non-empty and equally long")
    for v in p:
        if not (isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError(f"invalid probability {v!r}")
    for v in y:
        if v not in (0, 1):
            raise ValueError(f"invalid label {v!r}")
    return p, y


def bin_index(v, n_bins=N_BINS):
    """Left-closed/right-open equal-width bins; the last bin includes 1.0. Exact for edges b/n_bins."""
    from fractions import Fraction
    b = int(Fraction(v) * n_bins)          # floor of exact rational value
    return min(b, n_bins - 1)


def ece_bins(p, y, n_bins=N_BINS):
    p, y = _check(p, y)
    N = len(p)
    rows = []
    for b in range(n_bins):
        idx = [i for i, v in enumerate(p) if bin_index(v, n_bins) == b]
        n = len(idx)
        lo, hi = b / n_bins, (b + 1) / n_bins
        if n == 0:
            rows.append({"bin": b, "lower": lo, "upper": hi, "upper_closed": b == n_bins - 1, "n": 0,
                         "mean_prob": None, "observed_mel_fraction": None, "abs_gap": None, "weight": 0.0,
                         "contribution": 0.0, "empty": True})
            continue
        mp = sum(p[i] for i in idx) / n
        my = sum(y[i] for i in idx) / n
        rows.append({"bin": b, "lower": lo, "upper": hi, "upper_closed": b == n_bins - 1, "n": n,
                     "mean_prob": mp, "observed_mel_fraction": my, "abs_gap": abs(mp - my), "weight": n / N,
                     "contribution": n / N * abs(mp - my), "empty": False})
    return {"convention": "ece_pos_v1", "n_bins": n_bins, "N": N, "ece": sum(r["contribution"] for r in rows),
            "n_empty_bins": sum(r["empty"] for r in rows), "bins": rows}


def correctness_table(pairs):
    """pairs: dicts with label, pred_d, pred_c. Correct = (pred == label)."""
    t = {"both_correct": 0, "dermoscopy_only_correct": 0, "clinical_only_correct": 0, "both_incorrect": 0}
    for p in pairs:
        cd, cc = int(p["pred_d"] == p["label"]), int(p["pred_c"] == p["label"])
        t[{(1, 1): "both_correct", (1, 0): "dermoscopy_only_correct", (0, 1): "clinical_only_correct",
           (0, 0): "both_incorrect"}[(cd, cc)]] += 1
    n = sum(t.values())
    t["n"] = n
    t["accuracy_dermoscopy"] = (t["both_correct"] + t["dermoscopy_only_correct"]) / n if n else None
    t["accuracy_clinical"] = (t["both_correct"] + t["clinical_only_correct"]) / n if n else None
    t["accuracy_diff"] = t["accuracy_dermoscopy"] - t["accuracy_clinical"] if n else None
    return t


def flatten_clusters(clusters):
    """List of clusters (each a list of rows), with repeats -> flat row list preserving multiplicity."""
    return [r for c in clusters for r in c]


def binary_metrics(rows, t):
    """rows: dicts with prob, label. Returns sens, spec (None if undefined)."""
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    sens = sum(r["prob"] >= t for r in pos) / len(pos) if pos else None
    spec = sum(r["prob"] < t for r in neg) / len(neg) if neg else None
    return sens, spec


def as_array_safe(x):
    return np.asarray(x, dtype=float)
