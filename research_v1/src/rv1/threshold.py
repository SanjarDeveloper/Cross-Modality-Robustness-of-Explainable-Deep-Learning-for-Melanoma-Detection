"""v1 operating threshold (02_rev2 §1.1) and threshold-based descriptive counts.

Rule v1_highest_t_img_sens95: positive = melanoma (label 1); predict positive iff p >= t;
k = ceil(0.95 * n_pos) computed with exact integer arithmetic (ceil(95*n/100)); t = k-th largest
melanoma probability (the highest t with image sensitivity >= 0.95). NOT ESTIMABLE (no fallback)
if n_pos < 20, any score/label invalid, or t not finite. Values are never rounded.
"""
import math

RULE = "v1_highest_t_img_sens95"
TARGET_NUM, TARGET_DEN = 95, 100
MIN_POS = 20


def required_positives(n_pos):
    return -((-TARGET_NUM * n_pos) // TARGET_DEN)          # exact integer ceil


def estimate(probs, labels):
    if len(probs) != len(labels):
        return {"status": "NOT_ESTIMABLE", "reason": "length mismatch", "threshold": None}
    if any(y not in (0, 1) for y in labels):
        return {"status": "NOT_ESTIMABLE", "reason": "labels must be 0/1", "threshold": None}
    if not all(isinstance(p, float) and math.isfinite(p) for p in probs):
        return {"status": "NOT_ESTIMABLE", "reason": "non-finite or non-float score", "threshold": None}
    pos = sorted((p for p, y in zip(probs, labels) if y == 1), reverse=True)
    n_pos, n_neg = len(pos), len(labels) - len(pos)
    if n_pos < MIN_POS:
        return {"status": "NOT_ESTIMABLE", "reason": f"n_pos {n_pos} < {MIN_POS}", "threshold": None,
                "n_pos": n_pos}
    k = required_positives(n_pos)
    t = pos[k - 1]
    c = counts(probs, labels, t)
    return {"status": "ESTIMATED", "rule": RULE, "threshold": t, "threshold_repr": repr(t), "n_pos": n_pos,
            "n_neg": n_neg, "k": k, "ties_at_threshold_pos": sum(p == t for p in pos),
            "ties_at_threshold_all": sum(p == t for p in probs),
            "next_higher_pos_score": pos[k - 2] if k >= 2 else None,
            "next_lower_pos_score": pos[k] if k < n_pos else None, **c}


def predict(p, t):
    return int(p >= t)


def counts(probs, labels, t):
    tp = sum(1 for p, y in zip(probs, labels) if y == 1 and p >= t)
    fn = sum(1 for p, y in zip(probs, labels) if y == 1 and p < t)
    tn = sum(1 for p, y in zip(probs, labels) if y == 0 and p < t)
    fp = sum(1 for p, y in zip(probs, labels) if y == 0 and p >= t)
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "sensitivity": tp / (tp + fn) if tp + fn else None, "sens_num_den": [tp, tp + fn],
            "specificity": tn / (tn + fp) if tn + fp else None, "spec_num_den": [tn, tn + fp]}
