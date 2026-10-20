"""Stage 5B helpers: model-specific thresholds, base/adapted prediction joins, paired cluster
bootstrap statistics (both models scored on the same draws), Secondary B estimability."""
import math

from . import auroc_paired as AP, desc_stats as DS, threshold as TH


class ModelThreshold:
    """A threshold bound to ONE model checkpoint; applying it to another model's rows raises."""

    def __init__(self, model_sha, threshold, threshold_sha, label):
        self.model_sha, self.t, self.threshold_sha, self.label = model_sha, float(threshold), threshold_sha, label

    def predict_row(self, row):
        if row["checkpoint_sha256"] != self.model_sha:
            raise ValueError(f"threshold {self.label} applied to rows of another model")
        return TH.predict(float(row["prob"]), self.t)


def join_models(base_rows, adapt_rows, base_sha, adapt_sha):
    """One-to-one join by record_uid. Returns list of dicts with label, cluster info, p_base, p_adapt."""
    def index(rows, sha, name):
        d = {}
        for r in rows:
            u = r["record_uid"]
            if u in d:
                raise ValueError(f"{name}: duplicate {u}")
            if r["checkpoint_sha256"] != sha:
                raise ValueError(f"{name}: wrong model for {u}")
            p = float(r["prob"])
            if not (math.isfinite(p) and 0.0 <= p <= 1.0):
                raise ValueError(f"{name}: invalid prob {u}")
            d[u] = r
        return d
    b, a = index(base_rows, base_sha, "base"), index(adapt_rows, adapt_sha, "adapted")
    if set(b) != set(a):
        raise ValueError(f"record sets differ: {len(set(b) ^ set(a))} unmatched")
    out = []
    for u in sorted(b):
        if b[u]["label_mel"] != a[u]["label_mel"]:
            raise ValueError(f"label mismatch {u}")
        out.append({"record_uid": u, "label": int(b[u]["label_mel"]), "p_base": float(b[u]["prob"]),
                    "p_adapt": float(a[u]["prob"]), "pred_base": int(b[u]["pred"]), "pred_adapt": int(a[u]["pred"])})
    return out


def auc_pair(rows):
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    if not pos or not neg:
        return None
    ab = AP.auc([r["p_base"] for r in pos], [r["p_base"] for r in neg])
    aa = AP.auc([r["p_adapt"] for r in pos], [r["p_adapt"] for r in neg])
    return ab, aa, aa - ab


def sens_pair(rows, tb, ta):
    pos = [r for r in rows if r["label"] == 1]
    if not pos:
        return None
    sb = sum(r["p_base"] >= tb for r in pos) / len(pos)
    sa = sum(r["p_adapt"] >= ta for r in pos) / len(pos)
    return sb, sa, sa - sb


def spec_pair(rows, tb, ta):
    neg = [r for r in rows if r["label"] == 0]
    if not neg:
        return None
    sb = sum(r["p_base"] < tb for r in neg) / len(neg)
    sa = sum(r["p_adapt"] < ta for r in neg) / len(neg)
    return sb, sa, sa - sb


def cluster_stat(fn, *args):
    return lambda clusters: fn(DS.flatten_clusters(clusters), *args)


def secondary_b(adapt_val_probs, adapt_val_labels):
    r = TH.estimate([float(p) for p in adapt_val_probs], [int(y) for y in adapt_val_labels])
    return {"status": r["status"], "reason": r.get("reason"), "n_pos": r.get("n_pos"),
            "min_required": TH.MIN_POS, "threshold": r.get("threshold")}
