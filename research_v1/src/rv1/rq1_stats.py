"""RQ1 paired statistics (02_rev2 §3, §2 bootstrap rules). Pure Python/numpy.

- build_pairs: reconstruct Derm7pt pairs by pair_id from prediction rows (never by row order);
  rejects duplicates, incomplete pairs, inconsistent labels/flags/identities, non-finite
  probabilities, and pred != (prob >= threshold).
- mcnemar_exact: two-sided exact McNemar = exact binomial test on discordant pairs,
  p = min(1, 2 * sum_{i<=min(b,c)} C(b+c, i) / 2^(b+c)), computed in exact rational arithmetic.
- pair_bootstrap: percentile bootstrap over pair IDs (both modalities move together; repeated
  draws keep multiplicity); undefined replicates counted, excluded from quantiles, never redrawn.
"""
import math
from fractions import Fraction

import numpy as np

MODS = ("dermoscopy", "clinical")


class PairError(ValueError):
    pass


def build_pairs(rows, threshold, threshold_sha, checkpoint_sha):
    pairs = {}
    seen = set()
    for r in rows:
        u = r["record_uid"]
        if u in seen:
            raise PairError(f"duplicate record_uid {u}")
        seen.add(u)
        if r["modality"] not in MODS:
            raise PairError(f"{u}: bad modality {r['modality']}")
        p = float(r["prob"])
        if not math.isfinite(p) or r.get("status", "ok") != "ok":
            raise PairError(f"{u}: non-finite probability or failed status")
        if r["threshold_sha256"] != threshold_sha or r["checkpoint_sha256"] != checkpoint_sha:
            raise PairError(f"{u}: model/threshold identity mismatch")
        if float(r["threshold"]) != threshold:
            raise PairError(f"{u}: threshold value mismatch")
        pred = int(r["pred"])
        if pred != int(p >= threshold):
            raise PairError(f"{u}: pred != (prob >= threshold)")
        slot = pairs.setdefault(r["pair_id"], {})
        if r["modality"] in slot:
            raise PairError(f"pair {r['pair_id']}: duplicate {r['modality']} record")
        slot[r["modality"]] = {"record_uid": u, "label": int(r["label_mel"]), "prob": p, "pred": pred,
                               "primary": r["in_derm7pt_primary"], "same_file": r["same_file_pair"]}
    out = {}
    for pid, s in pairs.items():
        if set(s) != set(MODS):
            raise PairError(f"pair {pid}: incomplete ({sorted(s)})")
        d, c = s["dermoscopy"], s["clinical"]
        if d["label"] != c["label"] or d["label"] not in (0, 1):
            raise PairError(f"pair {pid}: label inconsistent")
        if d["primary"] != c["primary"] or d["same_file"] != c["same_file"]:
            raise PairError(f"pair {pid}: cohort flag inconsistent")
        out[pid] = {"pair_id": pid, "label": d["label"], "primary": d["primary"] == "1",
                    "same_file": d["same_file"] == "1", "pred_d": d["pred"], "pred_c": c["pred"],
                    "prob_d": d["prob"], "prob_c": c["prob"],
                    "uid_d": d["record_uid"], "uid_c": c["record_uid"]}
    return out


def paired_table(pairs, label):
    t = {"both": 0, "first_only": 0, "second_only": 0, "neither": 0}
    for p in pairs:
        if p["label"] != label:
            continue
        a, b = p["pred_d"], p["pred_c"]
        key = {(1, 1): "both", (1, 0): "first_only", (0, 1): "second_only", (0, 0): "neither"}[(a, b)]
        t[key] += 1
    return t


def mcnemar_exact(b, c):
    """Two-sided exact McNemar (binomial, p=0.5). Returns dict with exact Fraction and float."""
    if b < 0 or c < 0 or int(b) != b or int(c) != c:
        raise ValueError("counts must be non-negative integers")
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_exact": Fraction(1), "p_value": 1.0}
    k = min(b, c)
    tail = Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2 ** n)
    p = min(Fraction(1), 2 * tail)
    return {"b": b, "c": c, "n_discordant": n, "p_exact": p, "p_value": float(p),
            "p_value_sci": f"{float(p):.3e}", "log10_p": math.log10(p.numerator) - math.log10(p.denominator)}


def sens_diff(pairs_list, label=1):
    """(rate_d, rate_c, diff) of pred==label over pairs with that label; None if no such pairs.
    label=1 -> sensitivities; label=0 -> specificities (pred==0)."""
    sel = [p for p in pairs_list if p["label"] == label]
    if not sel:
        return None
    if label == 1:
        rd = sum(p["pred_d"] for p in sel) / len(sel)
        rc = sum(p["pred_c"] for p in sel) / len(sel)
    else:
        rd = sum(1 - p["pred_d"] for p in sel) / len(sel)
        rc = sum(1 - p["pred_c"] for p in sel) / len(sel)
    return rd, rc, rd - rc


def resample_ids(ids, rng):
    """One bootstrap replicate: len(ids) draws of pair IDs with replacement (multiplicity kept)."""
    idx = rng.integers(0, len(ids), size=len(ids))
    return [ids[i] for i in idx]


def pair_bootstrap(pairs_by_id, ids, stat, B, seed, alpha=0.05, method="linear"):
    """stat(list_of_pair_dicts) -> tuple of floats or None (undefined). Resampling unit: pair ID."""
    rng = np.random.Generator(np.random.PCG64(seed))
    reps, undefined = [], 0
    for _ in range(B):
        draw = resample_ids(ids, rng)
        v = stat([pairs_by_id[i] for i in draw])
        if v is None or not all(math.isfinite(x) for x in v):
            undefined += 1
            continue
        reps.append(v)
    arr = np.array(reps, dtype=float)
    q = [alpha / 2, 1 - alpha / 2]
    ci = np.quantile(arr, q, axis=0, method=method).T.tolist() if len(reps) else None
    return {"B": B, "seed": seed, "n_defined": len(reps), "n_undefined": undefined,
            "undefined_fraction": undefined / B, "unreliable": undefined / B > 0.05,
            "quantiles": q, "quantile_method": f"numpy.quantile(method='{method}')",
            "ci": ci, "replicates": arr}
