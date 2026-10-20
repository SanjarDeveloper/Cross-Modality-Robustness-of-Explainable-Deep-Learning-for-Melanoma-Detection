"""Stage 3B pilot logic (Addendum D): frozen pair selection, IG convergence assessment with
explicit denominators, batch-equivalence check, and paired eligibility. Pure Python/numpy.

Nothing here replaces a missing comparison with zero or drops a modality; every count and
denominator is returned with the result.
"""
import math

import numpy as np

from .seeds import stable_seed

PILOT_SALT = "stage3b_xai_pilot_v1"
FAIL_RATE_MAX = 0.05
P95_REL_MAX = 0.20
MEDIAN_DAUC_MAX = 0.005
ENDPOINT_TOL = 1e-4
BATCH_AUC_TOL = 1e-3


def select_pilot_pairs(xai_rows, derm_rows, n_per_label, salt=PILOT_SALT):
    """xai_rows: rows of xai_derm7pt_v1.csv; derm_rows: rows of derm7pt_v1.csv.
    Returns [{pair_id, label_mel, rank_key, label_rank, record_uid_<mod>, image_uid_<mod>}...]
    (melanoma first, then non-melanoma, each in rank order). Raises if any selected pair fails
    primary-cohort membership; nothing is silently skipped."""
    by_uid = {r["record_uid"]: r for r in derm_rows}
    ids = [r["pair_id"] for r in xai_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate pair_id in XAI manifest")
    out = []
    for lab in ("1", "0"):
        rs = sorted((r for r in xai_rows if r["label_mel"] == lab),
                    key=lambda r: (stable_seed(salt, r["pair_id"]), r["pair_id"]))
        if len(rs) < n_per_label:
            raise ValueError(f"label {lab}: only {len(rs)} pairs")
        for k, r in enumerate(rs[:n_per_label]):
            rec = {"pair_id": r["pair_id"], "label_mel": lab, "label_rank": k,
                   "rank_key": stable_seed(salt, r["pair_id"])}
            for mod in ("dermoscopy", "clinical"):
                u = r[f"record_uid_{mod}"]
                d = by_uid.get(u)
                problems = []
                if d is None:
                    problems.append("record missing from derm7pt manifest")
                else:
                    if d["in_derm7pt_primary"] != "1":
                        problems.append("not in primary cohort")
                    if d["modality"] != mod:
                        problems.append(f"modality {d['modality']}")
                    if d["pair_id"] != r["pair_id"]:
                        problems.append("pair_id mismatch")
                    if d["label_mel"] != lab:
                        problems.append("label mismatch")
                    if d["image_uid"] != r[f"image_uid_{mod}"]:
                        problems.append("image_uid mismatch")
                if problems:
                    raise ValueError(f"{u}: {problems}")
                rec[f"record_uid_{mod}"] = u
                rec[f"image_uid_{mod}"] = d["image_uid"]
            out.append(rec)
    return out


def batch_subset(pairs, n_per_label=2):
    """First n melanoma and first n non-melanoma pairs in frozen pilot order."""
    return [p["pair_id"] for p in pairs if p["label_mel"] == "1"][:n_per_label] + \
        [p["pair_id"] for p in pairs if p["label_mel"] == "0"][:n_per_label]


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def curve_usable(rec):
    """rec: per-reference IG record at one m with keys eligible (bool), curve_status,
    endpoint_status, auc_ins, auc_del."""
    return (rec.get("eligible") is True and rec.get("curve_status") == "ok" and
            rec.get("endpoint_status") == "pass" and _finite(rec.get("auc_ins")) and _finite(rec.get("auc_del")))


def assess_candidate(at_m, at_2m, refs):
    """at_m / at_2m: {ref_key: record} with keys completeness_failure (bool), e_rel (float|None),
    eligible, curve_status, endpoint_status, auc_ins, auc_del, map_flags. refs: the full list of
    reference keys (the denominator). A ref missing from at_m counts as a completeness failure
    (not computed) and is reported."""
    n = len(refs)
    missing = [r for r in refs if r not in at_m]
    fails = [r for r in refs if r in at_m and at_m[r]["completeness_failure"]]
    n_fail = len(fails) + len(missing)
    rel = [at_m[r]["e_rel"] for r in refs if r in at_m and at_m[r]["e_rel"] is not None]
    rel_ok = [v for v in rel if _finite(v)]
    n_undef = n - len(missing) - len(rel)
    both = [r for r in refs if r in at_m and r in at_2m and curve_usable(at_m[r]) and curve_usable(at_2m[r])]
    not_both = {r: _why_not(at_m.get(r), at_2m.get(r)) for r in refs if r not in both}
    out = {"n_refs": n, "n_missing": len(missing),
           "a": {"failures": n_fail, "denominator": n, "rate": n_fail / n if n else None},
           "b": {"n_defined": len(rel), "n_undefined": n_undef, "n_nonfinite_defined": len(rel) - len(rel_ok)},
           "c": {"n_eligible_both": len(both), "denominator_all": n, "not_eligible_both": not_both}}
    out["a"]["status"] = "UNRESOLVED" if n == 0 else ("pass" if n_fail / n <= FAIL_RATE_MAX else "fail")
    if not rel:
        out["b"].update(p95=None, status="UNRESOLVED")
    elif len(rel_ok) != len(rel):
        out["b"].update(p95=None, status="UNRESOLVED")
    else:
        p = float(np.percentile(rel_ok, 95))
        out["b"].update(p95=p, status="pass" if p <= P95_REL_MAX else "fail")
    for k, key in (("ins", "auc_ins"), ("del", "auc_del")):
        if not both:
            out["c"][k] = {"median_abs": None, "max_abs": None, "status": "UNRESOLVED"}
            continue
        d = [abs(at_m[r][key] - at_2m[r][key]) for r in both]
        med = float(np.median(d))
        out["c"][k] = {"median_abs": med, "max_abs": float(max(d)),
                       "status": "pass" if med < MEDIAN_DAUC_MAX else "fail"}
    st = [out["a"]["status"], out["b"]["status"], out["c"]["ins"]["status"], out["c"]["del"]["status"]]
    out["overall"] = "UNRESOLVED" if "UNRESOLVED" in st and "fail" not in st else \
        ("fail" if "fail" in st else "pass")
    return out


def _why_not(a, b):
    r = []
    for tag, rec in (("m", a), ("2m", b)):
        if rec is None:
            r.append(f"{tag}:not_computed")
            continue
        if rec.get("eligible") is not True:
            r += [f"{tag}:{x}" for x in (rec.get("reasons") or ["ineligible"])]
        elif rec.get("curve_status") != "ok":
            r.append(f"{tag}:curve_{rec.get('curve_status')}")
        elif rec.get("endpoint_status") != "pass":
            r.append(f"{tag}:endpoint_failure")
    return r or ["auc_nonfinite"]


def choose_m(assessments, candidates=(128, 256)):
    """Smallest candidate with overall 'pass'; else None (UNRESOLVED)."""
    for m in sorted(candidates):
        if assessments.get(m, {}).get("overall") == "pass":
            return m
    return None


def batch_equivalence(pairs_of_curves):
    """pairs_of_curves: list of (id, curve_b11, curve_b1). Returns summary with pass flag."""
    rows, ok = [], True
    for cid, c11, c1 in pairs_of_curves:
        r = {"id": cid, "endpoint_b11": c11.get("endpoint_status"), "endpoint_b1": c1.get("endpoint_status")}
        good = c11.get("curve_status") == "ok" and c1.get("curve_status") == "ok"
        if good:
            r["d_ins"] = abs(c11["auc_insertion_v1"] - c1["auc_insertion_v1"])
            r["d_del"] = abs(c11["auc_deletion_v1"] - c1["auc_deletion_v1"])
            r["max_point_diff"] = max(abs(a - b) for k in ("insertion", "deletion") for a, b in zip(c11[k], c1[k]))
            r["pass"] = (r["endpoint_b11"] == "pass" and r["endpoint_b1"] == "pass" and
                         r["d_ins"] <= BATCH_AUC_TOL and r["d_del"] <= BATCH_AUC_TOL)
        else:
            r["pass"] = False
        ok &= r["pass"]
        rows.append(r)
    return {"pass": bool(ok and rows), "n_compared": len(rows), "rows": rows,
            "max_d_auc": max((max(r["d_ins"], r["d_del"]) for r in rows if "d_ins" in r), default=None),
            "endpoint_tol": ENDPOINT_TOL, "auc_tol": BATCH_AUC_TOL}


def pair_table(pairs, expl_status, ctrl_status, explainers, controls):
    """expl_status[(record_uid, explainer)] = dict(usable: bool, reasons: [..]);
    ctrl_status[(record_uid, explainer, control)] = dict(available: bool, reason: str|None).
    Returns rows per (pair, explainer, contrast) where contrast is 'explainer' or a control name."""
    rows = []
    for p in pairs:
        uids = {m: p[f"record_uid_{m}"] for m in ("dermoscopy", "clinical")}
        for e in explainers:
            ex = {m: expl_status[(uids[m], e)] for m in uids}
            e_ok = all(ex[m]["usable"] for m in uids)
            e_reasons = [f"{m}:{x}" for m in uids for x in ex[m]["reasons"]] if not e_ok else []
            rows.append({"pair_id": p["pair_id"], "label_mel": p["label_mel"], "explainer": e,
                         "contrast": "explainer", "complete_pair": e_ok, "reasons": e_reasons})
            for c in controls:
                cs = {m: ctrl_status[(uids[m], e, c)] for m in uids}
                c_ok = e_ok and all(cs[m]["available"] for m in uids)
                reasons = list(e_reasons) if not e_ok else []
                reasons += [f"{m}:control:{cs[m]['reason']}" for m in uids if not cs[m]["available"] and
                            not (not ex[m]["usable"] and "explainer" in str(cs[m]["reason"]))]
                rows.append({"pair_id": p["pair_id"], "label_mel": p["label_mel"], "explainer": e,
                             "contrast": c, "complete_pair": c_ok, "reasons": [] if c_ok else reasons})
    return rows
