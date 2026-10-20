"""Stage 11A: E4.6 data audit and frozen evaluation design (no feature values, no model fitting).

Protocol 02_rev2 §6.7 E4.6 (authoritative text): "does a model built on deployment-available features computed from the
clinical image alone improve error prediction over a probability-only baseline? Method: 5-fold cross-validation grouped
by case. Features that need the twin image are reported as mechanistic only." Everything else here is a RETROSPECTIVE
ADDITION, frozen before any feature value is computed or any model is fitted.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.e46_design --config ../configs/e46_design_11a_v1.json --out-root ../runs/e46_design_11a
"""
import argparse
import csv
import io
import json
import os

R4A = "research_v1/runs/predict_v1/20261001T183557Z_20b3af67_4959f943/"
R7A = "research_v1/runs/xai_prod_7a/20261003T173258Z_3b655f88_548b3a05"
K = 5
FOLD_SALT = "e46_folds_v1"
MIN_ERRORS_PER_FOLD = 10
MIN_NONERRORS_PER_FOLD = 10

# Fields a feature builder may read from a Stage 7A clinical record / 4A prediction row (whitelist).
FEATURE_INPUT_FIELDS = {
    "prediction": ("logit", "prob"),
    "xai_record": ("ig_completeness.failure", "explainers.ig512.(primary_curve|as_computed_curve).auc_insertion_v1/auc_deletion_v1",
                   "controls.random.auc_ins_each/auc_del_each", "explainers.gradcam.eligibility.map_flags",
                   "maps.npz: ig512_primary, gradcam (clinical reference row only)"),
}
FORBIDDEN = ("label_mel", "error", "correct", "pred", "threshold", "dermoscopy", "pair_partner", "mask", "mismatch", "donor",
             "D_shift", "modality_difference")


# ------------------------------------------------------------------ pure helpers (tested)
def outcome(label_mel, prob, threshold):
    """error = 1 when the clinical-image classification (prob >= threshold -> melanoma) disagrees with the label."""
    pred = 1 if prob >= threshold else 0
    return int(pred != int(label_mel))


def assign_folds(cases, strata, k=K, salt=FOLD_SALT):
    """Deterministic case-grouped, stratified assignment. cases: list of case ids (unique); strata: {case: stratum}.
    Within each stratum, cases are ordered by stable_seed(salt, case) and dealt round-robin starting at a stratum-specific
    offset (continuing the deal across strata) so fold sizes stay balanced."""
    from .seeds import stable_seed
    if len(set(cases)) != len(cases):
        raise ValueError("duplicate case")
    out, pos = {}, 0
    for s in sorted({strata[c] for c in cases}):
        for c in sorted((c for c in cases if strata[c] == s), key=lambda c: (stable_seed(salt, c), c)):
            out[c] = pos % k
            pos += 1
    return out


def check_folds(folds, err, k=K):
    per = {f: {"n": 0, "errors": 0} for f in range(k)}
    for c, f in folds.items():
        per[f]["n"] += 1
        per[f]["errors"] += err[c]
    ok = all(v["errors"] >= MIN_ERRORS_PER_FOLD and v["n"] - v["errors"] >= MIN_NONERRORS_PER_FOLD for v in per.values())
    return ok, per


def leakage_check(feature_names):
    bad = [f for f in feature_names if any(x.lower() in f.lower() for x in FORBIDDEN)]
    if bad:
        raise ValueError(f"forbidden predictor(s): {bad}")
    return True


FEATURES = {
    "prob_basis": {"set": "M0, M1, M2", "definition": "restricted cubic spline of logit(p) (p = Stage 4A clinical probability, fixed "
                   "model), 4 knots at the 5/35/65/95th percentiles of the TRAINING fold logits -> 3 columns (linear + 2 RCS "
                   "terms, Harrell form, knots fitted in-fold)", "source": "Stage 4A predictions_derm7pt_all.csv (logit)",
                   "note": "nonlinear so that errors at both low and high p can be captured by the probability-only baseline"},
    "ig_delta_ins": {"set": "M1, M2", "definition": "AUC_ins(IG m=512 positive map) - mean of the 5 random-map AUC_ins (controls_v1); "
                     "primary curve if valid, otherwise the stored as-computed curve", "source": "Stage 7A records (clinical row)"},
    "ig_delta_del": {"set": "M1, M2", "definition": "mean of the 5 random-map AUC_del - AUC_del(IG map); same curve rule",
                     "source": "Stage 7A records"},
    "ig_completeness_failure": {"set": "M1, M2", "definition": "1 if the clinical IG map failed §6.6 completeness, else 0 "
                                "(label-free numerical diagnostic)", "source": "Stage 7A records"},
    "ig_conc_top10": {"set": "M1, M2", "definition": "saliency concentration: sum of the top 10% (5,018 of 50,176) pixel values of "
                      "the IG positive map divided by the total positive mass; undefined (missing) if the map is all zero",
                      "source": "Stage 7A maps.npz ig512_primary (CPU)"},
    "gradcam_zero": {"set": "M1, M2", "definition": "1 if the clinical Grad-CAM positive map is all-zero/constant, else 0",
                     "source": "Stage 7A records"},
    "gradcam_conc_top10": {"set": "M1, M2", "definition": "as ig_conc_top10 for the Grad-CAM map; missing when gradcam_zero = 1",
                           "source": "Stage 7A maps.npz gradcam (CPU)"},
    "d_baseline": {"set": "M2 only (secondary)", "definition": "p_full - p_base: model response to the blurred baseline of the same "
                   "clinical image (not an explanation; a model-perturbation response)", "source": "Stage 7A curve records"},
}
EXCLUDED = {"true label / correctness / prediction-vs-label": "outcome-defining only (outcome and fold stratification)",
            "dermoscopic twin and any modality difference (D, drift)": "needs the twin image -> mechanistic only, not used",
            "ground-truth lesion masks / localisation": "not available at deployment (and not available for Derm7pt)",
            "mismatch controls": "donor chosen by label -> label-dependent",
            "predictive entropy": "deterministic function of p; would add no XAI information (not used)",
            "centre/Sobel controls": "excluded to keep the feature set small (frozen)"}


# ------------------------------------------------------------------ build
def _abs(cfg, rel):
    from . import config as C
    return C.resolve(cfg, rel)


def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    from . import config as C, provenance as P, xai_prod as XP
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    cfg = C.load(a.config)
    thr = XP._read_json(_abs(cfg, R4A + "threshold.json"))
    t = thr["threshold"]
    msha = thr.get("model_sha256") or thr.get("model", {}).get("sha256")
    if msha != XP.MODEL_SHA or thr["status"] != "ESTIMATED":
        raise RuntimeError("threshold not bound to the seed-0 model")
    preds = XP._read_csv_rows(_abs(cfg, R4A + "predictions_derm7pt_all.csv"))
    cl = [r for r in preds if r["modality"] == "clinical"]
    if any(r["checkpoint_sha256"] != XP.MODEL_SHA or float(r["threshold"]) != t or int(r["pred"]) != int(float(r["prob"]) >= t) for r in cl):
        raise RuntimeError("prediction identity/rule")
    run7 = _abs(cfg, R7A)
    s7, s7sha = XP._load(run7, "spec.json")
    x7 = {}
    for ch in s7["chunks"]:
        d = os.path.join(run7, f"chunk_{ch['chunk']:02d}")
        if not XP.verify_dir(d, s7sha):
            raise RuntimeError(d)
        for r in XP._read_json(os.path.join(d, "records.json")):
            if r["modality"] == "clinical":
                x7[r["record_uid"]] = {"chunk": ch["chunk"], "ig_fail": r["ig_completeness"]["failure"],
                                       "gc_zero": r["explainers"]["gradcam"]["status"] != "primary",
                                       "ig_status": r["explainers"]["ig512"]["status"], "logit_7a": r["logit_x"]}
    by = {r["record_uid"]: r for r in cl}
    rows_full, rows_xai = [], []
    for r in cl:
        if r["in_derm7pt_primary"] != "1":
            continue
        e = outcome(r["label_mel"], float(r["prob"]), t)
        rows_full.append({"case_id": r["pair_id"], "record_uid": r["record_uid"], "label_mel": r["label_mel"], "error": e,
                          "error_type": ("FN" if r["label_mel"] == "1" else "FP") if e else "", "in_xai_cohort": int(r["record_uid"] in x7)})
    for u, v in sorted(x7.items()):
        r = by[u]
        e = outcome(r["label_mel"], float(r["prob"]), t)
        rows_xai.append({"case_id": r["pair_id"], "record_uid": u, "label_mel": r["label_mel"], "error": e,
                         "error_type": ("FN" if r["label_mel"] == "1" else "FP") if e else "", "ig_completeness_failure": int(v["ig_fail"]),
                         "gradcam_zero": int(v["gc_zero"]), "xai_chunk": v["chunk"],
                         "logit_abs_diff_7a_vs_4a": abs(v["logit_7a"] - float(r["logit"]))})
    for name, rows in (("full", rows_full), ("xai", rows_xai)):
        if len({x["case_id"] for x in rows}) != len(rows):
            raise RuntimeError(f"{name}: duplicate case")
    folds = {}
    for name, rows in (("full", rows_full), ("xai", rows_xai)):
        strata = {x["case_id"]: f"{x['label_mel']}|{x['error']}" for x in rows}
        f = assign_folds([x["case_id"] for x in rows], strata)
        ok, per = check_folds(f, {x["case_id"]: x["error"] for x in rows})
        if not ok:
            raise RuntimeError(f"{name}: fold sufficiency {per}")
        for x in rows:
            x["fold"] = f[x["case_id"]]
        folds[name] = per
    sub = {"ig_valid": [x for x in rows_xai if not x["ig_completeness_failure"]],
           "gradcam_nonzero": [x for x in rows_xai if not x["gradcam_zero"]]}
    sub_folds = {k: check_folds({x["case_id"]: x["fold"] for x in v}, {x["case_id"]: x["error"] for x in v})[1] for k, v in sub.items()}

    def comp(rows):
        return {"cases": len(rows), "images": len(rows), "mel": sum(x["label_mel"] == "1" for x in rows),
                "nonmel": sum(x["label_mel"] == "0" for x in rows), "errors": sum(x["error"] for x in rows),
                "FN": sum(x["error_type"] == "FN" for x in rows), "FP": sum(x["error_type"] == "FP" for x in rows)}
    leakage_check(list(FEATURES))
    spec = {"spec_version": "e46_design_11a_v1", "created_utc": _now(),
            "authoritative_text": "02_rev2 §6.7 E4.6 (see module docstring); §9 row 8: v0 in-sample drift model -> grouped CV with "
                                  "deployment-available features",
            "status": "exploratory, retrospective internal cross-validation; class-enriched, previously examined XAI cohort; "
                      "no deployment validation; no general-population error prevalence; Derm7pt 12-test Holm family untouched",
            "model": {"checkpoint_sha256": XP.MODEL_SHA, "threshold": t, "threshold_source": R4A + "threshold.json",
                      "predictions": R4A + "predictions_derm7pt_all.csv"},
            "outcome": "error = 1 if (prob_clinical >= threshold) != label_mel (FN for melanoma, FP for non-melanoma); the label "
                       "defines the outcome and fold strata only",
            "cohorts": {"full_clinical_primary": {"definition": "Derm7pt PRIMARY clinical images (1 per case)", **comp(rows_full),
                                                  "use": "outcome prevalence in all Derm7pt primary clinical cases and probability-only "
                                                         "context models (M0)"},
                        "xai_clinical": {"definition": "clinical references of the frozen Stage 7A XAI cohort (498 cases; all 249 "
                                                       "melanoma pairs + 249 non-melanoma); NO dermoscopic eligibility required",
                                         **comp(rows_xai), "use": "PRIMARY E4.6 comparison M0 vs M1 (and M2 secondary)"},
                        "method_subsets": {k: comp(v) for k, v in sub.items()},
                        "note": "the 373 (IG) and 275 (Grad-CAM) paired modality cohorts are NOT used"},
            "features": FEATURES, "excluded_predictors": EXCLUDED, "feature_input_whitelist": FEATURE_INPUT_FIELDS,
            "models": {"family": "L2-penalised logistic regression (sklearn LogisticRegression, lbfgs, C=1.0 fixed, max_iter=5000, "
                                 "class_weight=None, intercept)", "tuning": "none (fixed settings)",
                       "M0": "probability-only: prob_basis (3 spline columns)", "M0_linear_reference": "logit(p) only (reported, "
                       "not a comparator)", "M1": "M0 + ig_delta_ins, ig_delta_del, ig_completeness_failure, ig_conc_top10, "
                       "gradcam_zero, gradcam_conc_top10", "M2": "M1 + d_baseline (secondary)",
                       "in_fold_processing": "spline knots, median imputation of missing concentration values, standardisation: "
                                             "fitted on the training folds only, applied to the held-out fold",
                       "missing_maps": "IG/Grad-CAM curves exist for all 498 (as-computed for invalid maps, with the validity "
                                       "indicator as a feature); concentration missing only for all-zero maps -> in-fold median "
                                       "imputation; the zero-map indicator carries the missingness"},
            "cross_validation": {"k": K, "grouping": "case (one clinical image per case; case_id = pair_id = cluster)",
                                 "stratification": "label x error, deterministic stable_seed order, round-robin",
                                 "fold_sufficiency": f">= {MIN_ERRORS_PER_FOLD} errors and >= {MIN_NONERRORS_PER_FOLD} non-errors per fold",
                                 "fold_counts": folds, "subset_fold_counts": sub_folds,
                                 "identical_folds": "M0, M1 and M2 use the same folds and the same cases in every comparison"},
            "metrics": {"primary": "out-of-fold AUROC for error (pooled OOF predictions over the 5 held-out folds)",
                        "secondary": ["OOF Brier score", "OOF log-loss", "calibration intercept and slope (descriptive)",
                                      "per-fold AUROCs (descriptive only; not 5 independent experiments)"],
                        "comparison": "Delta AUROC = AUROC(M1) - AUROC(M0) on identical cases/OOF predictions; also Delta Brier",
                        "uncertainty": "paired case bootstrap of the fixed OOF predictions, B=2000, seed 20261300, percentile 95% "
                                       "CI; does NOT capture refitting or fold-assignment uncertainty",
                        "tests": "none (exploratory)",
                        "sensitivity": ["ig_valid subset (completeness pass), parent folds restricted", "M2 vs M0",
                                        "full clinical cohort: M0 vs M0_linear only (no XAI features available)"],
                        "interpretation": "XAI is 'useful' only as an exploratory, in-cohort statement if Delta AUROC CI excludes "
                                          "0 on the primary cohort; entropy-like reformulations of p are not XAI information"}}
    from . import provenance as Pv
    receipt = Pv.build_receipt(cfg, a.config, "completed", details={"phase": "design"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "e46_spec.json", spec)
    publish_csv(out, "cohort_full_clinical.csv", rows_full, list(rows_full[0]))
    publish_csv(out, "cohort_xai_clinical_folds.csv", rows_xai, list(rows_xai[0]))
    publish_json(out, "feature_dictionary.json", {"features": FEATURES, "excluded": EXCLUDED, "whitelist": FEATURE_INPUT_FIELDS,
                                                  "forbidden_substrings": FORBIDDEN})
    for n in ("e46_spec.json", "cohort_full_clinical.csv", "cohort_xai_clinical_folds.csv", "feature_dictionary.json"):
        publish_bytes(out, n + ".sha256", f"{P.sha256_file(os.path.join(out, n))}  {n}\n".encode())
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out)) if not n.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_design.json")
    print(json.dumps({k: spec["cohorts"][k] for k in ("full_clinical_primary", "xai_clinical", "method_subsets")}, indent=1))
    print(json.dumps(folds, indent=0), json.dumps(sub_folds))
    print("max 7A-4A logit diff", max(x["logit_abs_diff_7a_vs_4a"] for x in rows_xai), "verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


if __name__ == "__main__":
    main()
