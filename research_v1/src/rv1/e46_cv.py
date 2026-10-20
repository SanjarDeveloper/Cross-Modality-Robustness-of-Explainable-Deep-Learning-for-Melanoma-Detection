"""Stage 11B: E4.6 clinical-only feature table and frozen cross-validation (Stage 11A spec + clarification note v1).

Phases:
  features  build the frozen feature table from saved Stage 4A predictions and Stage 7A clinical artefacts (CPU only),
            with per-feature provenance and missingness reasons
  cv        fit M0 / M0_linear / M1 / M2 with fold-local spline knots, imputation and scaling; out-of-fold predictions;
            metrics; paired case bootstrap of the fixed OOF predictions

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.e46_cv --config ../configs/e46_cv_11b_v1.json --phase features --out-root ../runs/e46_cv_11b
  ... --phase cv --run-dir <dir>
"""
import argparse
import json
import math
import os
import time

import numpy as np

R11A = "research_v1/runs/e46_design_11a/20261010T054550Z_620b65e3_2926da95"
SPEC_SHA = "8c4ebedc309b4d5136cf13810d7fb43f07cb6fc45d7f6235eddb414393f838ec"
R4A = "research_v1/runs/predict_v1/20261001T183557Z_20b3af67_4959f943/"
R7A = "research_v1/runs/xai_prod_7a/20261003T173258Z_3b655f88_548b3a05"
TOPFRAC = 0.10
KNOT_Q = (5, 35, 65, 95)
C_REG, MAX_ITER = 1.0, 5000
B, SEED = 2000, 20261300
XAI_COLS = ("ig_delta_ins", "ig_delta_del", "ig_completeness_failure", "ig_conc_top10", "gradcam_zero", "gradcam_conc_top10")
IMPUTE_COLS = ("ig_conc_top10", "gradcam_conc_top10")
MODELS = {"M0": ("spline",), "M0_linear": ("linear",), "M1": ("spline", *XAI_COLS), "M2": ("spline", *XAI_COLS, "d_baseline")}
XAI_RECORD_KEYS = ("record_uid", "ig_completeness", "explainers", "controls")   # whitelist passed to the builder


# ------------------------------------------------------------------ feature builder (tested)
def sanitize(rec):
    """Whitelisted view of a Stage 7A clinical record: no label, pair partner, mismatch or donor fields."""
    out = {k: rec[k] for k in XAI_RECORD_KEYS}
    out["explainers"] = {e: {k: v for k, v in out["explainers"][e].items() if k in ("status", "eligibility", "primary_curve", "as_computed_curve")}
                         for e in ("ig512", "gradcam")}
    out["controls"] = {"random": {k: out["controls"]["random"][k] for k in ("status", "auc_ins_each", "auc_del_each", "curves")}}
    return out


def concentration(m, frac=TOPFRAC):
    a = np.asarray(m, dtype=np.float64).reshape(-1)
    tot = a.sum()
    if not np.isfinite(tot) or tot <= 0:
        return None
    k = int(round(frac * a.size))
    return float(np.sort(a)[::-1][:k].sum() / tot)


def build_features(pred, xrec, ig_map, gc_map):
    """pred: {'logit','prob'} (4A clinical); xrec: sanitize(record); maps: clinical IG primary and Grad-CAM arrays.
    Returns (features, provenance)."""
    f, prov = {}, {}
    f["logit"], f["prob"] = float(pred["logit"]), float(pred["prob"])
    ig = xrec["explainers"]["ig512"]
    src = "primary_curve" if ig["status"] == "primary" else "as_computed_curve"
    cv = ig.get(src)
    rnd = xrec["controls"]["random"]
    if cv is None or cv.get("curve_status") != "ok" or rnd["status"] != "available":
        f["ig_delta_ins"] = f["ig_delta_del"] = None
        prov["ig_delta"] = "missing: no usable IG curve or random control"
    else:
        f["ig_delta_ins"] = cv["auc_insertion_v1"] - float(np.mean(rnd["auc_ins_each"]))
        f["ig_delta_del"] = float(np.mean(rnd["auc_del_each"])) - cv["auc_deletion_v1"]
        prov["ig_delta"] = src + ("" if src == "primary_curve" else " (NOT a valid explanation: " + ";".join(ig["eligibility"]["ineligibility_reasons"]) + ")")
    f["ig_completeness_failure"] = int(bool(xrec["ig_completeness"]["failure"]))
    f["ig_conc_top10"] = concentration(ig_map)
    prov["ig_conc_top10"] = "ig512_primary map" if f["ig_conc_top10"] is not None else "missing: all-zero IG positive map"
    f["gradcam_zero"] = int(xrec["explainers"]["gradcam"]["status"] != "primary")
    f["gradcam_conc_top10"] = None if f["gradcam_zero"] else concentration(gc_map)
    prov["gradcam_conc_top10"] = "gradcam map" if f["gradcam_conc_top10"] is not None else "missing: all-zero/constant Grad-CAM map"
    c0 = rnd["curves"][0]
    f["d_baseline"] = c0["p_full"] - c0["p_base"]
    prov["d_baseline"] = "p_full - p_base (random curve 0; model response, not an explanation)"
    return f, prov


# ------------------------------------------------------------------ models (tested)
def fit_logistic(X, y, C=C_REG, penalised=True, tol=1e-10, max_iter=100):
    """Exact Newton-Raphson for the sklearn LogisticRegression objective: 0.5*||w||^2 + C*sum(logloss), intercept
    unpenalised (penalised=False: plain maximum likelihood). Returns (w, b, n_iter, converged). scikit-learn is not
    importable in .venv_model, so the frozen objective is solved here (Stage 11A clarification note v1, section 5)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, d = X.shape
    Z = np.hstack([X, np.ones((n, 1))])
    beta = np.zeros(d + 1)
    lam = np.zeros(d + 1)
    if penalised:
        lam[:d] = 1.0 / C
    for it in range(1, max_iter + 1):
        eta = Z @ beta
        p = 1 / (1 + np.exp(-eta))
        g = Z.T @ (p - y) + lam * beta
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(lam)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(g)) < tol or np.max(np.abs(step)) < tol:
            return beta[:d], float(beta[d]), it, True
    return beta[:d], float(beta[d]), max_iter, False


def rcs_basis(x, knots):
    """Harrell restricted cubic spline, 4 knots -> [x, s1, s2]; s_j = ((x-t_j)+^3 - (x-t3)+^3 (t4-t_j)/(t4-t3)
    + (x-t4)+^3 (t3-t_j)/(t4-t3)) / (t4-t1)^2."""
    x = np.asarray(x, dtype=np.float64)
    t1, t2, t3, t4 = knots
    if not (t1 < t2 < t3 < t4):
        raise ValueError(f"degenerate knots {knots}")
    p = lambda z: np.maximum(z, 0.0) ** 3
    cols = [x]
    for tj in (t1, t2):
        cols.append((p(x - tj) - p(x - t3) * (t4 - tj) / (t4 - t3) + p(x - t4) * (t3 - tj) / (t4 - t3)) / (t4 - t1) ** 2)
    return np.column_stack(cols)


class FoldModel:
    """Fold-local preprocessing + L2 logistic regression; all parameters learned from the training rows only."""

    def __init__(self, cols):
        self.cols = cols

    def _design(self, F, fit):
        parts, names = [], []
        if "spline" in self.cols:
            if fit:
                self.knots = tuple(float(v) for v in np.percentile(F["logit"], KNOT_Q))
            parts.append(rcs_basis(F["logit"], self.knots))
            names += ["logit", "rcs1", "rcs2"]
        if "linear" in self.cols:
            parts.append(np.asarray(F["logit"], float)[:, None])
            names.append("logit")
        for c in self.cols:
            if c in ("spline", "linear"):
                continue
            v = np.array([np.nan if x is None else x for x in F[c]], dtype=np.float64)
            if c in IMPUTE_COLS:
                if fit:
                    obs = v[~np.isnan(v)]
                    if obs.size == 0:
                        raise ValueError(f"all-missing training column {c}")
                    self.medians = getattr(self, "medians", {})
                    self.medians[c] = float(np.median(obs))
                v = np.where(np.isnan(v), self.medians[c], v)
            if np.isnan(v).any():
                raise ValueError(f"unexpected missing values in {c}")
            parts.append(v[:, None])
            names.append(c)
        X = np.hstack(parts)
        if fit:
            self.mean, self.sd = X.mean(0), X.std(0)
            if (self.sd == 0).any():
                raise ValueError(f"zero-variance training column {np.array(names)[self.sd == 0]}")
            self.names = names
        return (X - self.mean) / self.sd

    def fit(self, F, y):
        X = self._design(F, True)
        self.w, self.b, self.n_iter, self.converged = fit_logistic(X, y)
        return self

    def predict(self, F):
        return 1 / (1 + np.exp(-(self._design(F, False) @ self.w + self.b)))

    def params(self):
        return {"knots": getattr(self, "knots", None), "medians": getattr(self, "medians", {}), "mean": self.mean.tolist(),
                "sd": self.sd.tolist(), "names": self.names, "coef": self.w.tolist(), "intercept": self.b, "solver": "numpy Newton-Raphson (sklearn objective, C=1)",
                "n_iter": self.n_iter, "converged": self.converged}


def subset(F, idx):
    return {k: [v[i] for i in idx] if isinstance(v, list) else np.asarray(v)[idx] for k, v in F.items()}


def oof(F, y, folds, cols, k=5):
    """One out-of-fold prediction per case; returns (pred, per-fold metadata)."""
    pred = np.full(len(y), np.nan)
    meta = []
    for f in range(k):
        tr = np.where(folds != f)[0]
        te = np.where(folds == f)[0]
        m = FoldModel(cols).fit(subset(F, tr), y[tr])
        pred[te] = m.predict(subset(F, te))
        meta.append(dict(fold=f, n_train=len(tr), n_test=len(te), **m.params()))
    if np.isnan(pred).any():
        raise RuntimeError("missing OOF prediction")
    return pred, meta


def auroc(y, p):
    from .rq2_stats import auc_ranks
    return auc_ranks(p, np.asarray(y, bool))


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(y, p):
    """Slope: logistic regression of y on logit(p) (unpenalised); intercept: calibration-in-the-large with offset."""
    lp = np.log(np.clip(p, 1e-15, 1 - 1e-15) / np.clip(1 - p, 1e-15, 1))
    slope = float(fit_logistic(lp[:, None], y, penalised=False)[0][0])
    a = 0.0
    for _ in range(100):
        q = 1 / (1 + np.exp(-(lp + a)))
        g, h = np.sum(y - q), np.sum(q * (1 - q))
        a += g / h
        if abs(g / h) < 1e-12:
            break
    return {"intercept": float(a), "slope": slope}


def paired_bootstrap(y, preds, comps, B_=B, seed=SEED):
    """preds: {model: OOF}; comps: [(a, b)] -> AUROC(a) - AUROC(b) and Brier(a) - Brier(b) per replicate, shared draws."""
    rng = np.random.Generator(np.random.PCG64(seed))
    y = np.asarray(y)
    n = len(y)
    reps = {f"{a}-{b}": {"dauc": [], "dbrier": []} for a, b in comps}
    reps.update({m: {"auc": []} for m in preds})
    und = 0
    for _ in range(B_):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.all() or not yb.any():
            und += 1
            continue
        au = {m: auroc(yb, p[idx]) for m, p in preds.items()}
        for m in preds:
            reps[m]["auc"].append(au[m])
        for a, b in comps:
            reps[f"{a}-{b}"]["dauc"].append(au[a] - au[b])
            reps[f"{a}-{b}"]["dbrier"].append(brier(yb, preds[a][idx]) - brier(yb, preds[b][idx]))
    ci = lambda v: np.quantile(v, [0.025, 0.975], method="linear").tolist()
    return {k: {kk: ci(vv) for kk, vv in v.items()} for k, v in reps.items()}, und


# ------------------------------------------------------------------ phases
def _abs(cfg, rel):
    from . import config as C
    return C.resolve(cfg, rel)


def phase_features(cfg, a):
    from . import provenance as P, xai_prod as XP
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    r11 = _abs(cfg, R11A)
    if P.sha256_file(os.path.join(r11, "e46_spec.json")) != SPEC_SHA:
        raise RuntimeError("11A spec hash")
    for n in ("e46_spec.json", "cohort_full_clinical.csv", "cohort_xai_clinical_folds.csv", "feature_dictionary.json"):
        with open(os.path.join(r11, n + ".sha256"), encoding="utf-8") as f:
            if f.read().split()[0] != P.sha256_file(os.path.join(r11, n)):
                raise RuntimeError(f"11A {n} hash")
    cohort = XP._read_csv_rows(os.path.join(r11, "cohort_xai_clinical_folds.csv"))
    preds = {r["record_uid"]: r for r in XP._read_csv_rows(_abs(cfg, R4A + "predictions_derm7pt_all.csv"))}
    run7 = _abs(cfg, R7A)
    s7, s7sha = XP._load(run7, "spec.json")
    want = {r["record_uid"] for r in cohort}
    rows, src_hash = [], {}
    for ch in s7["chunks"]:
        d = os.path.join(run7, f"chunk_{ch['chunk']:02d}")
        if not XP.verify_dir(d, s7sha):
            raise RuntimeError(d)
        z = np.load(os.path.join(d, "maps.npz"))
        keys = [str(k) for k in z["ref_keys"]]
        for i, r in enumerate(XP._read_json(os.path.join(d, "records.json"))):
            if r["record_uid"] not in want or r["modality"] != "clinical":
                continue
            if keys[i] != r["ref_key"]:
                raise RuntimeError("map/record order")
            pr = preds[r["record_uid"]]
            f, prov = build_features({"logit": pr["logit"], "prob": pr["prob"]}, sanitize(r), z["ig512_primary"][i], z["gradcam"][i])
            rows.append({"record_uid": r["record_uid"], **f, "prov": json.dumps(prov, sort_keys=True), "source_chunk": ch["chunk"]})
        src_hash[f"chunk_{ch['chunk']:02d}/chunk_manifest.json"] = P.sha256_file(os.path.join(d, "chunk_manifest.json"))
    if {r["record_uid"] for r in rows} != want or len(rows) != len(want):
        raise RuntimeError("feature coverage")
    order = {r["record_uid"]: i for i, r in enumerate(cohort)}
    rows.sort(key=lambda r: order[r["record_uid"]])
    from . import provenance as Pv
    receipt = Pv.build_receipt(cfg, a.config, "completed", details={"phase": "features"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_csv(out, "feature_table.csv", rows, list(rows[0]))
    miss = {c: sum(r[c] is None for r in rows) for c in ("ig_delta_ins", "ig_delta_del", "ig_conc_top10", "gradcam_conc_top10", "d_baseline")}
    publish_json(out, "feature_meta.json", {"n": len(rows), "missing": miss, "as_computed_ig_rows": sum("NOT a valid" in r["prov"] for r in rows),
                                             "sources": {"stage4a_predictions": P.sha256_file(_abs(cfg, R4A + "predictions_derm7pt_all.csv")),
                                                         "stage7a_chunks": src_hash, "stage11a_spec": SPEC_SHA},
                                             "builder_sha256": P.sha256_file(__file__)})
    for n in ("feature_table.csv", "feature_meta.json"):
        publish_bytes(out, n + ".sha256", f"{P.sha256_file(os.path.join(out, n))}  {n}\n".encode())
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out)) if not n.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_features.json")
    print("run dir", out, miss, "verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def _table(out, r11):
    from . import xai_prod as XP
    ft = XP._read_csv_rows(os.path.join(out, "feature_table.csv"))
    co = {r["record_uid"]: r for r in XP._read_csv_rows(os.path.join(r11, "cohort_xai_clinical_folds.csv"))}
    num = lambda v: None if v in ("", "None") else float(v)
    F = {c: [num(r[c]) for r in ft] for c in ("logit", "prob", *XAI_COLS, "d_baseline")}
    F["logit"] = np.array(F["logit"])
    y = np.array([int(co[r["record_uid"]]["error"]) for r in ft])
    folds = np.array([int(co[r["record_uid"]]["fold"]) for r in ft])
    return ft, F, y, folds


def _eval(y, preds, folds):
    out = {}
    for m, p in preds.items():
        out[m] = {"auroc": auroc(y, p), "brier": brier(y, p), "logloss": logloss(y, p), **calibration(y, p),
                  "per_fold_auroc_descriptive": [auroc(y[folds == f], p[folds == f]) for f in range(5)]}
    return out


def phase_cv(cfg, a):
    from . import provenance as P, xai_prod as XP
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    t0 = time.perf_counter()
    out = os.path.abspath(a.run_dir)
    for n in ("feature_table.csv", "feature_meta.json"):
        with open(os.path.join(out, n + ".sha256"), encoding="utf-8") as f:
            if f.read().split()[0] != P.sha256_file(os.path.join(out, n)):
                raise RuntimeError(f"{n} changed")
    r11 = _abs(cfg, R11A)
    ft, F, y, folds = _table(out, r11)
    res, oof_rows, meta = {}, [], {}
    # primary cohort: 498 XAI clinical
    preds = {}
    for m, cols in MODELS.items():
        preds[m], meta[f"primary|{m}"] = oof(F, y, folds, cols)
    comps = [("M1", "M0"), ("M2", "M0"), ("M0", "M0_linear")]
    ci, und = paired_bootstrap(y, preds, comps)
    ev = _eval(y, preds, folds)
    res["primary"] = {"n": int(len(y)), "errors": int(y.sum()), "metrics": ev, "bootstrap_ci95": ci, "undefined_replicates": und,
                      "differences": {f"{a_}-{b_}": {"dauc": ev[a_]["auroc"] - ev[b_]["auroc"], "dbrier": ev[a_]["brier"] - ev[b_]["brier"]}
                                      for a_, b_ in comps}}
    for i, r in enumerate(ft):
        oof_rows.append({"record_uid": r["record_uid"], "cohort": "primary_xai", "fold": int(folds[i]), "error": int(y[i]),
                         **{f"oof_{m}": float(p[i]) for m, p in preds.items()}})
    # sensitivity: IG-valid subset, refit within subset on restricted parent folds (clarification note v1 §3)
    idx = np.array([i for i in range(len(y)) if F["ig_completeness_failure"][i] == 0])
    Fs, ys, fs = subset(F, idx), y[idx], folds[idx]
    ps = {}
    sub_models = {"M0": MODELS["M0"], "M1": tuple(c for c in MODELS["M1"] if c != "ig_completeness_failure")}   # note v2
    for m, cols in sub_models.items():
        ps[m], meta[f"ig_valid|{m}"] = oof(Fs, ys, fs, cols)
    ci_s, und_s = paired_bootstrap(ys, ps, [("M1", "M0")], seed=SEED + 1)
    ev_s = _eval(ys, ps, fs)
    res["ig_valid_refit"] = {"model_note": "M1 here = M1 without ig_completeness_failure (structurally 0 in the subset; clarification note v2)", "n": int(len(ys)), "errors": int(ys.sum()), "metrics": ev_s, "bootstrap_ci95": ci_s, "undefined_replicates": und_s,
                             "differences": {"M1-M0": {"dauc": ev_s["M1"]["auroc"] - ev_s["M0"]["auroc"], "dbrier": ev_s["M1"]["brier"] - ev_s["M0"]["brier"]}}}
    for j, i in enumerate(idx):
        oof_rows.append({"record_uid": ft[i]["record_uid"], "cohort": "ig_valid_refit", "fold": int(fs[j]), "error": int(ys[j]),
                         "oof_M0": float(ps["M0"][j]), "oof_M1": float(ps["M1"][j])})
    # full clinical cohort: M0 vs M0_linear (probability only)
    full = XP._read_csv_rows(os.path.join(r11, "cohort_full_clinical.csv"))
    pr = {r["record_uid"]: r for r in XP._read_csv_rows(_abs(cfg, R4A + "predictions_derm7pt_all.csv"))}
    # full-cohort folds are not in the CSV for the full cohort -> recompute with the frozen 11A algorithm and verify counts
    from .e46_design import assign_folds, check_folds
    fd = assign_folds([r["case_id"] for r in full], {r["case_id"]: f"{r['label_mel']}|{r['error']}" for r in full})
    spec11 = XP._read_json(os.path.join(r11, "e46_spec.json"))
    per = check_folds(fd, {r["case_id"]: int(r["error"]) for r in full})[1]
    if {str(k): v for k, v in per.items()} != spec11["cross_validation"]["fold_counts"]["full"]:
        raise RuntimeError("full-cohort folds do not reproduce the frozen 11A counts")
    Ff = {"logit": np.array([float(pr[r["record_uid"]]["logit"]) for r in full])}
    yf = np.array([int(r["error"]) for r in full])
    ff = np.array([fd[r["case_id"]] for r in full])
    pf = {}
    for m in ("M0", "M0_linear"):
        pf[m], meta[f"full|{m}"] = oof(Ff, yf, ff, MODELS[m])
    ci_f, und_f = paired_bootstrap(yf, pf, [("M0", "M0_linear")], seed=SEED + 2)
    ev_f = _eval(yf, pf, ff)
    res["full_clinical"] = {"n": int(len(yf)), "errors": int(yf.sum()), "metrics": ev_f, "bootstrap_ci95": ci_f, "undefined_replicates": und_f,
                            "differences": {"M0-M0_linear": {"dauc": ev_f["M0"]["auroc"] - ev_f["M0_linear"]["auroc"]}}}
    conv = {k: [m_["converged"] for m_ in v] for k, v in meta.items()}
    if not all(all(v) for v in conv.values()):
        raise RuntimeError(f"non-convergence {conv}")
    res["convergence_all"] = True
    res["runtime_s"] = time.perf_counter() - t0
    res["notes"] = {"interval": "paired case bootstrap of fixed OOF predictions; excludes refitting and fold-assignment uncertainty",
                    "reading": "see STAGE11A_CLARIFICATION_NOTE_v1 §2", "per_fold": "descriptive only; not independent experiments"}
    publish_json(out, "cv_results.json", res)
    publish_json(out, "fold_models.json", meta)
    publish_csv(out, "oof_predictions.csv", oof_rows, sorted({k for r in oof_rows for k in r}, key=lambda k: list(oof_rows[0]).index(k) if k in oof_rows[0] else 99))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "cv"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out)) if not n.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_cv.json")
    print(json.dumps({k: res[k] for k in ("primary", "ig_valid_refit", "full_clinical")}, indent=1, default=str)[:6000])
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("features", "cv"))
    ap.add_argument("--out-root")
    ap.add_argument("--run-dir")
    a = ap.parse_args()
    from . import config as C
    cfg = C.load(a.config)
    {"features": phase_features, "cv": phase_cv}[a.phase](cfg, a)


if __name__ == "__main__":
    main()
