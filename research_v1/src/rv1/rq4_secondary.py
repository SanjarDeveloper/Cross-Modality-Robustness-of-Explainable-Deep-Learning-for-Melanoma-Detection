"""Stage 7C: RQ4 secondary analyses from verified Stage 7A/7B artifacts (CPU only):
  normalized scores (02_rev2 §6.5; delta 0.05 primary, 0.02/0.10 sensitivity), exploratory E4.3
  regression, and bootstrap uncertainty for centre / Sobel / mismatch control contrasts.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq4_secondary --config ../configs/rq4_secondary_7c_v1.json --phase spec --out-root ../runs/rq4_secondary_7c
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq4_secondary --config ../configs/rq4_secondary_7c_v1.json --phase analyze --run-dir <dir>
"""
import argparse
import datetime
import json
import math
import os

import numpy as np

B = 2000
DELTAS = (0.05, 0.02, 0.10)
EXPL = ("gradcam", "ig512")
MODS = ("dermoscopy", "clinical")


# ------------------------------------------------------------------ pure helpers (tested)
def norm_category(d, delta):
    if d >= delta:
        return "defined"
    if d <= -delta:
        return "baseline_more_melanoma_like"
    return "near_zero_denominator"


def normalized(auc_ins, auc_del, p_full, p_base, delta):
    """Returns (category, ins_norm, del_norm). No clipping; undefined unless d >= delta."""
    d = p_full - p_base
    cat = norm_category(d, delta)
    if cat != "defined":
        return cat, None, None
    return cat, (auc_ins - p_base) / d, (p_full - auc_del) / d


def norm_contrasts(expl, rand_mean, p_full, p_base, delta):
    """Control-referenced normalized contrasts using the SAME image's endpoints/denominator.
    expl, rand_mean: (auc_ins, auc_del). Equals raw contrasts / d (checked by tests)."""
    c1, ei, ed = normalized(expl[0], expl[1], p_full, p_base, delta)
    _, ri, rd = normalized(rand_mean[0], rand_mean[1], p_full, p_base, delta)
    if c1 != "defined":
        return c1, None, None
    return c1, ei - ri, ed - rd


def ols_fit(y, X):
    """OLS of y on [1, X]. Returns None if rank-deficient; else coefficients and diagnostics."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    A = np.column_stack([np.ones(len(y)), X])
    rank = int(np.linalg.matrix_rank(A))
    if rank < A.shape[1] or len(y) <= A.shape[1]:
        return None
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    sst = float(((y - y.mean()) ** 2).sum())
    return {"coef": coef.tolist(), "rank": rank, "cond": float(np.linalg.cond(A)), "n": int(len(y)),
            "resid_sd": float(np.sqrt((resid ** 2).sum() / (len(y) - A.shape[1]))),
            "r2": float(1 - (resid ** 2).sum() / sst) if sst > 0 else None,
            "resid_quantiles": np.quantile(resid, [0, 0.25, 0.5, 0.75, 1]).tolist()}


def bootstrap_ols(y, X, B_, seed):
    y, X = np.asarray(y, float), np.asarray(X, float)
    rng = np.random.Generator(np.random.PCG64(seed))
    reps, undefined = [], 0
    for _ in range(B_):
        idx = rng.integers(0, len(y), size=len(y))
        f = ols_fit(y[idx], X[idx])
        if f is None:
            undefined += 1
            continue
        reps.append(f["coef"])
    a = np.asarray(reps)
    ci = np.quantile(a, [0.025, 0.975], axis=0, method="linear").T.tolist() if len(reps) else None
    return {"ci95": ci, "n_undefined_rank_deficient": undefined, "unreliable": undefined / B_ > 0.05, "replicates": a}


def matched(rows, value_keys):
    """rows: per-image dicts with pair_id, modality and value keys; returns {pair: {mod: row}} for pairs
    whose values are all finite in BOTH modalities."""
    by = {}
    for r in rows:
        if r["modality"] in by.setdefault(r["pair_id"], {}):
            raise ValueError(f"duplicate {r['pair_id']}/{r['modality']}")
        by[r["pair_id"]][r["modality"]] = r
    out = {}
    for p, d in by.items():
        if set(d) != set(MODS):
            raise ValueError(f"incomplete pair {p}")
        if all(d[m].get(k) is not None and math.isfinite(d[m][k]) for m in MODS for k in value_keys):
            out[p] = d
    return out


# ------------------------------------------------------------------ phases
def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _paths(cfg):
    from . import config as C
    si = cfg["supporting_inputs"]
    return (C.resolve(cfg, si["stage7a_audit_receipt"]).rsplit(os.sep, 1)[0],
            C.resolve(cfg, si["stage7b_analyze_receipt"]).rsplit(os.sep, 1)[0])


def phase_spec(cfg, a):
    from . import provenance as P
    from .desc_4b3 import publish_bytes, publish_json
    r7a, r7b = _paths(cfg)
    rec7a = json.load(open(os.path.join(r7a, "receipt_audit.json"), encoding="utf-8"))
    rec7b = json.load(open(os.path.join(r7b, "receipt_analyze.json"), encoding="utf-8"))
    for base, rec in ((r7a, rec7a), (r7b, rec7b)):
        for n, h in rec["outputs"].items():
            if P.sha256_file(os.path.join(base, n)) != h:
                raise RuntimeError(f"changed {base}/{n}")
    s = cfg["seeds"]["bootstrap_seed"]
    spec = {"spec_version": "rq4_secondary_7c_v1", "created_utc": _now(),
            "inputs": {"stage7a_run": os.path.basename(r7a), "stage7a_receipt_outputs": rec7a["outputs"],
                       "stage7b_run": os.path.basename(r7b), "stage7b_receipt_outputs": rec7b["outputs"]},
            "normalization": {"source": "02_rev2 §6.5", "d": "p_full - p_base (from the image's stored curve endpoints)",
                              "ins_norm": "(ins_auc - p_base)/d", "del_norm": "(p_full - del_auc)/d",
                              "deltas": {"primary": 0.05, "sensitivity": [0.02, 0.10]},
                              "categories": "d>=delta defined; d<=-delta baseline_more_melanoma_like; |d|<delta near_zero",
                              "no_clipping": True,
                              "controls": "random-control AUC means normalized with the SAME recipient image's p_full, p_base, d",
                              "contrast": "ins: norm(expl)-norm(rand); del: norm(expl)-norm(rand) (= raw contrast / d)",
                              "population": "Stage 7B primary matched eligible pairs AND normalization defined in both modalities",
                              "reported": "raw contrasts on the normalized subset, normalized contrasts on the same subset, "
                                          "and the original raw-population result (Stage 7B) separately",
                              "inference": "descriptive median, HL, pair bootstrap 95% CI; no tests; not in the Holm family"},
            "E4_3": {"source": "02_rev2 §6.5 E4.3 (exploratory)",
                     "population": "Stage 7B primary raw matched eligible pairs (retrospective decision: the protocol does "
                                   "not restrict E4.3 to the normalization subset)",
                     "model": "D = b0 + b1*x1 + b2*x2 + e; D = raw contrast_clinical - raw contrast_dermoscopy; "
                              "x1 = p_full_clin - p_full_derm; x2 = d_clin - d_derm; uncentered covariates",
                     "fit": "OLS (numpy lstsq), rank check (matrix_rank), condition number, residual SD, R^2",
                     "intervals": "pair bootstrap percentile 95% (B=2000); rank-deficient replicates counted as undefined",
                     "interpretation": "b0 = predicted D at x1=x2=0 (not a universally adjusted effect); association only",
                     "retrospective_details": "fitting method, interval method and covariate centering are retrospective "
                                              "implementation decisions (not specified in the protocol)"},
            "alternative_controls": {"controls": ["centre", "sobel", "mismatch"], "population": "Stage 7B primary pairs with "
                                     "the control available in both modalities (mismatch: recipient + planned donor valid; "
                                     "frozen donor assignment)", "estimates": "median and HL per modality and D; pair "
                                     "bootstrap 95% CIs", "context": "random-control contrasts on the identical mismatch subset",
                                     "mismatch_note": "donor maps are reused across recipients; recipient-case bootstrap treats "
                                                      "the realized donor assignment as fixed"},
            "bootstrap": {"B": B, "seed_base": s, "rng": "numpy Generator(PCG64); integers(0,K,K); pairs shared within a block",
                          "quantiles": "numpy linear 0.025/0.975", "undefined": "counted; >5% flagged"},
            "missing_data": "no imputation; undefined categories reported by explainer x modality x class"}
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "analysis_spec.json", spec)
    h = P.sha256_file(os.path.join(out, "analysis_spec.json"))
    publish_bytes(out, "analysis_spec.json.sha256", f"{h}  analysis_spec.json\n".encode())
    receipt["outputs"] = {"analysis_spec.json": h}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out)


def _load_images(r7a, r7b):
    """Per explainer per image: status, raw expl AUCs, random means, p_full/p_base, raw contrasts and
    alternative-control contrasts (from verified 7A records / scores_v2)."""
    from .xai_prod import _read_csv_rows, _read_json
    scores = {(r["ref_key"], r["explainer"], r["control"]): r for r in _read_csv_rows(os.path.join(r7a, "scores_v2.csv"))}
    prim7b = {(r["explainer"], r["pair_id"]) for r in _read_csv_rows(os.path.join(r7b, "per_pair_inputs.csv"))
              if r["population"] == "primary"}
    out = {e: [] for e in EXPL}
    for d in sorted(os.listdir(r7a)):
        if not (d.startswith("chunk_") and os.path.isdir(os.path.join(r7a, d))):
            continue
        for r in _read_json(os.path.join(r7a, d, "records.json")):
            rnd = r["controls"]["random"]
            ri, rd = sum(rnd["auc_ins_each"]) / 5, sum(rnd["auc_del_each"]) / 5
            for e in EXPL:
                ex = r["explainers"][e]
                cv = ex.get("primary_curve") or ex.get("as_computed_curve") or {}
                row = {"ref_key": r["ref_key"], "pair_id": r["pair_id"], "modality": r["modality"],
                       "label_mel": r["label_mel"], "status": ex["status"], "primary_pair": (e, r["pair_id"]) in prim7b,
                       "p_full": cv.get("p_full"), "p_base": cv.get("p_base"),
                       "expl_ins": cv.get("auc_insertion_v1") if ex["status"] == "primary" else None,
                       "expl_del": cv.get("auc_deletion_v1") if ex["status"] == "primary" else None,
                       "rand_ins": ri, "rand_del": rd}
                row["d"] = row["p_full"] - row["p_base"] if row["p_full"] is not None else None
                for cn in ("random", "centre", "sobel", "mismatch"):
                    s = scores[(r["ref_key"], e, cn)]
                    ok = s["control_status"] == "available"
                    row[f"{cn}_ins"] = float(s["delta_ins"]) if ok else None
                    row[f"{cn}_del"] = float(s["delta_del"]) if ok else None
                out[e].append(row)
    return out


def phase_analyze(cfg, a):
    import io
    from . import provenance as P, rq4_stats as R4
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    out = os.path.abspath(a.run_dir)
    sp = os.path.join(out, "analysis_spec.json")
    if open(sp + ".sha256", encoding="utf-8").read().split()[0] != P.sha256_file(sp):
        raise RuntimeError("spec hash mismatch")
    spec = json.load(open(sp, encoding="utf-8"))
    r7a, r7b = _paths(cfg)
    for base, key in ((r7a, "stage7a_receipt_outputs"), (r7b, "stage7b_receipt_outputs")):
        for n, h in spec["inputs"][key].items():
            if P.sha256_file(os.path.join(base, n)) != h:
                raise RuntimeError(f"changed {n}")
    imgs = _load_images(r7a, r7b)
    s0 = spec["bootstrap"]["seed_base"]
    res, reps, cats_rows, pair_rows = {"spec_sha256": P.sha256_file(sp)}, {}, [], []
    maxdiff = 0.0
    seed = s0
    for e in EXPL:
        rows = imgs[e]
        prim_rows = [r for r in rows if r["primary_pair"]]
        raw_pop = matched(prim_rows, ("random_ins", "random_del"))
        if len(raw_pop) != {"gradcam": 275, "ig512": 373}[e]:
            raise RuntimeError(f"{e}: primary pairs {len(raw_pop)}")
        # ---- normalization per delta
        for delta in DELTAS:
            for r in rows:
                cat, ni, nd = (norm_contrasts((r["expl_ins"], r["expl_del"]), (r["rand_ins"], r["rand_del"]),
                                              r["p_full"], r["p_base"], delta) if r["status"] == "primary"
                               else (norm_category(r["d"], delta), None, None))
                r[f"cat_{delta}"], r[f"nins_{delta}"], r[f"ndel_{delta}"] = cat, ni, nd
                if ni is not None:
                    maxdiff = max(maxdiff, abs(ni - r["random_ins"] / r["d"]), abs(nd - r["random_del"] / r["d"]))
            sub = matched(prim_rows, (f"nins_{delta}", f"ndel_{delta}"))
            ids = sorted(sub)
            vecs = {}
            for c in ("ins", "del"):
                for m in MODS:
                    vecs[f"raw_subset|{c}|{m}"] = np.array([sub[p][m][f"random_{c}"] for p in ids])
                    vecs[f"norm|{c}|{m}"] = np.array([sub[p][m][f"n{c}_{delta}"] for p in ids])
                vecs[f"raw_subset|{c}|D"] = vecs[f"raw_subset|{c}|clinical"] - vecs[f"raw_subset|{c}|dermoscopy"]
                vecs[f"norm|{c}|D"] = vecs[f"norm|{c}|clinical"] - vecs[f"norm|{c}|dermoscopy"]
            bs, meta = R4.bootstrap_stats(vecs, B, seed) if ids else ({}, {"n_undefined": B})
            seed += 1
            blk = {"n_pairs": len(ids), "by_label": {lab: sum(sub[p]["dermoscopy"]["label_mel"] == lab for p in ids)
                                                     for lab in ("1", "0")}, "bootstrap": meta, "stats": {}}
            for k, v in vecs.items():
                blk["stats"][k] = {"n": len(v), "median": R4.sample_median(v), "hl": R4.hodges_lehmann(v),
                                   "median_ci95": bs[k]["median"]["ci95"] if ids else None,
                                   "hl_ci95": bs[k]["hl"]["ci95"] if ids else None}
                if ids:
                    reps[f"norm|{e}|{delta}|{k}"] = bs[k]["hl"]["replicates"]
            res[f"norm|{e}|{delta}"] = blk
            for m in MODS:
                for lab in ("1", "0"):
                    rr = [r for r in rows if r["modality"] == m and r["label_mel"] == lab]
                    cats_rows.append({"explainer": e, "delta": delta, "modality": m, "label_mel": lab, "n": len(rr),
                                      "defined": sum(r[f"cat_{delta}"] == "defined" for r in rr),
                                      "baseline_more_melanoma_like": sum(r[f"cat_{delta}"] == "baseline_more_melanoma_like" for r in rr),
                                      "near_zero_denominator": sum(r[f"cat_{delta}"] == "near_zero_denominator" for r in rr),
                                      "map_ineligible": sum(r["status"] != "primary" for r in rr),
                                      "defined_and_primary_pair": sum(r[f"nins_{delta}"] is not None and r["primary_pair"] for r in rr)})
        # ---- E4.3 on raw primary pairs
        ids = sorted(raw_pop)
        e43 = {}
        for c in ("ins", "del"):
            D = np.array([raw_pop[p]["clinical"][f"random_{c}"] - raw_pop[p]["dermoscopy"][f"random_{c}"] for p in ids])
            x1 = np.array([raw_pop[p]["clinical"]["p_full"] - raw_pop[p]["dermoscopy"]["p_full"] for p in ids])
            x2 = np.array([raw_pop[p]["clinical"]["d"] - raw_pop[p]["dermoscopy"]["d"] for p in ids])
            X = np.column_stack([x1, x2])
            fit = ols_fit(D, X)
            bt = bootstrap_ols(D, X, B, seed)
            seed += 1
            reps[f"e43|{e}|{c}"] = bt.pop("replicates")
            near0 = int(np.sum((np.abs(x1) < 0.05) & (np.abs(x2) < 0.05)))
            e43[c] = {"n": len(ids), "mean_D": float(D.mean()), "median_D": R4.sample_median(D), "hl_D": R4.hodges_lehmann(D),
                      "x1_range": [float(x1.min()), float(x1.max())], "x2_range": [float(x2.min()), float(x2.max())],
                      "x1_x2_correlation": float(np.corrcoef(x1, x2)[0, 1]),
                      "n_pairs_with_abs_x1_and_x2_below_0.05": near0,
                      "reference_point_x1=x2=0_inside_observed_ranges": bool(x1.min() <= 0 <= x1.max() and x2.min() <= 0 <= x2.max()),
                      "fit": fit, "coef_names": ["intercept", "beta1_x1_dp_full", "beta2_x2_dd"], "bootstrap": bt}
            for p in ids:
                pass
        res[f"E4.3|{e}"] = e43
        # ---- alternative controls
        alt = {}
        for cn in ("centre", "sobel", "mismatch"):
            sub = matched(prim_rows, (f"{cn}_ins", f"{cn}_del", "random_ins", "random_del"))
            ids2 = sorted(sub)
            vecs = {}
            for c in ("ins", "del"):
                for m in MODS:
                    vecs[f"{cn}|{c}|{m}"] = np.array([sub[p][m][f"{cn}_{c}"] for p in ids2])
                vecs[f"{cn}|{c}|D"] = vecs[f"{cn}|{c}|clinical"] - vecs[f"{cn}|{c}|dermoscopy"]
                if cn == "mismatch":
                    for m in MODS:
                        vecs[f"random_on_mismatch_subset|{c}|{m}"] = np.array([sub[p][m][f"random_{c}"] for p in ids2])
                    vecs[f"random_on_mismatch_subset|{c}|D"] = (vecs[f"random_on_mismatch_subset|{c}|clinical"] -
                                                                vecs[f"random_on_mismatch_subset|{c}|dermoscopy"])
            bs, meta = R4.bootstrap_stats(vecs, B, seed)
            seed += 1
            blk = {"n_pairs": len(ids2), "by_label": {lab: sum(sub[p]["dermoscopy"]["label_mel"] == lab for p in ids2)
                                                      for lab in ("1", "0")}, "bootstrap": meta, "stats": {}}
            for k, v in vecs.items():
                blk["stats"][k] = {"n": len(v), "median": R4.sample_median(v), "hl": R4.hodges_lehmann(v),
                                   "median_ci95": bs[k]["median"]["ci95"], "hl_ci95": bs[k]["hl"]["ci95"]}
                reps[f"alt|{e}|{k}"] = bs[k]["hl"]["replicates"]
            alt[cn] = blk
        res[f"alt|{e}"] = alt
        for r in rows:
            pair_rows.append({k: r[k] for k in ("ref_key", "pair_id", "modality", "label_mel", "status", "primary_pair",
                                                "p_full", "p_base", "d", "expl_ins", "expl_del", "rand_ins", "rand_del",
                                                "random_ins", "random_del", "centre_ins", "centre_del", "sobel_ins",
                                                "sobel_del", "mismatch_ins", "mismatch_del")} |
                             {f"cat_{dl}": r[f"cat_{dl}"] for dl in DELTAS} |
                             {f"nins_{dl}": r[f"nins_{dl}"] for dl in DELTAS} | {f"ndel_{dl}": r[f"ndel_{dl}"] for dl in DELTAS} |
                             {"explainer": e})
    res["check_norm_equals_raw_over_d_max_abs_diff"] = maxdiff
    if maxdiff > 1e-9:
        raise RuntimeError(f"normalized contrast != raw/d ({maxdiff})")
    publish_json(out, "results.json", res)
    publish_csv(out, "per_image_inputs.csv", pair_rows, list(pair_rows[0]))
    publish_csv(out, "normalization_categories.csv", cats_rows, list(cats_rows[0]))
    tab = []
    for e in EXPL:
        for delta in DELTAS:
            blk = res[f"norm|{e}|{delta}"]
            for c in ("ins", "del"):
                for m in ("dermoscopy", "clinical", "D"):
                    rs, nm = blk["stats"][f"raw_subset|{c}|{m}"], blk["stats"][f"norm|{c}|{m}"]
                    tab.append({"explainer": e, "delta": delta, "curve": c, "modality_or_D": m, "n_pairs": blk["n_pairs"],
                                "mel": blk["by_label"]["1"], "nonmel": blk["by_label"]["0"],
                                "raw_subset_median": rs["median"], "raw_subset_hl": rs["hl"], "raw_subset_hl_ci": json.dumps(rs["hl_ci95"]),
                                "norm_median": nm["median"], "norm_hl": nm["hl"], "norm_hl_ci": json.dumps(nm["hl_ci95"])})
    publish_csv(out, "normalized_vs_raw_table.csv", tab, list(tab[0]))
    alt_tab = []
    for e in EXPL:
        for cn, blk in res[f"alt|{e}"].items():
            for k, st in blk["stats"].items():
                alt_tab.append({"explainer": e, "block": cn, "stat": k, "n_pairs": blk["n_pairs"], "median": st["median"],
                                "median_ci": json.dumps(st["median_ci95"]), "hl": st["hl"], "hl_ci": json.dumps(st["hl_ci95"])})
    publish_csv(out, "alternative_controls_table.csv", alt_tab, list(alt_tab[0]))
    e43t = []
    for e in EXPL:
        for c, x in res[f"E4.3|{e}"].items():
            for i, nm in enumerate(x["coef_names"]):
                e43t.append({"explainer": e, "curve": c, "n": x["n"], "term": nm, "coef": x["fit"]["coef"][i],
                             "ci_lo": x["bootstrap"]["ci95"][i][0], "ci_hi": x["bootstrap"]["ci95"][i][1],
                             "r2": x["fit"]["r2"], "cond": x["fit"]["cond"], "rank": x["fit"]["rank"],
                             "undefined_reps": x["bootstrap"]["n_undefined_rank_deficient"]})
    publish_csv(out, "e43_table.csv", e43t, list(e43t[0]))
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k.replace("|", "__"): v for k, v in reps.items()})
    publish_bytes(out, "bootstrap_replicates.npz", buf.getvalue())
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out)) if not n.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    print("maxdiff", maxdiff, "\nverify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "analyze"))
    ap.add_argument("--out-root")
    ap.add_argument("--run-dir")
    a = ap.parse_args()
    from . import config as C
    cfg = C.load(a.config)
    if cfg["purpose"] != "analysis_v1":
        raise C.ConfigError("requires purpose 'analysis_v1'")
    (phase_spec if a.phase == "spec" else phase_analyze)(cfg, a)


if __name__ == "__main__":
    main()
