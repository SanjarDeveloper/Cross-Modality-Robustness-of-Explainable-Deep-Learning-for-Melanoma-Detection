"""Stage 7B: primary RQ4 statistics (E4.1 / E4.2, 12-test Holm family) and the separate as-computed
sensitivity analysis, from the verified Stage 7A v2 artifacts only (CPU).

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq4_analysis --config ../configs/rq4_stats_7b_v1.json --phase spec --out-root ../runs/rq4_stats_7b
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq4_analysis --config ../configs/rq4_stats_7b_v1.json --phase analyze --run-dir <dir>
"""
import argparse
import csv
import datetime
import json
import math
import os

B = 2000
EXPL = ("gradcam", "ig512")
MODS = ("dermoscopy", "clinical")
CURVES = ("ins", "del")
EXPECTED_PAIRS = {"gradcam": 275, "ig512": 373}
EXPECTED_MISMATCH = {"gradcam": 146, "ig512": 280}
VALID_FILES = ("scores_v2.csv", "image_status_v2.csv", "coverage_audit_v2.json", "spec.json", "mismatch_assignment.json",
               "amendment_mismatch_v2.json")


def _f(v):
    return None if v in (None, "") else float(v)


def matched_pairs(rows, primary):
    """rows: per-image dicts (ref_key, pair_id, modality, status, delta_ins/delta_del, as_ins/as_del).
    primary=True: pairs whose map is primary in BOTH modalities with finite primary deltas.
    primary=False: as-computed population: finite as-computed deltas in both modalities.
    Duplicate keys or incomplete pairs raise."""
    keys = [r["ref_key"] for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate ref_key")
    by = {}
    for r in rows:
        if r["modality"] in by.setdefault(r["pair_id"], {}):
            raise ValueError(f"duplicate modality in pair {r['pair_id']}")
        by[r["pair_id"]][r["modality"]] = r
    out = {}
    for pid, d in by.items():
        if set(d) != set(MODS):
            raise ValueError(f"incomplete pair {pid}")
        vals = {}
        for m in MODS:
            r = d[m]
            if primary:
                ok = r["status"] == "primary" and _f(r["delta_ins"]) is not None and _f(r["delta_del"]) is not None
                v = (_f(r["delta_ins"]), _f(r["delta_del"]))
            else:
                v = (_f(r["as_ins"]), _f(r["as_del"]))
                ok = all(x is not None and math.isfinite(x) for x in v)
            if not ok:
                break
            vals[m] = {"ins": v[0], "del": v[1], "label_mel": r["label_mel"], "status": r["status"]}
        else:
            out[pid] = vals
    return out


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _run(cfg):
    from . import config as C
    return C.resolve(cfg, cfg["supporting_inputs"]["stage7a_audit_receipt"]).rsplit(os.sep, 1)[0]


def phase_spec(cfg, a):
    import scipy
    from . import config as C, provenance as P
    from .desc_4b3 import publish_bytes, publish_json
    run = _run(cfg)
    rec = json.load(open(C.resolve(cfg, cfg["supporting_inputs"]["stage7a_audit_receipt"]), encoding="utf-8"))
    ins = {n: {"path": os.path.relpath(os.path.join(run, n), cfg["project_root"]).replace("\\", "/"),
               "sha256": P.sha256_file(os.path.join(run, n))} for n in VALID_FILES}
    for n, h in rec["outputs"].items():
        if P.sha256_file(os.path.join(run, n)) != h:
            raise RuntimeError(f"Stage 7A output changed: {n}")
    s = cfg["seeds"]["bootstrap_seed"]
    spec = {"spec_version": "rq4_stats_7b_v1", "created_utc": _now(), "stage7a_run": os.path.basename(run),
            "inputs": ins, "stage7a_audit_receipt_outputs": rec["outputs"],
            "excluded_artifacts": "invalid_v1_donor_bug/* (never read; analysis reads only the v2 files listed, chunk_*/ and "
                                  "mismatch2_*/ records verified against the audit receipt)",
            "model": "Stage 3A seed-0 ResNet-50 (303d21e7...) only",
            "ig_note": "IG m=512 fixed; 138/996 completeness failures (13.9%); pilot adequacy did not transfer; not globally "
                       "converged. No production-wide stop rule exists in protocol/addenda (02_rev2 §6.6 requires the failure "
                       "rate to be reported; the convergence criterion is a pilot rule)",
            "populations": {"primary": "matched eligible pairs per explainer: map primary-eligible in both modalities AND "
                                       "random control available (5 finite AUCs) in both; expected Grad-CAM 275, IG 373",
                            "as_computed_sensitivity": "non-primary, separate: valid maps use their primary curves; finite "
                                                       "constant / completeness-failed maps use stored as-computed curves "
                                                       "(original deterministic pixel ordering); non-finite maps or failed "
                                                       "curves unavailable; completeness-failed IG maps remain numerically "
                                                       "unvalidated",
                            "other_controls": "centre, Sobel, mismatch: descriptive only, within primary pairs where the "
                                              "control is available in both modalities (mismatch: recipient and planned donor "
                                              "valid); expected mismatch pairs Grad-CAM 146, IG 280"},
            "contrasts": {"insertion": "AUC_ins(explainer) - mean_k AUC_ins(random_k)", "deletion":
                          "mean_k AUC_del(random_k) - AUC_del(explainer)", "random": "5 seeds averaged within image; not observations",
                          "E4.2": "D = contrast_clinical - contrast_dermoscopy per pair"},
            "tests": {"family": "12 = E4.1 (2 explainers x 2 curves x 2 modalities) + E4.2 (2 x 2); Holm at alpha 0.05",
                      "wilcoxon": {"scipy_version": scipy.__version__, "zero_method": "wilcox (zeros dropped)",
                                   "method": "approx (normal) with scipy tie-corrected variance", "correction": False,
                                   "alternative": "two-sided", "label": "not exact"},
                      "untestable": "all-zero input -> p unavailable, carried as p=1 in Holm (family size stays 12)",
                      "gate": "explainer x curve passes iff E4.1 dermoscopy Holm-adjusted p < 0.05 AND HL > 0; otherwise its "
                              "E4.2 is reported but interpreted descriptively; it is never removed from the family",
                      "assumption": "signed-rank test assumes a distribution symmetric about its centre; HL estimates the "
                                    "pseudomedian; the sample median is reported separately"},
            "estimates": {"sample_median": "ordinary median", "hodges_lehmann": "median of Walsh averages (i<=j), one-sample",
                          "bootstrap": {"B": B, "seeds": {"primary_gradcam": s, "primary_ig512": s + 1,
                                                          "ascomp_gradcam": s + 2, "ascomp_ig512": s + 3},
                                        "unit": "pair (both modalities together; multiplicity kept)",
                                        "quantiles": "numpy linear, 0.025/0.975",
                                        "undefined": "counted; >5% flagged unreliable"}},
            "claim_limits": "class-enriched cohort; exclusions select on model evidence; IG numerical failures; seed 0 only; "
                            "24 pilot pairs previously inspected; metrics describe behaviour under the pixel-replacement "
                            "procedure, not clinical correctness or causal explanation"}
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "analysis_spec.json", spec)
    h = P.sha256_file(os.path.join(out, "analysis_spec.json"))
    publish_bytes(out, "analysis_spec.json.sha256", f"{h}  analysis_spec.json\n".encode())
    receipt["outputs"] = {"analysis_spec.json": h}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out)


def _image_rows(run, ssha_chunks):
    """Per explainer, per image rows combining scores_v2 (primary deltas, random) and as-computed deltas
    from chunk records."""
    from .xai_prod import _read_csv_rows, _read_json
    status = {(r["ref_key"], r["explainer"]): r for r in _read_csv_rows(os.path.join(run, "image_status_v2.csv"))}
    scores = _read_csv_rows(os.path.join(run, "scores_v2.csv"))
    sc = {(r["ref_key"], r["explainer"], r["control"]): r for r in scores}
    if len(sc) != len(scores):
        raise ValueError("duplicate score rows")
    recs = {}
    for d in sorted(os.listdir(run)):
        if d.startswith("chunk_") and os.path.isdir(os.path.join(run, d)):
            for r in _read_json(os.path.join(run, d, "records.json")):
                if r["ref_key"] in recs:
                    raise ValueError("duplicate record")
                recs[r["ref_key"]] = r
    rows = {e: [] for e in EXPL}
    checks = {"primary_vs_records_max_abs_diff": 0.0}
    for k, r in recs.items():
        rnd = r["controls"]["random"]
        ok_r = rnd["status"] == "available" and len(rnd["auc_ins_each"]) == 5 and \
            all(v is not None and math.isfinite(v) for v in rnd["auc_ins_each"] + rnd["auc_del_each"])
        mi = sum(rnd["auc_ins_each"]) / 5 if ok_r else None
        md = sum(rnd["auc_del_each"]) / 5 if ok_r else None
        for e in EXPL:
            ex = r["explainers"][e]
            cv = ex.get("primary_curve") if ex["status"] == "primary" else ex.get("as_computed_curve")
            good = cv is not None and cv.get("curve_status") == "ok" and ok_r
            as_ins = cv["auc_insertion_v1"] - mi if good else None
            as_del = md - cv["auc_deletion_v1"] if good else None
            s = sc[(k, e, "random")]
            st = status[(k, e)]
            prim = st["status"] == "primary" and s["control_status"] == "available"
            if prim:
                checks["primary_vs_records_max_abs_diff"] = max(checks["primary_vs_records_max_abs_diff"],
                                                                abs(float(s["delta_ins"]) - as_ins), abs(float(s["delta_del"]) - as_del))
            rows[e].append({"ref_key": k, "pair_id": r["pair_id"], "modality": r["modality"], "label_mel": r["label_mel"],
                            "status": "primary" if prim else st["status"], "map_flags": st["map_flags"], "reasons": st["reasons"],
                            "delta_ins": s["delta_ins"] if prim else "", "delta_del": s["delta_del"] if prim else "",
                            "as_ins": "" if as_ins is None else repr(as_ins), "as_del": "" if as_del is None else repr(as_del),
                            "as_source": "primary_curve" if ex["status"] == "primary" else ("as_computed_curve" if cv else "none"),
                            "pilot_member": r["stage3b_pilot_member"]})
    return rows, sc, checks


def _summ(x):
    from . import rq4_stats as R4
    return {"n": len(x), "median": R4.sample_median(x), "hl": R4.hodges_lehmann(x)}


def phase_analyze(cfg, a):
    import io
    import numpy as np
    from . import config as C, provenance as P, rq4_stats as R4
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    out = os.path.abspath(a.run_dir)
    sp = os.path.join(out, "analysis_spec.json")
    if open(sp + ".sha256", encoding="utf-8").read().split()[0] != P.sha256_file(sp):
        raise RuntimeError("spec hash mismatch")
    spec = json.load(open(sp, encoding="utf-8"))
    run = _run(cfg)
    for n, v in spec["inputs"].items():
        if P.sha256_file(os.path.join(run, n)) != v["sha256"]:
            raise RuntimeError(f"input changed {n}")
    for n, h in spec["stage7a_audit_receipt_outputs"].items():
        if P.sha256_file(os.path.join(run, n)) != h:
            raise RuntimeError(f"Stage 7A output changed {n}")
    rows, sc, checks = _image_rows(run, None)
    if checks["primary_vs_records_max_abs_diff"] > 1e-12:
        raise RuntimeError(f"scores_v2 vs records disagreement {checks}")
    seeds = spec["estimates"]["bootstrap"]["seeds"]
    res = {"checks": checks, "spec_sha256": P.sha256_file(sp)}
    tests, reps = [], {}
    pair_tables = {}
    for e in EXPL:
        for pop in ("primary", "ascomp"):
            mp = matched_pairs(rows[e], primary=(pop == "primary"))
            ids = sorted(mp)
            vecs = {}
            for c in CURVES:
                for m in MODS:
                    vecs[f"E4.1|{c}|{m}"] = np.array([mp[p][m][c] for p in ids])
                vecs[f"E4.2|{c}|D"] = np.array([mp[p]["clinical"][c] - mp[p]["dermoscopy"][c] for p in ids])
            bs, meta = R4.bootstrap_stats(vecs, B, seeds[f"{pop}_{e}"])
            reps.update({f"{pop}|{e}|{k}|{s}": bs[k][s]["replicates"] for k in bs for s in ("median", "hl")})
            block = {"n_pairs": len(ids), "by_label": {lab: sum(mp[p]["dermoscopy"]["label_mel"] == lab for p in ids) for lab in ("1", "0")},
                     "bootstrap": meta, "stats": {}}
            for k, v in vecs.items():
                st = dict(_summ(v), median_ci95=bs[k]["median"]["ci95"], hl_ci95=bs[k]["hl"]["ci95"])
                if pop == "primary":
                    st["wilcoxon"] = R4.wilcoxon_test(v)
                block["stats"][k] = st
            res[f"{pop}|{e}"] = block
            pair_tables[(pop, e)] = (ids, mp)
        if res[f"primary|{e}"]["n_pairs"] != EXPECTED_PAIRS[e]:
            raise RuntimeError(f"{e}: primary pairs {res[f'primary|{e}']['n_pairs']} != expected {EXPECTED_PAIRS[e]}")
    fam = []
    for e in EXPL:
        for c in CURVES:
            for m in MODS:
                fam.append((e, f"E4.1|{c}|{m}"))
        for c in CURVES:
            fam.append((e, f"E4.2|{c}|D"))
    pv = [res[f"primary|{e}"]["stats"][k]["wilcoxon"]["p_value"] for e, k in fam]
    adj = R4.holm(pv)
    if len(fam) != 12:
        raise RuntimeError("family size")
    gates = {}
    for e in EXPL:
        for c in CURVES:
            k = f"E4.1|{c}|dermoscopy"
            i = fam.index((e, k))
            gates[(e, c)] = R4.gate(adj[i], res[f"primary|{e}"]["stats"][k]["hl"])
    table = []
    for (e, k), p, pa in zip(fam, pv, adj):
        st = res[f"primary|{e}"]["stats"][k]
        est, c, m = k.split("|")
        asst = res[f"ascomp|{e}"]["stats"][k]
        table.append({"test": f"{est} {e} {c} {m}", "explainer": e, "estimand": est, "curve": c,
                      "modality_or_contrast": m, "n_pairs": st["n"], "n_nonzero": st["wilcoxon"]["n_nonzero"],
                      "median": st["median"], "median_ci_lo": st["median_ci95"][0], "median_ci_hi": st["median_ci95"][1],
                      "hl": st["hl"], "hl_ci_lo": st["hl_ci95"][0], "hl_ci_hi": st["hl_ci95"][1],
                      "wilcoxon_W": st["wilcoxon"]["statistic"], "p_raw": p, "p_holm": pa, "reject_holm_0.05": pa < 0.05,
                      "gate_passed_for_explainer_curve": gates[(e, c)],
                      "interpretation": ("confirmatory" if est == "E4.1" or gates[(e, c)] else "descriptive (gate not passed)"),
                      "ascomp_n_pairs": asst["n"], "ascomp_median": asst["median"], "ascomp_hl": asst["hl"],
                      "ascomp_hl_ci_lo": asst["hl_ci95"][0], "ascomp_hl_ci_hi": asst["hl_ci95"][1],
                      "ascomp_minus_primary_hl": asst["hl"] - st["hl"]})
    res["family"] = table
    res["gates"] = {f"{e}|{c}": v for (e, c), v in gates.items()}
    # eligibility tables
    elig = []
    for e in EXPL:
        for m in MODS:
            for lab in ("1", "0"):
                rr = [r for r in rows[e] if r["modality"] == m and r["label_mel"] == lab]
                reasons = {}
                for r in rr:
                    if r["status"] != "primary":
                        reasons[r["reasons"] or r["status"]] = reasons.get(r["reasons"] or r["status"], 0) + 1
                elig.append({"explainer": e, "modality": m, "label_mel": lab, "n": len(rr),
                             "primary": sum(r["status"] == "primary" for r in rr), "excluded": len(rr) - sum(r["status"] == "primary" for r in rr),
                             "reasons": json.dumps(reasons), "as_computed_available": sum(r["as_ins"] != "" for r in rr)})
    # other controls (descriptive)
    other = []
    for e in EXPL:
        ids, mp = pair_tables[("primary", e)]
        for cn in ("centre", "sobel", "mismatch"):
            ok = [p for p in ids if all(sc[(f"{_uid(rows[e], p, m)}", e, cn)]["control_status"] == "available" for m in MODS)]
            row = {"explainer": e, "control": cn, "n_pairs": len(ok)}
            for c in CURVES:
                key = "delta_ins" if c == "ins" else "delta_del"
                for m in MODS:
                    v = [float(sc[(_uid(rows[e], p, m), e, cn)][key]) for p in ok]
                    s = _summ(v)
                    row[f"{c}_{m}_median"], row[f"{c}_{m}_hl"] = s["median"], s["hl"]
                D = [float(sc[(_uid(rows[e], p, "clinical"), e, cn)][key]) - float(sc[(_uid(rows[e], p, "dermoscopy"), e, cn)][key]) for p in ok]
                s = _summ(D)
                row[f"{c}_D_median"], row[f"{c}_D_hl"] = s["median"], s["hl"]
            other.append(row)
        mm = next(r for r in other if r["explainer"] == e and r["control"] == "mismatch")["n_pairs"]
        if mm != EXPECTED_MISMATCH[e]:
            raise RuntimeError(f"{e}: mismatch pairs {mm} != expected {EXPECTED_MISMATCH[e]}")
    res["other_controls_descriptive"] = other
    res["eligibility"] = elig
    publish_json(out, "results.json", res)
    publish_csv(out, "rq4_family_table.csv", table, list(table[0]))
    publish_csv(out, "eligibility_table.csv", elig, list(elig[0]))
    publish_csv(out, "other_controls_descriptive.csv", other, list(other[0]))
    flat = [dict(r, explainer=e) for e in EXPL for r in rows[e]]
    publish_csv(out, "per_image_inputs.csv", flat, list(flat[0]))
    pp = []
    for (pop, e), (ids, mp) in pair_tables.items():
        for p in ids:
            pp.append({"population": pop, "explainer": e, "pair_id": p, "label_mel": mp[p]["dermoscopy"]["label_mel"],
                       "ins_dermoscopy": mp[p]["dermoscopy"]["ins"], "ins_clinical": mp[p]["clinical"]["ins"],
                       "del_dermoscopy": mp[p]["dermoscopy"]["del"], "del_clinical": mp[p]["clinical"]["del"],
                       "status_dermoscopy": mp[p]["dermoscopy"]["status"], "status_clinical": mp[p]["clinical"]["status"]})
    publish_csv(out, "per_pair_inputs.csv", pp, list(pp[0]))
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k.replace("|", "__"): v for k, v in reps.items()})
    publish_bytes(out, "bootstrap_replicates.npz", buf.getvalue())
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))
                          if not n.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    for r in table:
        print({k: (round(v, 5) if isinstance(v, float) and not k.startswith("p_") else v) for k, v in r.items()
               if k in ("test", "n_pairs", "median", "hl", "hl_ci_lo", "hl_ci_hi", "p_raw", "p_holm", "interpretation",
                        "ascomp_n_pairs", "ascomp_hl")})
    print(res["gates"], checks, "\nverify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def _uid(rows, pair_id, mod):
    for r in rows:
        if r["pair_id"] == pair_id and r["modality"] == mod:
            return r["ref_key"]
    raise KeyError(pair_id)


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
