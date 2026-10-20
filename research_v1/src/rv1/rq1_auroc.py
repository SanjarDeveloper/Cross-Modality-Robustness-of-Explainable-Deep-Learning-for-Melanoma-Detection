"""Stage 4B2: RQ1 E1.3 secondary paired AUROC difference from saved Stage 4A predictions (CPU only).

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq1_auroc --config ../configs/analysis_rq1_auroc_v1.json --phase spec --out-root ../runs/analysis_rq1_auroc
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq1_auroc --config ../configs/analysis_rq1_auroc_v1.json --phase analyze --run-dir ../runs/analysis_rq1_auroc/<run_id>
"""
import argparse
import csv
import datetime
import json
import math
import os

B = 2000


def _spec(cfg):
    from . import config as C, provenance as P
    si = cfg["supporting_inputs"]
    inputs = {k: {"path": si[k], "sha256": P.sha256_file(C.resolve(cfg, si[k]))}
              for k in ("pred_derm7pt", "threshold", "threshold_sidecar", "receipt_predict", "verification",
                        "model_checkpoint", "model_best_pointer")}
    with open(C.resolve(cfg, si["threshold"]), encoding="utf-8") as f:
        thr = json.load(f)
    src = os.path.join(os.path.dirname(__file__), "auroc_paired.py")
    return {
        "spec_version": "rq1_auroc_e13_v1", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": "02_rev2 §3 E1.3 (secondary; DeLong unadjusted; case-bootstrap CI), §2 bootstrap rules",
        "inputs": inputs,
        "model": {"sha256": thr["model"]["sha256"], "epoch": thr["model"]["best_epoch"]},
        "threshold_identity": {"sha256": inputs["threshold"]["sha256"],
                               "note": "used only to validate prediction rows; AUROC is threshold-free"},
        "cohorts": {"PRIMARY": "in_derm7pt_primary == '1': 1002 pairs, 249 melanoma (analysis cohort)",
                    "ALL": "1011 pairs, 252 melanoma (sensitivity analysis only)",
                    "pairing": "by pair_id via rq1_stats.build_pairs (Stage 4B1 logic); no filtering by correctness, "
                               "XAI eligibility or pilot membership"},
        "score": "continuous melanoma probability (prob column), exact float",
        "direction": "AUROC_dermoscopy - AUROC_clinical (0-1 scale)",
        "auroc": "Mann-Whitney; tied positive/negative pairs count 1/2; cross-checked with scipy.stats.mannwhitneyu",
        "delong": {"method": "correlated/paired DeLong, DeLong ER, DeLong DM, Clarke-Pearson DL (1988) Biometrics "
                             "44:837-845; midrank computation per Sun X, Xu W (2014) IEEE SPL 21:1389-1393",
                   "implementation": "research_v1/src/rv1/auroc_paired.py (own code, no reused code/license)",
                   "implementation_sha256": P.sha256_file(src),
                   "test": "two-sided z = diff/SE; p = 2*Phi(-|z|); unadjusted; SECONDARY (not in the primary family)",
                   "zero_variance": "z/p reported undefined, not replaced"},
        "bootstrap": {"type": "percentile case/pair cluster bootstrap (ordinary, NOT class-stratified)", "B": B,
                      "level": 0.95, "seed": cfg["seeds"]["bootstrap_seed"], "seed_offsets": {"PRIMARY": 0, "ALL": 1},
                      "rng": "numpy.random.Generator(PCG64(seed)); Generator.integers(0, K, K) over sorted pair_id list",
                      "unit": "pair_id from all pairs of the cohort; both modalities together; multiplicity kept",
                      "quantiles": "numpy.quantile(method='linear'), 0.025/0.975",
                      "undefined": "one-class replicates counted, excluded from quantiles, not redrawn; >5% -> unreliable"},
        "conditionality": "conditional on the fixed seed-0 model; excludes training uncertainty",
        "independence_unit": "case/pair; no patient identifiers in Derm7pt"}


def phase_spec(cfg, a):
    from . import provenance as P
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    p = os.path.join(out, "analysis_spec.json")
    with open(p, "x", encoding="utf-8") as f:
        json.dump(_spec(cfg), f, indent=1)
    h = P.sha256_file(p)
    with open(p + ".sha256", "x", encoding="utf-8") as f:
        f.write(f"{h}  analysis_spec.json\n")
    receipt["outputs"] = {"analysis_spec.json": h}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("spec sha256", h, "\nrun dir", out)


def _auc_stat(ps):
    from . import auroc_paired as AP
    pd_ = [p["prob_d"] for p in ps if p["label"] == 1]
    nd = [p["prob_d"] for p in ps if p["label"] == 0]
    if not pd_ or not nd:
        return None
    pc = [p["prob_c"] for p in ps if p["label"] == 1]
    nc = [p["prob_c"] for p in ps if p["label"] == 0]
    ad, ac = AP.auc(pd_, nd), AP.auc(pc, nc)
    return ad, ac, ad - ac


def phase_analyze(cfg, a):
    import numpy as np
    from scipy.stats import mannwhitneyu
    from . import auroc_paired as AP, config as C, provenance as P, rq1_stats as RS
    out = os.path.abspath(a.run_dir)
    sp = os.path.join(out, "analysis_spec.json")
    with open(sp + ".sha256", encoding="utf-8") as f:
        if f.read().split()[0] != P.sha256_file(sp):
            raise RuntimeError("spec hash mismatch")
    with open(sp, encoding="utf-8") as f:
        spec = json.load(f)
    for k, v in spec["inputs"].items():
        if P.sha256_file(C.resolve(cfg, v["path"])) != v["sha256"]:
            raise RuntimeError(f"input changed: {k}")
    if P.sha256_file(os.path.join(os.path.dirname(__file__), "auroc_paired.py")) != spec["delong"]["implementation_sha256"]:
        raise RuntimeError("DeLong implementation changed since spec")
    with open(C.resolve(cfg, cfg["supporting_inputs"]["threshold"]), encoding="utf-8") as f:
        thr = json.load(f)
    with open(C.resolve(cfg, cfg["supporting_inputs"]["pred_derm7pt"]), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not all(0.0 <= float(r["prob"]) <= 1.0 for r in rows):
        raise RuntimeError("probability outside [0,1]")
    pairs = RS.build_pairs(rows, thr["threshold"], spec["threshold_identity"]["sha256"], spec["model"]["sha256"])
    cohorts = {"PRIMARY": {k: v for k, v in pairs.items() if v["primary"]}, "ALL": pairs}
    res = {"checks": {c: {"pairs": len(v), "mel_pairs": sum(p["label"] == 1 for p in v.values())} for c, v in cohorts.items()}}
    if res["checks"] != {"PRIMARY": {"pairs": 1002, "mel_pairs": 249}, "ALL": {"pairs": 1011, "mel_pairs": 252}}:
        raise RuntimeError(f"cohort counts differ {res['checks']}")
    reps = {}
    for off, (name, coh) in enumerate(cohorts.items()):
        ids = sorted(coh)
        ps = [coh[k] for k in ids]
        y = [p["label"] for p in ps]
        d = AP.delong_paired([p["prob_d"] for p in ps], [p["prob_c"] for p in ps], y)
        xc = {}
        for mod, key in (("dermoscopy", "prob_d"), ("clinical", "prob_c")):
            pos = [p[key] for p in ps if p["label"] == 1]
            neg = [p[key] for p in ps if p["label"] == 0]
            xc[mod] = mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))
        xc_ok = abs(xc["dermoscopy"] - d["auc_a"]) <= 1e-12 and abs(xc["clinical"] - d["auc_b"]) <= 1e-12
        bs = RS.pair_bootstrap(coh, ids, _auc_stat, B, spec["bootstrap"]["seed"] + off)
        reps[name] = bs.pop("replicates")
        res[name] = {"role": "E1.3 SECONDARY analysis cohort" if name == "PRIMARY" else "sensitivity analysis",
                     "n_pairs": len(ps), "n_mel": d["m"], "n_nonmel": d["n"],
                     "auroc_dermoscopy": d["auc_a"], "auroc_clinical": d["auc_b"], "diff_auroc": d["diff"],
                     "delong": d, "crosscheck_scipy_mwu": dict(xc, agree_1e12=xc_ok),
                     "bootstrap": bs,
                     "ci95": {"dermoscopy": bs["ci"][0], "clinical": bs["ci"][1], "diff": bs["ci"][2]},
                     "units": "AUROC on 0-1 scale; difference in AUROC units (not percentage points)"}
        if not xc_ok:
            raise RuntimeError("AUROC cross-check failed")
    res["spec_sha256"] = P.sha256_file(sp)
    with open(os.path.join(out, "results.json"), "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item())
    np.savez_compressed(os.path.join(out, "bootstrap_replicates.npz"), **reps,
                        columns=np.array(["auroc_dermoscopy", "auroc_clinical", "diff"]))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze", "spec_sha256": res["spec_sha256"]})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("results.json", "bootstrap_replicates.npz",
                                                                           "analysis_spec.json")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    for name in cohorts:
        r = res[name]
        print(name, {k: r[k] for k in ("n_pairs", "auroc_dermoscopy", "auroc_clinical", "diff_auroc", "ci95")},
              {k: r["delong"][k] for k in ("se_diff", "cov_ab", "corr_ab", "z", "p_value_sci", "log10_p")},
              "undefined", r["bootstrap"]["n_undefined"], r["crosscheck_scipy_mwu"])
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


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
