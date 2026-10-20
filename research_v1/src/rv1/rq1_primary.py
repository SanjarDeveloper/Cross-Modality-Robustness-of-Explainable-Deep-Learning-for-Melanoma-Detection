"""Stage 4B1: primary RQ1 paired analysis from saved Stage 4A predictions (CPU only).

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq1_primary --config ../configs/analysis_rq1_primary_v1.json --phase spec --out-root ../runs/analysis_rq1
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq1_primary --config ../configs/analysis_rq1_primary_v1.json --phase analyze --run-dir ../runs/analysis_rq1/<run_id>

The specification (with input hashes) is written and hashed BEFORE any statistic is computed.
"""
import argparse
import csv
import datetime
import json
import os

B = 2000
ALPHA = 0.05


def _spec(cfg):
    from . import config as C, provenance as P
    si = cfg["supporting_inputs"]
    inputs = {k: {"path": si[k], "sha256": P.sha256_file(C.resolve(cfg, si[k]))}
              for k in ("pred_derm7pt", "pred_ham_test", "threshold", "threshold_sidecar", "receipt_predict",
                        "verification", "model_checkpoint", "model_best_pointer")}
    with open(C.resolve(cfg, si["threshold"]), encoding="utf-8") as f:
        thr = json.load(f)
    return {
        "spec_version": "rq1_primary_v1", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": "02_RESEARCH_PROTOCOL_V1_DRAFT_rev2 §2 (bootstrap), §3 (RQ1 E1.1, E1.2, E1.4); Addendum G",
        "inputs": inputs,
        "model": {"checkpoint": si["model_checkpoint"], "sha256": thr["model"]["sha256"], "epoch": thr["model"]["best_epoch"]},
        "threshold": {"value": thr["threshold"], "repr": thr["threshold_repr"], "rule": thr["rule"],
                      "sha256": inputs["threshold"]["sha256"]},
        "cohorts": {"PRIMARY": "Derm7pt pairs with in_derm7pt_primary == '1' (expected 1002 pairs, 249 melanoma)",
                    "ALL": "all Derm7pt pairs (expected 1011 pairs, 252 melanoma); prespecified sensitivity analysis",
                    "pairing": "by pair_id; exactly one dermoscopy and one clinical record; never by row order"},
        "direction": "dermoscopy minus clinical",
        "primary_hypothesis": {"id": "E1.1", "test": "two-sided exact McNemar (exact binomial on discordant pairs, "
                                                     "p = min(1, 2*P[X<=min(b,c)]), X~Bin(b+c, 0.5)); no chi-square",
                               "population": "PRIMARY melanoma pairs", "alpha": ALPHA, "family_size": 1,
                               "b": "detected under dermoscopy only", "c": "detected under clinical only"},
        "estimation": {"id": "E1.2", "estimand": "paired sensitivity difference (dermoscopy - clinical), pp, "
                                                 "over PRIMARY melanoma pairs"},
        "secondary_descriptive": ["E1.4 specificity per modality and paired difference over PRIMARY negative "
                                  "(non-melanoma) pairs", "ALL-cohort repeat of E1.1/E1.2/specificity"],
        "bootstrap": {"type": "percentile, pair (case-cluster) bootstrap", "B": B, "level": 0.95,
                      "seed": cfg["seeds"]["bootstrap_seed"],
                      "rng": "numpy.random.Generator(numpy.random.PCG64(seed)); draws via Generator.integers(0, K, size=K) "
                             "over the sorted pair_id list; one independent generator per (cohort, estimand)",
                      "seed_derivation": "seed + offset: PRIMARY sens 0, PRIMARY spec 1, ALL sens 2, ALL spec 3",
                      "unit": "pair_id (both modalities move together; repeated draws keep multiplicity); "
                              "sensitivity resamples melanoma pairs only, specificity non-melanoma pairs only",
                      "quantiles": "numpy.quantile(method='linear') at 0.025 and 0.975",
                      "undefined": "counted and reported; excluded from quantiles; not redrawn; CI flagged unreliable "
                                   "if > 5% undefined"},
        "conditionality": "intervals are conditional on the fixed seed-0 model and frozen threshold; they exclude "
                          "training and threshold-selection uncertainty",
        "independence_unit": "case/pair; Derm7pt provides no patient identifier, so patient independence is not claimed"}


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


def phase_analyze(cfg, a):
    import numpy as np
    from . import config as C, provenance as P, rq1_stats as RS
    out = os.path.abspath(a.run_dir)
    sp = os.path.join(out, "analysis_spec.json")
    with open(sp + ".sha256", encoding="utf-8") as f:
        if f.read().split()[0] != P.sha256_file(sp):
            raise RuntimeError("spec hash mismatch")
    with open(sp, encoding="utf-8") as f:
        spec = json.load(f)
    for k, v in spec["inputs"].items():
        if P.sha256_file(C.resolve(cfg, v["path"])) != v["sha256"]:
            raise RuntimeError(f"input changed since spec: {k}")
    with open(C.resolve(cfg, cfg["supporting_inputs"]["threshold"]), encoding="utf-8") as f:
        thr = json.load(f)
    t = thr["threshold"]
    with open(C.resolve(cfg, cfg["supporting_inputs"]["pred_derm7pt"]), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    pairs = RS.build_pairs(rows, t, spec["threshold"]["sha256"], spec["model"]["sha256"])
    cohorts = {"PRIMARY": {k: v for k, v in pairs.items() if v["primary"]}, "ALL": pairs}
    checks = {"n_rows": len(rows),
              "PRIMARY": {"pairs": len(cohorts["PRIMARY"]), "mel_pairs": sum(v["label"] == 1 for v in cohorts["PRIMARY"].values())},
              "ALL": {"pairs": len(pairs), "mel_pairs": sum(v["label"] == 1 for v in pairs.values())},
              "same_file_pairs": sorted(k for k, v in pairs.items() if v["same_file"]),
              "same_file_pairs_not_primary": all(not pairs[k]["primary"] for k in pairs if pairs[k]["same_file"])}
    checks["expected_counts_ok"] = (checks["PRIMARY"] == {"pairs": 1002, "mel_pairs": 249} and
                                    checks["ALL"] == {"pairs": 1011, "mel_pairs": 252})
    if not checks["expected_counts_ok"]:
        raise RuntimeError(f"cohort counts differ: {checks}")
    seed0 = spec["bootstrap"]["seed"]
    res = {"checks": checks}
    boot_store = {}
    for ci_, (name, coh) in enumerate(cohorts.items()):
        plist = list(coh.values())
        mel = paired = RS.paired_table(plist, 1)
        neg = RS.paired_table(plist, 0)
        mc = RS.mcnemar_exact(mel["first_only"], mel["second_only"])
        sd, sc, dd = RS.sens_diff(plist, 1)
        pd_, pc_, pdd = RS.sens_diff(plist, 0)
        n_mel, n_neg = sum(mel.values()), sum(neg.values())
        mel_ids = sorted(k for k, v in coh.items() if v["label"] == 1)
        neg_ids = sorted(k for k, v in coh.items() if v["label"] == 0)
        bs = RS.pair_bootstrap(coh, mel_ids, lambda ps: RS.sens_diff(ps, 1), B, seed0 + 2 * ci_)
        bp = RS.pair_bootstrap(coh, neg_ids, lambda ps: RS.sens_diff(ps, 0), B, seed0 + 2 * ci_ + 1)
        boot_store[f"{name}_sens"] = bs.pop("replicates")
        boot_store[f"{name}_spec"] = bp.pop("replicates")
        res[name] = {
            "melanoma_paired_table": {"both_detected": mel["both"], "dermoscopy_only_b": mel["first_only"],
                                      "clinical_only_c": mel["second_only"], "both_missed": mel["neither"], "n": n_mel},
            "nonmelanoma_paired_table": {"both_correct_negative": neg["neither"], "dermoscopy_only_correct": neg["second_only"],
                                         "clinical_only_correct": neg["first_only"], "both_false_positive": neg["both"],
                                         "n": n_neg},
            "mcnemar_exact": {k: (str(v) if k == "p_exact" else v) for k, v in mc.items()},
            "sensitivity": {"dermoscopy": sd, "clinical": sc, "num_den_dermoscopy": [mel["both"] + mel["first_only"], n_mel],
                            "num_den_clinical": [mel["both"] + mel["second_only"], n_mel],
                            "diff_pp": 100 * dd,
                            "ci95_pp": {"dermoscopy": [100 * x for x in bs["ci"][0]], "clinical": [100 * x for x in bs["ci"][1]],
                                        "diff": [100 * x for x in bs["ci"][2]]}, "bootstrap": bs},
            "specificity": {"dermoscopy": pd_, "clinical": pc_,
                            "num_den_dermoscopy": [neg["neither"] + neg["second_only"], n_neg],
                            "num_den_clinical": [neg["neither"] + neg["first_only"], n_neg],
                            "diff_pp": 100 * pdd,
                            "ci95_pp": {"dermoscopy": [100 * x for x in bp["ci"][0]], "clinical": [100 * x for x in bp["ci"][1]],
                                        "diff": [100 * x for x in bp["ci"][2]]}, "bootstrap": bp}}
        res[name]["role"] = "PRIMARY (E1.1 confirmatory test, E1.2 estimate)" if name == "PRIMARY" else \
            "prespecified sensitivity analysis (not a second confirmatory result)"
    res["PRIMARY"]["decision"] = {"alpha": ALPHA, "reject_H0": res["PRIMARY"]["mcnemar_exact"]["p_value"] < ALPHA,
                                  "direction": "dermoscopy > clinical" if res["PRIMARY"]["melanoma_paired_table"]["dermoscopy_only_b"] >
                                  res["PRIMARY"]["melanoma_paired_table"]["clinical_only_c"] else "clinical >= dermoscopy"}
    res["spec_sha256"] = P.sha256_file(sp)
    with open(os.path.join(out, "results.json"), "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(os.path.join(out, "bootstrap_replicates.npz"),
                        **boot_store, columns=np.array(["dermoscopy", "clinical", "diff"]))
    with open(os.path.join(out, "paired_table.csv"), "x", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pair_id", "label_mel", "in_primary", "same_file_pair", "record_uid_dermoscopy", "record_uid_clinical",
                    "prob_dermoscopy", "prob_clinical", "pred_dermoscopy", "pred_clinical"])
        for k in sorted(pairs):
            v = pairs[k]
            w.writerow([k, v["label"], int(v["primary"]), int(v["same_file"]), v["uid_d"], v["uid_c"],
                        repr(v["prob_d"]), repr(v["prob_c"]), v["pred_d"], v["pred_c"]])
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze", "spec_sha256": res["spec_sha256"]})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("results.json", "bootstrap_replicates.npz",
                                                                           "paired_table.csv", "analysis_spec.json")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    print(json.dumps(res, indent=1, default=str)[:7000])
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
