"""Stage 4C: external baseline evaluation on PAD-UFES-20 and DDI (seed-0 ResNet-50, frozen HAM-VAL
threshold). Phases (separate processes): spec -> infer -> analyze.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.ext_eval --config ../configs/ext_eval_4c_v1.json --phase spec --out-root ../runs/ext_eval_4c
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.ext_eval --config ../configs/ext_eval_4c_v1.json --phase infer --run-dir <dir>
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.ext_eval --config ../configs/ext_eval_4c_v1.json --phase analyze --run-dir <dir>
"""
import argparse
import csv
import datetime
import json
import math
import os
import time

B = 2000


def check_join(manifest_rows, pred_rows):
    """Exact one-to-one coverage of manifest by predictions, with label agreement."""
    man = {r["record_uid"]: r for r in manifest_rows}
    seen = set()
    for r in pred_rows:
        u = r["record_uid"]
        if u in seen:
            raise RuntimeError(f"duplicate prediction {u}")
        if u not in man:
            raise RuntimeError(f"prediction not in manifest {u}")
        if man[u]["label_mel"] != r["label_mel"]:
            raise RuntimeError(f"label mismatch {u}")
        seen.add(u)
    if seen != set(man):
        raise RuntimeError(f"missing predictions: {len(set(man) - seen)}")
    return True


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _spec(cfg):
    from . import config as C, provenance as P
    si = cfg["supporting_inputs"]
    h = {k: {"path": v, "sha256": P.sha256_file(C.resolve(cfg, v))} for k, v in
         {"pad_manifest": cfg["manifests"]["pad_ufes_20"], "ddi_manifest": cfg["manifests"]["ddi"],
          "threshold": si["threshold"], "threshold_sidecar": si["threshold_sidecar"],
          "model_checkpoint": si["model_checkpoint"], "model_best_pointer": si["model_best_pointer"],
          "cache/pad_ufes_20_meta.json": si["cache/pad_ufes_20_meta.json"],
          "cache/ddi_meta.json": si["cache/ddi_meta.json"]}.items()}
    with open(C.resolve(cfg, si["threshold"]), encoding="utf-8") as f:
        thr = json.load(f)
    s = cfg["seeds"]["bootstrap_seed"]
    return {
        "spec_version": "ext_eval_4c_v1", "created_utc": _now(), "inputs": h,
        "model": {"sha256": thr["model"]["sha256"], "epoch": thr["model"]["best_epoch"]},
        "threshold": {"value": thr["threshold"], "repr": thr["threshold_repr"], "sha256": h["threshold"]["sha256"],
                      "rule": thr["rule"], "source": "Stage 4A HAM VAL; unchanged"},
        "inference": {"prep": "prep_v1_cached256 (same verified path as Stage 4A: rv1.predict_v1.infer)",
                      "precision": "xai_precision_v1", "batch": 16, "order": "manifest order within each cache",
                      "caches": ["pad_ufes_20 (2298)", "ddi (656)"]},
        "target": "melanoma (label_mel) vs non-melanoma; no malignant/benign substitution",
        "pad_roles": {"source": "02_rev2 §1.2 (cohort 'PAD all'), §3 E1.5 (PAD descriptive, patient-cluster bootstrap), "
                                "§7.1 frozen partition (rq5_role), §7.4 (PAD eval Fitzpatrick); no later addendum restricts PAD",
                      "reporting_cohorts": {"PAD_all": "E1.5 descriptive external baseline (all 2298 images, all roles; "
                                                       "role label kept in every row)",
                                            "PAD_eval": "frozen RQ5 evaluation cohort (baseline, unadapted model); "
                                                        "tone subgroups per §7.4"},
                      "adapt_train_adapt_val": "counts only; future adaptation roles; no metrics used for any choice",
                      "expected_eval": {"images": 1150, "patients": 687, "mel_images": 26, "mel_patients": 18}},
        "pad_bootstrap": {"unit": "patient (cluster_id); all images of a sampled patient; multiplicity kept; per-image "
                                  "labels retained (mixed-diagnosis patients not relabelled)",
                          "B": B, "seeds": {"PAD_all": s, "PAD_eval": s + 1},
                          "metrics": "image-level sensitivity, specificity, AUROC; same draws for all metrics"},
        "pad_tone": {"groups": "Fitzpatrick metadata 1..6 -> 'FST k'; empty -> 'missing' (not imputed)",
                     "rule": "counts for every group incl. missing; metrics only if >= 5 melanoma images; "
                             "subgroup metrics are point estimates with counts (no CI, no tests, no ranking)"},
        "ddi": {"expected": {"images": 656, "melanoma": 21},
                "tone_groups": {"12": "FST I-II", "34": "FST III-IV", "56": "FST V-VI"}, "missing": "none allowed",
                "sensitivity_ci": "Wilson score 95% (no continuity correction)",
                "auroc_ci": "image-level percentile bootstrap (DDI unit = image, 02_rev2 §1.2), B=2000",
                "seeds": {"overall": s + 2, "FST I-II": s + 3, "FST III-IV": s + 4, "FST V-VI": s + 5},
                "note": "no patient identifiers; patient independence not assumed or inferred"},
        "bootstrap_common": {"rng": "numpy Generator(PCG64(seed)); integers(0,K,K) over sorted unit IDs",
                             "quantiles": "numpy.quantile(method='linear') 0.025/0.975",
                             "undefined": "per metric counted; excluded; not redrawn; >5% -> unreliable"},
        "excluded": "no McNemar, no paired comparison with Derm7pt/HAM, no fairness/bias/causal conclusions, "
                    "no tone-group tests or ranking"}


def phase_spec(cfg, a):
    from . import provenance as P
    from .desc_4b3 import publish_bytes, publish_json
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "eval_spec.json", _spec(cfg))
    h = P.sha256_file(os.path.join(out, "eval_spec.json"))
    publish_bytes(out, "eval_spec.json.sha256", f"{h}  eval_spec.json\n".encode())
    receipt["outputs"] = {"eval_spec.json": h}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("spec", h, "\nrun dir", out)


def _load_spec(cfg, out):
    from . import config as C, provenance as P
    sp = os.path.join(out, "eval_spec.json")
    with open(sp + ".sha256", encoding="utf-8") as f:
        if f.read().split()[0] != P.sha256_file(sp):
            raise RuntimeError("spec hash mismatch")
    with open(sp, encoding="utf-8") as f:
        spec = json.load(f)
    for k, v in spec["inputs"].items():
        if P.sha256_file(C.resolve(cfg, v["path"])) != v["sha256"]:
            raise RuntimeError(f"input changed: {k}")
    return spec


def phase_infer(cfg, a):
    import numpy as np
    from . import config as C, decode as DC, provenance as P, threshold as TH
    from .desc_4b3 import publish_csv
    from .predict_v1 import infer, setup
    out = os.path.abspath(a.run_dir)
    spec = _load_spec(cfg, out)
    t = spec["threshold"]["value"]
    t0 = time.perf_counter()
    model, mc = setup(cfg)
    if mc["sha256"] != spec["model"]["sha256"]:
        raise RuntimeError("model mismatch")
    timing, batching = {"model_setup_s": time.perf_counter() - t0}, {}
    for ds, cache in (("pad_ufes_20", "pad_ufes_20"), ("ddi", "ddi")):
        rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"][ds]), ds)[0]
        uids = [r["record_uid"] for r in rows if r["cache_name"] == cache]
        if len(uids) != len(rows):
            raise RuntimeError(f"{ds}: rows outside cache")
        t1 = time.perf_counter()
        logits, probs, bi = infer(cfg, model, rows, cache, uids)
        timing[f"{ds}_inference_s"] = time.perf_counter() - t1
        batching[ds] = bi
        outrows = []
        for r, lg, p in zip(rows, logits, probs):
            if not (math.isfinite(lg) and math.isfinite(p) and 0.0 <= p <= 1.0):
                raise RuntimeError(f"invalid score {r['record_uid']}")
            outrows.append({"record_uid": r["record_uid"], "image_uid": r["image_uid"], "dataset": r["dataset"],
                            "cluster_id": r["cluster_id"],
                            "cluster_kind": "patient" if ds == "pad_ufes_20" else "image (no patient id)",
                            "rq5_role": r["rq5_role"], "stratum_rq5": r["stratum_rq5"], "label_mel": r["label_mel"],
                            "dx_raw": r["dx_raw"], "skin_tone_raw": r["skin_tone"], "modality": r["modality"],
                            "logit": repr(lg), "prob": repr(p), "threshold": repr(t), "pred": TH.predict(p, t),
                            "checkpoint_sha256": mc["sha256"], "threshold_sha256": spec["threshold"]["sha256"],
                            "prep_version": "prep_v1_cached256", "precision_policy": "xai_precision_v1",
                            "batch": 16, "status": "ok"})
        check_join(rows, outrows)
        publish_csv(out, f"predictions_{ds}.csv", outrows, list(outrows[0]))
    from .desc_4b3 import publish_json
    publish_json(out, "inference_log.json", {"timing_s": timing, "batching": batching, "model": mc,
                                              "inference_completed_utc": _now()})
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "infer"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in
                          ("predictions_pad_ufes_20.csv", "predictions_ddi.csv", "inference_log.json")}
    rp = P.write_receipt(receipt, out, name="receipt_infer.json")
    print(json.dumps(timing), "\nverify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def phase_analyze(cfg, a):
    import numpy as np
    from . import auroc_paired as AP, config as C, decode as DC, desc_stats as DS, ext_stats as ES, provenance as P
    from . import rq1_stats as RS, threshold as TH
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    t_start = time.perf_counter()
    out = os.path.abspath(a.run_dir)
    spec = _load_spec(cfg, out)
    with open(os.path.join(out, "receipt_infer.json"), encoding="utf-8") as f:
        ri = json.load(f)
    for n, h in ri["outputs"].items():
        if P.sha256_file(os.path.join(out, n)) != h:
            raise RuntimeError(f"prediction file changed: {n}")
    t = spec["threshold"]["value"]
    preds, mans = {}, {}
    for ds in ("pad_ufes_20", "ddi"):
        with open(os.path.join(out, f"predictions_{ds}.csv"), encoding="utf-8", newline="") as f:
            preds[ds] = list(csv.DictReader(f))
        mans[ds] = DC.read_manifest(C.resolve(cfg, cfg["manifests"][ds]), ds)[0]
        check_join(mans[ds], preds[ds])
        for r in preds[ds]:
            p = float(r["prob"])
            if not (0 <= p <= 1) or int(r["pred"]) != int(p >= t) or r["threshold_sha256"] != spec["threshold"]["sha256"] \
                    or r["checkpoint_sha256"] != spec["model"]["sha256"]:
                raise RuntimeError(f"row check failed {r['record_uid']}")
    seeds = spec["pad_bootstrap"]["seeds"]

    def point(rows):
        p = [float(r["prob"]) for r in rows]
        y = [int(r["label_mel"]) for r in rows]
        c = TH.counts(p, y, t)
        pos = [a_ for a_, b_ in zip(p, y) if b_ == 1]
        neg = [a_ for a_, b_ in zip(p, y) if b_ == 0]
        c["auroc"] = AP.auc(pos, neg) if pos and neg else None
        c["n"], c["n_mel"], c["n_nonmel"] = len(y), sum(y), len(y) - sum(y)
        return c

    def metric_fns():
        def rows_of(cl):
            return DS.flatten_clusters(cl)

        def sens(cl):
            f = rows_of(cl)
            pos = [r for r in f if r["label"] == 1]
            return (sum(r["prob"] >= t for r in pos) / len(pos),) if pos else None

        def spec_(cl):
            f = rows_of(cl)
            neg = [r for r in f if r["label"] == 0]
            return (sum(r["prob"] < t for r in neg) / len(neg),) if neg else None

        def auc(cl):
            f = rows_of(cl)
            pos = [r["prob"] for r in f if r["label"] == 1]
            neg = [r["prob"] for r in f if r["label"] == 0]
            return (AP.auc(pos, neg),) if pos and neg else None
        return {"sensitivity": sens, "specificity": spec_, "auroc": auc}

    def cluster_boot(rows, key, seed, metrics=("sensitivity", "specificity", "auroc")):
        cl = {}
        for r in rows:
            cl.setdefault(r[key], []).append({"label": int(r["label_mel"]), "prob": float(r["prob"])})
        ids = sorted(cl)
        fns = metric_fns()
        res, reps = {}, {}
        for m in metrics:
            b = RS.pair_bootstrap(cl, ids, fns[m], B, seed)
            reps[m] = b.pop("replicates")
            b["ci"] = b["ci"][0] if b["ci"] else None
            res[m] = b
        return res, reps, len(ids)
    results, reps_all = {"spec_sha256": P.sha256_file(os.path.join(out, "eval_spec.json"))}, {}
    # ---------------- PAD
    pad = preds["pad_ufes_20"]
    role_patients = ES.check_role_isolation(mans["pad_ufes_20"])
    roles = {}
    for role in ("ALL",) + ES.PAD_ROLES:
        rr = pad if role == "ALL" else [r for r in pad if r["rq5_role"] == role]
        roles[role] = {"images": len(rr), "patients": len({r["cluster_id"] for r in rr}),
                       "mel_images": sum(r["label_mel"] == "1" for r in rr),
                       "mel_patients": len({r["cluster_id"] for r in rr if r["label_mel"] == "1"}),
                       "mixed_label_patients": sum(len({x["label_mel"] for x in rr if x["cluster_id"] == pid}) > 1
                                                   for pid in {r["cluster_id"] for r in rr})}
    exp = spec["pad_roles"]["expected_eval"]
    ev = roles["eval"]
    results["pad_expected_eval_match"] = (ev["images"], ev["patients"], ev["mel_images"], ev["mel_patients"]) == \
        (exp["images"], exp["patients"], exp["mel_images"], exp["mel_patients"])
    if not results["pad_expected_eval_match"]:
        raise RuntimeError(f"PAD eval counts differ: {ev}")
    results["PAD_role_counts"] = roles
    results["PAD_role_patients_isolated"] = role_patients
    for name, rr, seed in (("PAD_all", pad, seeds["PAD_all"]), ("PAD_eval", [r for r in pad if r["rq5_role"] == "eval"],
                                                                seeds["PAD_eval"])):
        bs, reps, n_cl = cluster_boot(rr, "cluster_id", seed)
        reps_all.update({f"{name}_{k}": v for k, v in reps.items()})
        results[name] = {"role": spec["pad_roles"]["reporting_cohorts"][name], "n_patients": n_cl,
                         "point": point(rr), "ci95_patient_cluster": {k: v["ci"] for k, v in bs.items()},
                         "bootstrap": bs}
    # PAD eval tone
    tone = {}
    ev_rows = [r for r in pad if r["rq5_role"] == "eval"]
    for r in ev_rows:
        tone.setdefault(ES.pad_tone(r["skin_tone_raw"]), []).append(r)
    tone_tab = []
    for g in sorted(tone, key=lambda s: (s == "missing", s)):
        rr = tone[g]
        n_mel = sum(r["label_mel"] == "1" for r in rr)
        row = {"group": g, "images": len(rr), "patients": len({r["cluster_id"] for r in rr}), "mel_images": n_mel,
               "nonmel_images": len(rr) - n_mel, "metrics_reported": ES.subgroup_eligible(n_mel) and g != "missing"}
        if row["metrics_reported"]:
            pt = point(rr)
            row.update({k: pt[k] for k in ("TP", "FN", "TN", "FP", "sensitivity", "specificity", "auroc")})
        tone_tab.append(row)
    results["PAD_eval_tone"] = tone_tab
    # ---------------- DDI
    ddi = preds["ddi"]
    dds = spec["ddi"]["seeds"]
    if (len(ddi), sum(r["label_mel"] == "1" for r in ddi)) != (656, 21):
        raise RuntimeError("DDI counts")
    groups = {"overall": ddi}
    for r in ddi:
        groups.setdefault(ES.ddi_tone(r["skin_tone_raw"]), []).append(r)
    ddi_tab = []
    for g in ("overall", "FST I-II", "FST III-IV", "FST V-VI"):
        rr = groups[g]
        pt = point(rr)
        img_rows = [dict(r, img=r["record_uid"]) for r in rr]
        bs, reps, _ = cluster_boot(img_rows, "img", dds[g], metrics=("auroc",))
        reps_all[f"DDI_{g}_auroc"] = reps["auroc"]
        ddi_tab.append({"group": g, "images": pt["n"], "mel": pt["n_mel"], "nonmel": pt["n_nonmel"],
                        "TP": pt["TP"], "FN": pt["FN"], "TN": pt["TN"], "FP": pt["FP"],
                        "sensitivity": pt["sensitivity"], "sensitivity_wilson95": ES.wilson(pt["TP"], pt["TP"] + pt["FN"]),
                        "specificity": pt["specificity"], "auroc": pt["auroc"], "auroc_ci95_image_bootstrap": bs["auroc"]["ci"],
                        "auroc_boot_undefined": bs["auroc"]["n_undefined"], "auroc_boot_unreliable": bs["auroc"]["unreliable"]})
    results["DDI"] = ddi_tab
    results["analysis_runtime_s"] = time.perf_counter() - t_start
    publish_json(out, "results.json", results)
    publish_csv(out, "pad_role_counts.csv", [dict(v, role=k) for k, v in roles.items()],
                ["role", "images", "patients", "mel_images", "mel_patients", "mixed_label_patients"])
    publish_csv(out, "pad_eval_tone_table.csv", tone_tab,
                ["group", "images", "patients", "mel_images", "nonmel_images", "metrics_reported", "TP", "FN", "TN", "FP",
                 "sensitivity", "specificity", "auroc"])
    publish_csv(out, "ddi_table.csv", [dict(r, sensitivity_wilson95=json.dumps(r["sensitivity_wilson95"]),
                                            auroc_ci95_image_bootstrap=json.dumps(r["auroc_ci95_image_bootstrap"]))
                                       for r in ddi_tab], list(ddi_tab[0]))
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **reps_all)
    publish_bytes(out, "bootstrap_replicates.npz", buf.getvalue())
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in
                          ("results.json", "pad_role_counts.csv", "pad_eval_tone_table.csv", "ddi_table.csv",
                           "bootstrap_replicates.npz", "eval_spec.json")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    for k in ("PAD_all", "PAD_eval"):
        print(k, results[k]["n_patients"], results[k]["point"], results[k]["ci95_patient_cluster"],
              {m: v["n_undefined"] for m, v in results[k]["bootstrap"].items()})
    print(json.dumps(roles), "\n", json.dumps(tone_tab, default=str))
    for r in ddi_tab:
        print(r)
    print("runtime", results["analysis_runtime_s"], "\nverify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "infer", "analyze"))
    ap.add_argument("--out-root")
    ap.add_argument("--run-dir")
    a = ap.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from . import config as C
    from .seeds import seed_everything
    cfg = C.load(a.config)
    if cfg["purpose"] != "predict_v1":
        raise C.ConfigError("requires purpose 'predict_v1'")
    seed_everything(0, cfg["determinism"])
    {"spec": phase_spec, "infer": phase_infer, "analyze": phase_analyze}[a.phase](cfg, a)


if __name__ == "__main__":
    main()
