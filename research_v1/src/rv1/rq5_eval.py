"""Stage 5B: RQ5 base-vs-adapted evaluation (02_rev2 §7.3; Addendum H). Separate processes:
  spec -> threshold (adapted model on HAM VAL only) -> infer (adapted: PAD eval, HAM TEST, Derm7pt clinical)
  -> analyze (paired cluster bootstrap; both models on the same draws).

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq5_eval --config ../configs/rq5_eval_5b_v1.json --phase spec --out-root ../runs/rq5_eval_5b
  ... --phase threshold|infer|analyze --run-dir <dir>
"""
import argparse
import csv
import datetime
import glob
import json
import math
import os
import time

ADAPT_SHA = "a18bf969c241a96f63849e2559a9f73fba0985aa192075d3322dc49f175e1286"
BASE_SHA = "303d21e71986dc8f26dec84c6822a301f696060e5f08f613c90fc33c41a003dd"
B = 2000
BATCH = 16


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_selected(best, sha):
    if not (best["checkpoint"] == "ckpt_epoch4.pt" and best["epoch"] == 4 and best["sha256"] == sha == ADAPT_SHA):
        raise RuntimeError(f"selected adapted checkpoint identity mismatch: {best}")


def _read(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _paths(cfg):
    from . import config as C
    si = cfg["supporting_inputs"]
    return {k: C.resolve(cfg, v) for k, v in si.items()}


def _spec(cfg):
    from . import config as C, provenance as P
    si = cfg["supporting_inputs"]
    ins = {k: {"path": v, "sha256": P.sha256_file(C.resolve(cfg, v))} for k, v in si.items()
           if not k.startswith("cache/") or k.endswith("meta.json")}
    for m in ("ham10000", "derm7pt", "pad_ufes_20"):
        ins[f"manifest_{m}"] = {"path": cfg["manifests"][m], "sha256": P.sha256_file(C.resolve(cfg, cfg["manifests"][m]))}
    with open(C.resolve(cfg, si["base_threshold"]), encoding="utf-8") as f:
        bt = json.load(f)
    s = cfg["seeds"]["bootstrap_seed"]
    return {
        "spec_version": "rq5_eval_5b_v1", "created_utc": _now(), "inputs": ins,
        "models": {"base": {"sha256": BASE_SHA, "role": "Stage 3A primary model (epoch 8)"},
                   "adapted": {"sha256": ADAPT_SHA, "role": "Stage 5A PAD-adapted, selected epoch 4 (NOT latest epoch 5)"}},
        "thresholds": {"base": {"value": bt["threshold"], "sha256": ins["base_threshold"]["sha256"],
                                "source": "Stage 4A HAM VAL, unchanged"},
                       "adapted_secondary_A": {"rule": "v1_highest_t_img_sens95 (rv1.threshold.estimate) on full HAM VAL "
                                                       "(1529, configs/main_ham_val_full_v1.csv) with the adapted model; "
                                                       "saved+hashed in a separate phase before any other adapted inference",
                                               "val_list_sha256": ins["val_list"]["sha256"]},
                       "adapted_secondary_B": "NOT_ESTIMABLE by rule: PAD adapt_val has 9 melanoma images (< 20); not fitted",
                       "note": "model-specific thresholds; never interchanged"},
        "inference": {"prep": "prep_v1_cached256", "precision": "xai_precision_v1", "batch": BATCH,
                      "order": "manifest order within each cache (HAM VAL: val-list order)",
                      "adapted_cohorts": ["HAM VAL (threshold only)", "PAD eval (1150)", "HAM TEST (1548)",
                                          "Derm7pt clinical (1011; PRIMARY subset by flag)"]},
        "base_predictions": {"PAD eval": "Stage 4C predictions_pad_ufes_20.csv filtered rq5_role=eval",
                             "HAM TEST": "Stage 4A predictions_ham_test.csv",
                             "Derm7pt clinical": "Stage 4A predictions_derm7pt_all.csv modality=clinical"},
        "estimands": {"primary_RQ5_descriptive": "AUROC(adapted) - AUROC(base) on PAD eval (1150 images / 687 patients)",
                      "secondary_A": "sensitivity and specificity (adapted - base), each model at its own HAM-VAL threshold",
                      "forgetting": "AUROC(adapted) - AUROC(base) on HAM TEST and on Derm7pt PRIMARY clinical; "
                                    "Derm7pt ALL clinical as sensitivity analysis",
                      "direction": "adapted minus base"},
        "bootstrap": {"type": "paired percentile cluster bootstrap; both models scored on the SAME draws", "B": B,
                      "units": {"PAD eval": "patient (cluster_id)", "HAM TEST": "lesion (cluster_id)",
                                "Derm7pt": "case (pair_id; one clinical image per case)"},
                      "seeds": {"PAD": s, "HAM": s + 1, "Derm7pt_PRIMARY": s + 2, "Derm7pt_ALL": s + 3},
                      "metric_families": "AUROC / sensitivity / specificity bootstrapped separately with the SAME seed "
                                         "(identical draws), so undefined replicates are counted per metric",
                      "rng": "numpy Generator(PCG64); integers(0,K,K) over sorted cluster IDs",
                      "quantiles": "numpy.quantile(method='linear') 0.025/0.975",
                      "undefined": "counted per metric; excluded; never redrawn; >5% -> unreliable"},
        "no_tuning": "no checkpoint, threshold, recipe or analysis change after evaluation",
        "disclosure": "PAD eval baseline (base model) results were inspected in Stage 4C; the adaptation recipe "
                      "(Addendum H) was not chosen using them"}


def _clarification(cfg):
    from . import config as C, provenance as P
    p = _paths(cfg)
    with open(p["adapt_best"], encoding="utf-8") as f:
        best = json.load(f)
    check_selected(best, P.sha256_file(p["adapt_ckpt"]))
    with open(p["adapt_val_epoch4"], encoding="utf-8") as f:
        v4 = json.load(f)
    val = _read(p["adapt_val_list"])
    if v4["record_uids"] != [r["record_uid"] for r in val]:
        raise RuntimeError("epoch-4 validation records differ from the frozen adapt_val list")
    with open(p["adapt_selection_log"], encoding="utf-8") as f:
        dec = [json.loads(x) for x in f if x.strip()]
    return {"clarifies": "best.json free-text 'rule' says 'HAM VAL'; this record states the actual selection data",
            "actual_selection_dataset": "PAD-UFES-20 adapt_val (rq5_role=adapt_val)",
            "validation_list": cfg["supporting_inputs"]["adapt_val_list"],
            "validation_list_sha256": P.sha256_file(p["adapt_val_list"]),
            "n_val": len(val), "n_val_mel": sum(r["label_mel"] == "1" for r in val),
            "selected_epoch": best["epoch"], "checkpoint": best["checkpoint"], "checkpoint_sha256": best["sha256"],
            "metric": "image-level AUROC on adapt_val", "tie_rule": "earlier epoch wins ties",
            "selected_val_auroc": best["val_auroc"], "per_epoch": [{k: d[k] for k in ("epoch", "val_auroc", "improved")}
                                                                   for d in dec],
            "evidence": {"config": cfg["supporting_inputs"]["adapt_config"],
                         "val_epoch4": cfg["supporting_inputs"]["adapt_val_epoch4"],
                         "val_epoch4_sha256": P.sha256_file(p["adapt_val_epoch4"]),
                         "selection_decisions": cfg["supporting_inputs"]["adapt_selection_log"],
                         "best_json_sha256": P.sha256_file(p["adapt_best"])}}


def _publish_hashed(out, name, obj):
    from . import provenance as P
    from .desc_4b3 import publish_bytes, publish_json
    publish_json(out, name, obj)
    h = P.sha256_file(os.path.join(out, name))
    publish_bytes(out, name + ".sha256", f"{h}  {name}\n".encode())
    return h


def _check_hashed(out, name):
    from . import provenance as P
    p = os.path.join(out, name)
    with open(p + ".sha256", encoding="utf-8") as f:
        if f.read().split()[0] != P.sha256_file(p):
            raise RuntimeError(f"{name} hash mismatch")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _load_spec(cfg, out):
    from . import config as C, provenance as P
    spec = _check_hashed(out, "eval_spec.json")
    for k, v in spec["inputs"].items():
        if P.sha256_file(C.resolve(cfg, v["path"])) != v["sha256"]:
            raise RuntimeError(f"input changed: {k}")
    return spec


def phase_spec(cfg, a):
    from . import provenance as P
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    h1 = _publish_hashed(out, "selection_clarification.json", _clarification(cfg))
    h2 = _publish_hashed(out, "eval_spec.json", _spec(cfg))
    receipt["outputs"] = {"selection_clarification.json": h1, "eval_spec.json": h2}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out)


def _adapted_model(cfg):
    import timm
    from . import provenance as P, trainer as TR
    p = _paths(cfg)
    with open(p["adapt_best"], encoding="utf-8") as f:
        best = json.load(f)
    check_selected(best, P.sha256_file(p["adapt_ckpt"]))
    ck = TR._load_verified(p["adapt_ckpt"], ADAPT_SHA)
    if ck["epoch_completed"] != 4:
        raise RuntimeError("adapted checkpoint is not epoch 4")
    m = timm.create_model("resnet50", pretrained=False, num_classes=1)
    m.load_state_dict(ck["model"], strict=True)
    return m.to("cuda").eval()


def phase_threshold(cfg, a):
    from . import config as C, decode as DC, provenance as P, threshold as TH, trainer as TR
    from . import rq5_stats as R5
    from .desc_4b3 import publish_csv
    from .predict_v1 import infer
    out = os.path.abspath(a.run_dir)
    spec = _load_spec(cfg, out)
    t0 = time.perf_counter()
    model = _adapted_model(cfg)
    ham = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    by = {r["record_uid"]: r for r in ham}
    uids = [r["record_uid"] for r in _read(_paths(cfg)["val_list"])]
    if sorted(uids) != sorted(u for u, r in by.items() if r["split"] == "val"):
        raise RuntimeError("HAM VAL list coverage")
    logits, probs, bi = infer(cfg, model, ham, "ham_val", uids)
    labels = [int(by[u]["label_mel"]) for u in uids]
    est = TH.estimate(probs, labels)
    with open(_paths(cfg)["adapt_val_epoch4"], encoding="utf-8") as f:
        v4 = json.load(f)
    secb = R5.secondary_b(v4["probs"], v4["labels"])
    if secb["status"] != "NOT_ESTIMABLE":
        raise RuntimeError("Secondary B unexpectedly estimable")
    thr = dict(est, created_utc=_now(), model_sha256=ADAPT_SHA, model="adapted (Stage 5A epoch 4)",
               dataset="HAM10000 VAL (1529)", val_list_sha256=spec["thresholds"]["adapted_secondary_A"]["val_list_sha256"],
               batching=bi, auroc_ham_val=TR.auroc(labels, probs), secondary_B=secb,
               frozen_before_other_adapted_inference=True)
    if est["status"] != "ESTIMATED":
        thr["note"] = "NOT ESTIMABLE: Secondary A thresholded endpoints invalid; no fallback"
    h = _publish_hashed(out, "threshold_adapted.json", thr)
    publish_csv(out, "pred_adapted_ham_val.csv",
                [{"record_uid": u, "label_mel": y, "logit": repr(lg), "prob": repr(p), "checkpoint_sha256": ADAPT_SHA}
                 for u, y, lg, p in zip(uids, labels, logits, probs)],
                ["record_uid", "label_mel", "logit", "prob", "checkpoint_sha256"])
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "threshold", "seconds": time.perf_counter() - t0})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("threshold_adapted.json", "pred_adapted_ham_val.csv")}
    P.write_receipt(receipt, out, name="receipt_threshold.json")
    print(json.dumps({k: thr[k] for k in ("status", "threshold", "k", "n_pos", "TP", "FN", "TN", "FP", "sensitivity",
                                          "specificity", "auroc_ham_val", "secondary_B")}, indent=1, default=str), "\n", h)


def phase_infer(cfg, a):
    from . import config as C, decode as DC, provenance as P, threshold as TH
    from .desc_4b3 import publish_csv, publish_json
    from .predict_v1 import infer
    out = os.path.abspath(a.run_dir)
    _load_spec(cfg, out)
    thr = _check_hashed(out, "threshold_adapted.json")
    t = thr["threshold"]
    th_sha = __import__("rv1.provenance", fromlist=["x"]).sha256_file(os.path.join(out, "threshold_adapted.json"))
    started = _now()
    if not thr["created_utc"] < started:
        raise RuntimeError("threshold not frozen before inference")
    model = _adapted_model(cfg)
    timing = {}
    jobs = [("pad_eval", "pad_ufes_20", "pad_ufes_20", lambda r: r["rq5_role"] == "eval"),
            ("ham_test", "ham10000", "ham_test", lambda r: r["split"] == "test"),
            ("derm7pt_clinical", "derm7pt", "derm7pt_clinical", lambda r: r["cache_name"] == "derm7pt_clinical")]
    for name, ds, cache, sel in jobs:
        allrows = DC.read_manifest(C.resolve(cfg, cfg["manifests"][ds]), ds)[0]
        rr = [r for r in allrows if sel(r)]
        t0 = time.perf_counter()
        logits, probs, bi = infer(cfg, model, allrows, cache, [r["record_uid"] for r in rr])
        timing[name] = {"seconds": time.perf_counter() - t0, "batching": bi}
        rows = []
        for r, lg, p in zip(rr, logits, probs):
            if not (math.isfinite(lg) and 0.0 <= p <= 1.0):
                raise RuntimeError(f"invalid score {r['record_uid']}")
            rows.append({"record_uid": r["record_uid"], "image_uid": r["image_uid"], "dataset": ds,
                         "cluster_id": r["cluster_id"], "pair_id": r["pair_id"], "rq5_role": r["rq5_role"],
                         "split": r["split"], "in_derm7pt_primary": r["in_derm7pt_primary"], "modality": r["modality"],
                         "label_mel": r["label_mel"], "logit": repr(lg), "prob": repr(p), "threshold": repr(t),
                         "pred": TH.predict(p, t), "checkpoint_sha256": ADAPT_SHA, "threshold_sha256": th_sha,
                         "prep_version": "prep_v1_cached256", "precision_policy": "xai_precision_v1", "batch": BATCH})
        publish_csv(out, f"pred_adapted_{name}.csv", rows, list(rows[0]))
    publish_json(out, "inference_log.json", {"started_utc": started, "timing": timing})
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "infer"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))
                          if n.startswith("pred_adapted_") or n == "inference_log.json"}
    rp = P.write_receipt(receipt, out, name="receipt_infer.json")
    print(json.dumps({k: v["seconds"] for k, v in timing.items()}), "verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def phase_analyze(cfg, a):
    import io
    import numpy as np
    from . import config as C, decode as DC, provenance as P, rq1_stats as RS, rq5_stats as R5
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    t_start = time.perf_counter()
    out = os.path.abspath(a.run_dir)
    spec = _load_spec(cfg, out)
    with open(os.path.join(out, "receipt_infer.json"), encoding="utf-8") as f:
        for n, h in json.load(f)["outputs"].items():
            if P.sha256_file(os.path.join(out, n)) != h:
                raise RuntimeError(f"changed: {n}")
    thr_a = _check_hashed(out, "threshold_adapted.json")
    p = _paths(cfg)
    tb = R5.ModelThreshold(BASE_SHA, spec["thresholds"]["base"]["value"], spec["thresholds"]["base"]["sha256"], "base")
    ta = R5.ModelThreshold(ADAPT_SHA, thr_a["threshold"], P.sha256_file(os.path.join(out, "threshold_adapted.json")),
                           "adapted")
    base = {"pad_eval": [r for r in _read(p["base_pred_pad"]) if r["rq5_role"] == "eval"],
            "ham_test": _read(p["base_pred_ham_test"]),
            "derm7pt_clinical": [r for r in _read(p["base_pred_derm7pt"]) if r["modality"] == "clinical"]}
    adapt = {k: _read(os.path.join(out, f"pred_adapted_{k}.csv")) for k in base}
    for k in base:                                                # verify every thresholded prediction
        for r in base[k]:
            if tb.predict_row(r) != int(r["pred"]) or r["threshold_sha256"] != tb.threshold_sha:
                raise RuntimeError(f"base pred check {r['record_uid']}")
        for r in adapt[k]:
            if ta.predict_row(r) != int(r["pred"]) or r["threshold_sha256"] != ta.threshold_sha:
                raise RuntimeError(f"adapted pred check {r['record_uid']}")
    meta = {}
    for k in base:
        meta[k] = {r["record_uid"]: r for r in adapt[k]}
    pad_man = [r for r in DC.read_manifest(C.resolve(cfg, cfg["manifests"]["pad_ufes_20"]), "pad_ufes_20")[0]
               if r["rq5_role"] == "eval"]
    if sorted(r["record_uid"] for r in pad_man) != sorted(meta["pad_eval"]):
        raise RuntimeError("PAD eval coverage")
    seeds = spec["bootstrap"]["seeds"]
    results = {"spec_sha256": P.sha256_file(os.path.join(out, "eval_spec.json")),
               "thresholds": {"base": tb.t, "adapted": ta.t, "adapted_secondary_B": thr_a["secondary_B"]}}
    reps = {}

    def analyse(name, joined, unit_key, seed, want_thresholded):
        clusters = {}
        for r in joined:
            clusters.setdefault(r[unit_key], []).append(r)
        ids = sorted(clusters)
        res = {"n_images": len(joined), "n_mel": sum(r["label"] for r in joined),
               "n_clusters": len(ids), "n_mel_clusters": sum(any(x["label"] for x in v) for v in clusters.values())}
        fams = {"auroc": R5.cluster_stat(R5.auc_pair)}
        if want_thresholded:
            fams["sensitivity"] = R5.cluster_stat(R5.sens_pair, tb.t, ta.t)
            fams["specificity"] = R5.cluster_stat(R5.spec_pair, tb.t, ta.t)
        for m, fn in fams.items():
            pt = fn([joined])
            b = RS.pair_bootstrap(clusters, ids, fn, B, seed)
            reps[f"{name}_{m}"] = b.pop("replicates")
            res[m] = {"base": pt[0], "adapted": pt[1], "diff_adapted_minus_base": pt[2],
                      "ci95": {"base": b["ci"][0], "adapted": b["ci"][1], "diff": b["ci"][2]} if b["ci"] else None,
                      "bootstrap": b}
        if want_thresholded:
            for lab, tt, key in (("base", tb.t, "p_base"), ("adapted", ta.t, "p_adapt")):
                res[f"counts_{lab}"] = {
                    "TP": sum(r["label"] == 1 and r[key] >= tt for r in joined),
                    "FN": sum(r["label"] == 1 and r[key] < tt for r in joined),
                    "TN": sum(r["label"] == 0 and r[key] < tt for r in joined),
                    "FP": sum(r["label"] == 0 and r[key] >= tt for r in joined)}
        return res

    j = R5.join_models(base["pad_eval"], adapt["pad_eval"], BASE_SHA, ADAPT_SHA)
    for r in j:
        r["unit"] = meta["pad_eval"][r["record_uid"]]["cluster_id"]
    results["PAD_eval"] = analyse("PAD_eval", j, "unit", seeds["PAD"], True)
    exp = (1150, 26, 687)
    got = (results["PAD_eval"]["n_images"], results["PAD_eval"]["n_mel"], results["PAD_eval"]["n_clusters"])
    if got != exp or results["PAD_eval"]["n_mel_clusters"] != 18:
        raise RuntimeError(f"PAD eval counts {got}")
    j = R5.join_models(base["ham_test"], adapt["ham_test"], BASE_SHA, ADAPT_SHA)
    for r in j:
        r["unit"] = meta["ham_test"][r["record_uid"]]["cluster_id"]
    results["HAM_TEST_forgetting"] = analyse("HAM_TEST", j, "unit", seeds["HAM"], False)
    jd = R5.join_models(base["derm7pt_clinical"], adapt["derm7pt_clinical"], BASE_SHA, ADAPT_SHA)
    for r in jd:
        m = meta["derm7pt_clinical"][r["record_uid"]]
        r["unit"], r["primary"] = m["pair_id"], m["in_derm7pt_primary"] == "1"
    results["Derm7pt_PRIMARY_clinical_forgetting"] = analyse("Derm7pt_PRIMARY", [r for r in jd if r["primary"]], "unit",
                                                             seeds["Derm7pt_PRIMARY"], False)
    results["Derm7pt_ALL_clinical_forgetting_sensitivity"] = analyse("Derm7pt_ALL", jd, "unit", seeds["Derm7pt_ALL"], False)
    if (results["Derm7pt_PRIMARY_clinical_forgetting"]["n_images"], results["Derm7pt_PRIMARY_clinical_forgetting"]["n_mel"]) != (1002, 249):
        raise RuntimeError("Derm7pt counts")
    results["analysis_runtime_s"] = time.perf_counter() - t_start
    publish_json(out, "results.json", results)
    tab = []
    for k in ("PAD_eval", "HAM_TEST_forgetting", "Derm7pt_PRIMARY_clinical_forgetting",
              "Derm7pt_ALL_clinical_forgetting_sensitivity"):
        for m in ("auroc", "sensitivity", "specificity"):
            if m in results[k]:
                x = results[k][m]
                tab.append({"cohort": k, "metric": m, "n_images": results[k]["n_images"], "n_mel": results[k]["n_mel"],
                            "n_clusters": results[k]["n_clusters"], "base": x["base"], "adapted": x["adapted"],
                            "diff": x["diff_adapted_minus_base"], "diff_ci_lo": x["ci95"]["diff"][0],
                            "diff_ci_hi": x["ci95"]["diff"][1], "undefined": x["bootstrap"]["n_undefined"],
                            "unreliable": x["bootstrap"]["unreliable"]})
    publish_csv(out, "paired_table.csv", tab, list(tab[0]))
    buf = io.BytesIO()
    np.savez_compressed(buf, **reps, columns=np.array(["base", "adapted", "diff"]))
    publish_bytes(out, "bootstrap_replicates.npz", buf.getvalue())
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("results.json", "paired_table.csv",
                                                                           "bootstrap_replicates.npz", "eval_spec.json")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    for r in tab:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("counts", results["PAD_eval"]["counts_base"], results["PAD_eval"]["counts_adapted"])
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "threshold", "infer", "analyze"))
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
    {"spec": phase_spec, "threshold": phase_threshold, "infer": phase_infer, "analyze": phase_analyze}[a.phase](cfg, a)


if __name__ == "__main__":
    main()
