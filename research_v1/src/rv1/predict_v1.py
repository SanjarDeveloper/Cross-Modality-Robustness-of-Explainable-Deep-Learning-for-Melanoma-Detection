"""Stage 4A (Addendum G): v1 threshold on HAM VAL, then prediction tables for HAM TEST and Derm7pt ALL.

Two separate processes so the threshold is persisted before any test/external inference:
  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.predict_v1 --config ../configs/predict_v1_resnet50_s0.json --phase threshold --out-root ../runs/predict_v1
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.predict_v1 --config ../configs/predict_v1_resnet50_s0.json --phase predict --run-dir ../runs/predict_v1/<run_id>
"""
import argparse
import csv
import datetime
import json
import os

from .xai_pilot import EXPECTED_SHA, _read_csv, _sidecar_ok

BATCH = 16
SAME_FILE_TOL = 1e-5


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write_with_sidecar(path, obj):
    from . import provenance as P
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    h = P.sha256_file(path)
    with open(path + ".sha256", "x", encoding="utf-8") as f:
        f.write(f"{h}  {os.path.basename(path)}\n")
    return h


def setup(cfg):
    import timm
    import torch
    from . import config as C, provenance as P, trainer as TR
    si = cfg["supporting_inputs"]
    ck = C.resolve(cfg, si["model_checkpoint"])
    with open(C.resolve(cfg, si["model_best_pointer"]), encoding="utf-8") as f:
        best = json.load(f)
    sha = P.sha256_file(ck)
    if not (sha == best["sha256"] == EXPECTED_SHA and _sidecar_ok(ck) and best["checkpoint"] == os.path.basename(ck)):
        raise RuntimeError("checkpoint verification failed")
    st = TR._load_verified(ck, EXPECTED_SHA)
    model = timm.create_model("resnet50", pretrained=False, num_classes=1)
    model.load_state_dict(st["model"], strict=True)
    return model.to("cuda").eval(), {"checkpoint": si["model_checkpoint"], "sha256": sha, "best_epoch": best["epoch"],
                                     "sidecar_ok": True, "best_json_ok": True}


def infer(cfg, model, rows_all, cache_name, uids):
    import torch
    from . import torch_data as TD, xai_attr as XA
    root = cfg["project_root"]
    ds = TD.ManifestDataset(rows_all, cache_name, os.path.join(root, "cache"), os.path.join(root, "DATASETS"),
                            uids, mode="eval")
    out_u, logits, probs = [], [], []
    with XA.xai_precision(), torch.no_grad():
        for b in TD.make_epoch_loader(ds, 0, BATCH, False, 0, 0):
            lg = model(b["x"].to("cuda")).squeeze(1).float()
            logits += lg.cpu().tolist()
            probs += torch.sigmoid(lg).cpu().tolist()
            out_u += [r["record_uid"] for r in b["record"]]
    if out_u != list(uids):
        raise RuntimeError(f"{cache_name}: order changed")
    return logits, probs, {"cache": cache_name, "n": len(uids), "batch": BATCH, "n_full_batches": len(uids) // BATCH,
                           "final_partial_batch": len(uids) % BATCH, "order": "as listed (manifest/val-list order)"}


def phase_threshold(cfg, a):
    from . import config as C, decode as DC, provenance as P, threshold as TH, trainer as TR, xai_attr as XA
    model, mc = setup(cfg)
    si = cfg["supporting_inputs"]
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    vl = _read_csv(C.resolve(cfg, si["val_list"]))
    uids = [r["record_uid"] for r in vl]
    by = {r["record_uid"]: r for r in rows}
    if len(uids) != 1529 or len(set(uids)) != 1529 or any(by[u]["split"] != "val" for u in uids) or \
            sorted(uids) != sorted(r["record_uid"] for r in rows if r["split"] == "val"):
        raise RuntimeError("val list coverage failure")
    logits, probs, bi = infer(cfg, model, rows, "ham_val", uids)
    labels = [int(by[u]["label_mel"]) for u in uids]
    with open(C.resolve(cfg, si["stage3a_val_epoch8"]), encoding="utf-8") as f:
        v3 = json.load(f)
    cmp = {"record_uids_equal": v3["record_uids"] == uids, "labels_equal": v3["labels"] == labels,
           "max_abs_prob_diff": max(abs(x - y) for x, y in zip(v3["probs"], probs)),
           "auroc_recomputed": TR.auroc(labels, probs), "auroc_stage3a": v3["val_auroc"]}
    cmp["pass"] = cmp["record_uids_equal"] and cmp["labels_equal"] and cmp["max_abs_prob_diff"] == 0.0
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "threshold", "model": mc, "batching": bi})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    with open(os.path.join(out, "val_reproduction.json"), "x", encoding="utf-8") as f:
        json.dump(cmp, f, indent=1)
    if not cmp["pass"]:
        raise RuntimeError(f"HAM VAL reproduction failed: {cmp}")
    est = TH.estimate(probs, labels)
    lesion = {}
    for u, p, y in zip(uids, probs, labels):
        if y == 1:
            lesion.setdefault(by[u]["cluster_id"], []).append(p)
    if est["status"] == "ESTIMATED":
        det = sum(any(p >= est["threshold"] for p in ps) for ps in lesion.values())
        est["lesion_level_sensitivity_descriptive"] = {"detected_lesions": det, "melanoma_lesions": len(lesion)}
    with XA.xai_precision():
        pol = XA.precision_settings()
    thr = dict(est, created_utc=_now(), dataset="HAM10000 VAL (split_seed 0)", n_images=len(uids),
               val_manifest=cfg["manifests"]["ham10000"],
               val_manifest_sha256=P.sha256_file(C.resolve(cfg, cfg["manifests"]["ham10000"])),
               val_list=si["val_list"], val_list_sha256=P.sha256_file(C.resolve(cfg, si["val_list"])),
               model=mc, prep_version="prep_v1_cached256", precision_policy=pol, batching=bi,
               config_sha256=C.config_hash(cfg), frozen_before_external_inference=True,
               note="Selected on HAM VAL (also used for checkpoint selection). No later dataset may alter it.")
    h = _write_with_sidecar(os.path.join(out, "threshold.json"), thr)
    with open(os.path.join(out, "ham_val_predictions.csv"), "x", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["record_uid", "cluster_id", "label_mel", "logit", "prob", "pred"])
        for u, lg, p, y in zip(uids, logits, probs, labels):
            w.writerow([u, by[u]["cluster_id"], y, repr(lg), repr(p),
                        TH.predict(p, est["threshold"]) if est["threshold"] is not None else ""])
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))}
    rp = P.write_receipt(receipt, out, name="receipt_threshold.json")
    print(json.dumps({k: thr[k] for k in thr if k not in ("precision_policy",)}, indent=1))
    print("threshold sha256:", h, "\nrun dir:", out, "\nverify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def phase_predict(cfg, a):
    import numpy as np
    from . import config as C, decode as DC, provenance as P, threshold as TH, trainer as TR
    out = os.path.abspath(a.run_dir)
    tp = os.path.join(out, "threshold.json")
    with open(tp + ".sha256", encoding="utf-8") as f:
        th_sha = f.read().split()[0]
    if P.sha256_file(tp) != th_sha:
        raise RuntimeError("threshold.json hash mismatch")
    with open(tp, encoding="utf-8") as f:
        thr = json.load(f)
    if thr["status"] != "ESTIMATED":
        t = None
    else:
        t = thr["threshold"]
        if repr(t) != thr["threshold_repr"]:
            raise RuntimeError("threshold changed in serialization")
    model, mc = setup(cfg)
    if mc["sha256"] != thr["model"]["sha256"]:
        raise RuntimeError("model differs from threshold model")
    started = _now()
    ham = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    derm = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["derm7pt"]), "derm7pt")[0]
    ver = {"threshold_sha256": th_sha, "inference_started_utc": started, "threshold_created_utc": thr["created_utc"],
           "threshold_before_inference": thr["created_utc"] < started}
    jobs = [("ham_test", [r for r in ham if r["split"] == "test"]),
            ("derm7pt_dermoscopy", [r for r in derm if r["cache_name"] == "derm7pt_dermoscopy"]),
            ("derm7pt_clinical", [r for r in derm if r["cache_name"] == "derm7pt_clinical"])]
    rows_out, batching = [], {}
    for cache, rr in jobs:
        allrows = ham if cache == "ham_test" else derm
        uids = [r["record_uid"] for r in rr]
        logits, probs, bi = infer(cfg, model, allrows, cache, uids)
        batching[cache] = bi
        for r, lg, p in zip(rr, logits, probs):
            ok = np.isfinite(lg) and np.isfinite(p)
            rows_out.append({"record_uid": r["record_uid"], "image_uid": r["image_uid"], "pair_id": r["pair_id"],
                             "cluster_id": r["cluster_id"], "dataset": r["dataset"], "modality": r["modality"],
                             "label_mel": r["label_mel"], "split": r["split"],
                             "in_derm7pt_all": r["in_derm7pt_all"], "in_derm7pt_primary": r["in_derm7pt_primary"],
                             "same_file_pair": r["same_file_pair"], "cohort_flags": r["cohort_flags"],
                             "logit": repr(lg), "prob": repr(p), "threshold": repr(t) if t is not None else "",
                             "pred": (TH.predict(p, t) if ok else "") if t is not None else "",
                             "status": "ok" if ok else "FAILURE_nonfinite",
                             "checkpoint_sha256": mc["sha256"], "threshold_sha256": th_sha,
                             "prep_version": "prep_v1_cached256", "precision_policy": "xai_precision_v1",
                             "batch": BATCH, "cache_name": cache})
    fields = list(rows_out[0])
    for name, keep in (("predictions_ham_test.csv", lambda r: r["cache_name"] == "ham_test"),
                       ("predictions_derm7pt_all.csv", lambda r: r["cache_name"] != "ham_test")):
        with open(os.path.join(out, name), "x", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows([r for r in rows_out if keep(r)])
    # ---------------- verification
    ht = _read_csv(os.path.join(out, "predictions_ham_test.csv"))
    dt = _read_csv(os.path.join(out, "predictions_derm7pt_all.csv"))
    ver["reload_equal"] = [dict(r, batch=str(r["batch"]), pred=str(r["pred"])) for r in rows_out] == ht + dt
    ver["ham_test"] = {"n": len(ht), "unique": len({r["record_uid"] for r in ht}),
                       "coverage_equal_manifest_test": sorted(r["record_uid"] for r in ht) ==
                       sorted(r["record_uid"] for r in ham if r["split"] == "test"),
                       "melanoma": sum(r["label_mel"] == "1" for r in ht),
                       "failures": sum(r["status"] != "ok" for r in ht)}
    byu = {r["record_uid"]: r for r in derm}
    pairs = {}
    for r in dt:
        pairs.setdefault(r["pair_id"], []).append(r)
    ver["derm7pt"] = {"n_rows": len(dt), "unique": len({r["record_uid"] for r in dt}),
                      "coverage_equal_manifest": sorted(r["record_uid"] for r in dt) == sorted(byu),
                      "identity_join_ok": all(byu[r["record_uid"]]["image_uid"] == r["image_uid"] and
                                              byu[r["record_uid"]]["label_mel"] == r["label_mel"] for r in dt),
                      "n_pairs_all": len(pairs),
                      "pairs_two_modalities_same_label": all(sorted(x["modality"] for x in v) == ["clinical", "dermoscopy"]
                                                             and len({x["label_mel"] for x in v}) == 1
                                                             and len({x["in_derm7pt_primary"] for x in v}) == 1
                                                             for v in pairs.values()),
                      "n_pairs_primary": sum(v[0]["in_derm7pt_primary"] == "1" for v in pairs.values()),
                      "n_mel_pairs_primary": sum(v[0]["in_derm7pt_primary"] == "1" and v[0]["label_mel"] == "1"
                                                 for v in pairs.values()),
                      "n_mel_pairs_all": sum(v[0]["label_mel"] == "1" for v in pairs.values()),
                      "failures": sum(r["status"] != "ok" for r in dt)}
    ver["derm7pt"]["counts_ok"] = (ver["derm7pt"]["n_pairs_all"] == 1011 and ver["derm7pt"]["n_rows"] == 2022 and
                                   ver["derm7pt"]["n_pairs_primary"] == 1002 and ver["derm7pt"]["n_mel_pairs_primary"] == 249)
    # same-file pairs
    root = cfg["project_root"]
    imgs = {c: np.load(os.path.join(root, "cache", f"{c}_img.npy"), mmap_mode="r")
            for c in ("derm7pt_dermoscopy", "derm7pt_clinical")}
    sf = []
    for pid, v in pairs.items():
        if v[0]["same_file_pair"] != "1":
            continue
        d = {x["modality"]: x for x in v}
        a_, b_ = (imgs["derm7pt_" + m][int(byu[d[m]["record_uid"]]["cache_index"])] for m in ("dermoscopy", "clinical"))
        dp = abs(float(d["dermoscopy"]["prob"]) - float(d["clinical"]["prob"]))
        sf.append({"pair_id": pid, "cache_bytes_identical": bool(np.array_equal(a_, b_)), "abs_prob_diff": dp,
                   "abs_logit_diff": abs(float(d["dermoscopy"]["logit"]) - float(d["clinical"]["logit"])),
                   "pred_equal": d["dermoscopy"]["pred"] == d["clinical"]["pred"],
                   "batch_positions": {m: (int(byu[d[m]["record_uid"]]["cache_index"]) // BATCH,
                                           int(byu[d[m]["record_uid"]]["cache_index"]) % BATCH) for m in d},
                   "pass": bool(np.array_equal(a_, b_)) and dp <= SAME_FILE_TOL and
                   d["dermoscopy"]["pred"] == d["clinical"]["pred"]})
    ver["same_file_pairs"] = {"n": len(sf), "tolerance_prob": SAME_FILE_TOL, "all_pass": all(x["pass"] for x in sf),
                              "rows": sf}
    # deterministic recompute: first batch of each cache (identical batch composition)
    rc = {}
    for cache, rr in jobs:
        uids = [r["record_uid"] for r in rr][:BATCH]
        lg, pr, _ = infer(cfg, model, ham if cache == "ham_test" else derm, cache, uids)
        stored = {r["record_uid"]: r for r in rows_out}
        rc[cache] = max(abs(float(stored[u]["prob"]) - p) for u, p in zip(uids, pr))
    ver["recompute_first_batch_max_abs_prob_diff"] = rc
    ver["recompute_exact"] = all(v == 0.0 for v in rc.values())
    ver["checkpoint_sha256_after"] = P.sha256_file(C.resolve(cfg, cfg["supporting_inputs"]["model_checkpoint"]))
    ver["threshold_sha256_after"] = P.sha256_file(tp)
    ver["hashes_unchanged"] = ver["checkpoint_sha256_after"] == EXPECTED_SHA and ver["threshold_sha256_after"] == th_sha
    ver["batching"] = batching
    with open(os.path.join(out, "verification.json"), "x", encoding="utf-8") as f:
        json.dump(ver, f, indent=1)
    # ---------------- descriptive tables (from the persisted CSVs)
    def desc(rr):
        p = [float(r["prob"]) for r in rr]
        y = [int(r["label_mel"]) for r in rr]
        c = TH.counts(p, y, t) if t is not None else {}
        return dict({"n": len(rr), "n_mel": sum(y), "n_nonmel": len(y) - sum(y),
                     "auroc": TR.auroc(y, p) if 0 < sum(y) < len(y) else None}, **c)
    table = []
    table.append(dict({"dataset": "HAM10000 TEST", "cohort": "test split", "modality": "dermoscopy"}, **desc(ht)))
    for coh, sel in (("Derm7pt PRIMARY (1002 pairs)", lambda r: r["in_derm7pt_primary"] == "1"),
                     ("Derm7pt ALL (1011 pairs; sensitivity)", lambda r: True)):
        for mod in ("dermoscopy", "clinical"):
            table.append(dict({"dataset": "Derm7pt", "cohort": coh, "modality": mod},
                              **desc([r for r in dt if sel(r) and r["modality"] == mod])))
    paired = {}
    for coh, prim in (("primary", True), ("all", False)):
        cnt = {"both_detected": 0, "dermoscopy_only": 0, "clinical_only": 0, "both_missed": 0}
        for pid, v in pairs.items():
            if v[0]["label_mel"] != "1" or (prim and v[0]["in_derm7pt_primary"] != "1"):
                continue
            d = {x["modality"]: x["pred"] for x in v}
            key = {("1", "1"): "both_detected", ("1", "0"): "dermoscopy_only", ("0", "1"): "clinical_only",
                   ("0", "0"): "both_missed"}[(d["dermoscopy"], d["clinical"])]
            cnt[key] += 1
        cnt["n_melanoma_pairs"] = sum(cnt.values())
        paired[coh] = cnt
    with open(os.path.join(out, "descriptive_table.csv"), "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "cohort", "modality", "n", "n_mel", "n_nonmel", "TP", "FN", "TN", "FP",
                                          "sens_num_den", "sensitivity", "spec_num_den", "specificity", "auroc"],
                           lineterminator="\n")
        w.writeheader()
        for r in table:
            w.writerow({k: (f"{r[k][0]}/{r[k][1]}" if k.endswith("num_den") else r.get(k)) for k in w.fieldnames})
    with open(os.path.join(out, "paired_melanoma_counts.json"), "x", encoding="utf-8") as f:
        json.dump({"note": "descriptive; feeds later exact paired test; no test computed here",
                   "threshold_sha256": th_sha, **paired}, f, indent=1)
    receipt = P.build_receipt(cfg, a.config, "completed",
                              details={"phase": "predict", "model": mc, "threshold_sha256": th_sha})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))
                          if not n.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_predict.json")
    print(json.dumps({k: v for k, v in ver.items() if k not in ("batching",)}, indent=1, default=str)[:4000])
    for r in table:
        print(r)
    print(json.dumps(paired, indent=1))
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("threshold", "predict"))
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
    (phase_threshold if a.phase == "threshold" else phase_predict)(cfg, a)


if __name__ == "__main__":
    main()
