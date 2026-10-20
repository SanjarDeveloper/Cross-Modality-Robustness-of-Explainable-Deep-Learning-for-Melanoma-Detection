"""Stage 9C1: unperturbed RQ3 evaluation of the nine standard cells (ResNet-50 / EfficientNet-B3 /
DenseNet-121 x seeds 0-2). Phases (separate processes): spec -> threshold -> infer -> analyze.

Reuse: ResNet s0 threshold + HAM TEST + Derm7pt (Stage 4A) + PAD all (Stage 4C); ResNet s1/s2 thresholds +
Derm7pt (Stage 6B). Everything else is inferred here with the verified prep_v1_cached256 path,
xai_precision_v1, batch 16, manifest order. DDI is out of scope (not required for E1.1-E1.5).

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq3_eval --config ../configs/rq3_eval_9c1_v1.json --phase spec --out-root ../runs/rq3_eval_9c1
  ... --phase threshold|infer|analyze --run-dir <dir>
"""
import argparse
import csv
import datetime
import glob
import io
import json
import math
import os
import time

import numpy as np

B = 2000
P4A = "research_v1/runs/predict_v1/20261001T183557Z_20b3af67_4959f943/"
P4C = "research_v1/runs/ext_eval_4c/20261003T160420Z_c5ef9dda_9190c7cf/"
P6B = "research_v1/runs/seed_spread_6b/20261003T171246Z_16d5722f_d2b87d3d/"
COHORTS = ("ham_test", "derm7pt", "pad_all")


def models():
    m = [{"id": "resnet50_s0", "arch": "resnet50", "seed": 0, "run": "research_v1/runs/main_train/resnet50_s0_20260927T155924Z",
          "threshold_src": P4A + "threshold.json",
          "reuse": {"ham_test": P4A + "predictions_ham_test.csv", "derm7pt": P4A + "predictions_derm7pt_all.csv",
                    "pad_all": P4C + "predictions_pad_ufes_20.csv"}},
         {"id": "resnet50_s1", "arch": "resnet50", "seed": 1, "run": "research_v1/runs/main_train/resnet50_s1_20261003T163838Z",
          "threshold_src": P6B + "threshold_s1.json", "reuse": {"derm7pt": P6B + "derm7pt_pred_s1.csv"}},
         {"id": "resnet50_s2", "arch": "resnet50", "seed": 2, "run": "research_v1/runs/main_train/resnet50_s2_20261003T164841Z",
          "threshold_src": P6B + "threshold_s2.json", "reuse": {"derm7pt": P6B + "derm7pt_pred_s2.csv"}}]
    for arch, sh in (("efficientnet_b3", "effb3"), ("densenet121", "dn121")):
        for s in (0, 1, 2):
            m.append({"id": f"{arch}_s{s}", "arch": arch, "seed": s, "run": f"research_v1/runs/main_train/rq3_{sh}_s{s}_9b",
                      "threshold_src": None, "reuse": {}})
    return m


def check_selected(best, ck_sha, ck, arch, seed):
    """best.json must point to the selected (not merely latest) checkpoint and match its contents."""
    if best["sha256"] != ck_sha:
        raise RuntimeError("best.json hash != checkpoint file")
    if ck["epoch_completed"] != best["epoch"] or ck["config"]["training"]["arch"] != arch or \
            ck["config"]["seeds"]["train_seed"] != seed:
        raise RuntimeError("checkpoint identity mismatch")
    return True


def threshold_value(thr, model_sha):
    sha = thr.get("model_sha256") or thr.get("model", {}).get("sha256")
    if sha != model_sha or thr.get("status") != "ESTIMATED":
        raise ValueError("threshold not bound to this model")
    return thr["threshold"]


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _abs(cfg, rel):
    from . import config as C
    return C.resolve(cfg, rel)


def _read(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _hashed_json(out, name, obj=None):
    from . import provenance as P
    from .desc_4b3 import publish_bytes, publish_json
    p = os.path.join(out, name)
    if obj is not None:
        publish_json(out, name, obj)
        publish_bytes(out, name + ".sha256", f"{P.sha256_file(p)}  {name}\n".encode())
        return P.sha256_file(p)
    with open(p + ".sha256", encoding="utf-8") as f:
        if f.read().split()[0] != P.sha256_file(p):
            raise RuntimeError(f"{name} hash mismatch")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def phase_spec(cfg, a):
    from . import provenance as P
    reg = []
    for m in models():
        run = _abs(cfg, m["run"])
        with open(os.path.join(run, "best.json"), encoding="utf-8") as f:
            best = json.load(f)
        ck = os.path.join(run, best["checkpoint"])
        sha = P.sha256_file(ck)
        with open(ck + ".sha256", encoding="utf-8") as f:
            side = f.read().split()[0]
        if not (sha == side == best["sha256"]):
            raise RuntimeError(f"{m['id']}: hash mismatch")
        entry = dict(m, checkpoint=best["checkpoint"], epoch=best["epoch"], sha256=sha, best_val_auroc=best["val_auroc"],
                     reuse={k: {"path": v, "sha256": P.sha256_file(_abs(cfg, v))} for k, v in m["reuse"].items()})
        if m["threshold_src"]:
            entry["threshold_src"] = {"path": m["threshold_src"], "sha256": P.sha256_file(_abs(cfg, m["threshold_src"]))}
        reg.append(entry)
    s = cfg["seeds"]["bootstrap_seed"]
    spec = {"spec_version": "rq3_eval_9c1_v1", "created_utc": _now(), "models": reg,
            "manifests": {k: {"path": cfg["manifests"][k], "sha256": P.sha256_file(_abs(cfg, cfg["manifests"][k]))}
                          for k in ("ham10000", "derm7pt", "pad_ufes_20")},
            "cohorts": {"ham_test": "HAM10000 split=test (1548; lesion clusters 1155)",
                        "derm7pt": "Derm7pt ALL 2022 rows / 1011 pairs; PRIMARY flag retained (1002 pairs, 249 mel)",
                        "pad_all": "PAD-UFES-20 all 2298 images, rq5_role retained; patient clusters",
                        "excluded": "DDI (not required for E1.1-E1.5; out of this bounded stage)"},
            "inference": {"prep": "prep_v1_cached256", "precision": "xai_precision_v1", "batch": 16, "order": "manifest order per cache"},
            "threshold_rule": "v1_highest_t_img_sens95 on full HAM VAL (val-list order), per model; ResNet thresholds reused",
            "analyses": "per model: HAM TEST sens/spec/AUROC (lesion bootstrap); Derm7pt PRIMARY E1.1 counts + exact McNemar "
                        "(descriptive in RQ3), E1.2 sens diff (pair bootstrap), E1.3 AUROCs + paired DeLong + pair bootstrap, "
                        "E1.4 correctness + correctness McNemar + ECE (ece_pos_v1); PAD all sens/spec/AUROC (patient bootstrap); "
                        "Derm7pt ALL E1.1/E1.2 sensitivity analysis",
            "bootstrap": {"B": B, "seeds": {"ham_test": s, "derm_primary_sens": s + 1, "derm_all_sens": s + 2, "derm_auc": s + 3,
                                            "derm_spec": s + 4, "pad_all": s + 5},
                          "common_draws": "the same seed per cohort for every model -> identical cluster draws across models",
                          "note": "RQ3 comparison outputs are recomputed for all nine models (incl. ResNet s0) with these "
                                  "common draws; previously published Stage 4B/4C/6B intervals (other seeds) are unchanged"},
            "no_tests_between_architectures": True, "primary_model": "resnet50_s0 (unchanged)"}
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    receipt["outputs"] = {"spec.json": _hashed_json(out, "spec.json", spec)}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out)
    for e in reg:
        print(e["id"], e["checkpoint"], e["sha256"][:12])


def _model(cfg, entry):
    import timm
    from . import trainer as TR
    run = _abs(cfg, entry["run"])
    ck = TR._load_verified(os.path.join(run, entry["checkpoint"]), entry["sha256"])
    with open(os.path.join(run, "best.json"), encoding="utf-8") as f:
        check_selected(json.load(f), entry["sha256"], ck, entry["arch"], entry["seed"])
    m = timm.create_model(entry["arch"], pretrained=False, num_classes=1)
    m.load_state_dict(ck["model"], strict=True)
    return m.to("cuda").eval()


def _load_spec(cfg, out):
    from . import provenance as P
    spec = _hashed_json(out, "spec.json")
    for e in spec["models"]:
        if P.sha256_file(os.path.join(_abs(cfg, e["run"]), e["checkpoint"])) != e["sha256"]:
            raise RuntimeError(f"{e['id']} checkpoint changed")
        for k, v in e["reuse"].items():
            if P.sha256_file(_abs(cfg, v["path"])) != v["sha256"]:
                raise RuntimeError(f"reused artifact changed {e['id']}/{k}")
    return spec


def phase_threshold(cfg, a):
    from . import decode as DC, provenance as P, threshold as TH, trainer as TR
    from .predict_v1 import infer
    out = os.path.abspath(a.run_dir)
    spec = _load_spec(cfg, out)
    ham = DC.read_manifest(_abs(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    by = {r["record_uid"]: r for r in ham}
    uids = [r["record_uid"] for r in _read(_abs(cfg, cfg["supporting_inputs"]["val_list"]))]
    reg = {}
    for e in spec["models"]:
        if e["threshold_src"]:
            with open(_abs(cfg, e["threshold_src"]["path"]), encoding="utf-8") as f:
                thr = json.load(f)
            reg[e["id"]] = {"threshold": threshold_value(thr, e["sha256"]), "source": "reused", "path": e["threshold_src"]["path"],
                            "sha256_of_source": e["threshold_src"]["sha256"], "model_sha256": e["sha256"]}
            continue
        t0 = time.perf_counter()
        model = _model(cfg, e)
        _, probs, bi = infer(cfg, model, ham, "ham_val", uids)
        labels = [int(by[u]["label_mel"]) for u in uids]
        run = _abs(cfg, e["run"])
        with open(glob.glob(os.path.join(run, f"val_epoch{e['epoch']}_*.json"))[0], encoding="utf-8") as f:
            saved = json.load(f)
        d = max(abs(x - y) for x, y in zip(saved["probs"], probs))
        if saved["record_uids"] != uids or saved["labels"] != labels or d > 1e-6:
            raise RuntimeError(f"{e['id']}: HAM VAL reproduction failed ({d})")
        est = TH.estimate(probs, labels)
        reg[e["id"]] = dict(est, source="new (Stage 9C1)", model_sha256=e["sha256"], val_reproduction_max_abs_diff=d,
                            auroc_ham_val=TR.auroc(labels, probs), seconds=time.perf_counter() - t0)
        print(e["id"], est["threshold"], est["TP"], est["FN"], est["TN"], est["FP"], d, flush=True)
    h = _hashed_json(out, "threshold_registry.json", {"created_utc": _now(), "frozen_before_external_inference": True, "thresholds": reg})
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "threshold"})
    receipt["outputs"] = {"threshold_registry.json": h}
    P.write_receipt(receipt, out, name="receipt_threshold.json")


def phase_infer(cfg, a):
    from . import decode as DC, provenance as P, threshold as TH
    from .desc_4b3 import publish_csv, publish_json
    from .predict_v1 import infer
    out = os.path.abspath(a.run_dir)
    spec = _load_spec(cfg, out)
    reg = _hashed_json(out, "threshold_registry.json")
    with open(os.path.join(out, "threshold_registry.json"), encoding="utf-8") as f:
        pass
    started = _now()
    if not reg["created_utc"] < started:
        raise RuntimeError("threshold registry not frozen first")
    reg_sha = P.sha256_file(os.path.join(out, "threshold_registry.json"))
    mans = {"ham10000": DC.read_manifest(_abs(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0],
            "derm7pt": DC.read_manifest(_abs(cfg, cfg["manifests"]["derm7pt"]), "derm7pt")[0],
            "pad_ufes_20": DC.read_manifest(_abs(cfg, cfg["manifests"]["pad_ufes_20"]), "pad_ufes_20")[0]}
    jobs = {"ham_test": [("ham10000", "ham_test", lambda r: r["split"] == "test")],
            "derm7pt": [("derm7pt", "derm7pt_dermoscopy", lambda r: r["cache_name"] == "derm7pt_dermoscopy"),
                        ("derm7pt", "derm7pt_clinical", lambda r: r["cache_name"] == "derm7pt_clinical")],
            "pad_all": [("pad_ufes_20", "pad_ufes_20", lambda r: True)]}
    timing, statuses = {}, {}
    for e in spec["models"]:
        missing = [c for c in COHORTS if c not in e["reuse"]]
        statuses[e["id"]] = {c: ("reused" if c in e["reuse"] else "new") for c in COHORTS}
        if not missing:
            continue
        t = threshold_value({"model_sha256": reg["thresholds"][e["id"]]["model_sha256"], "status": "ESTIMATED",
                             "threshold": reg["thresholds"][e["id"]]["threshold"]}, e["sha256"])
        model = _model(cfg, e)
        for c in missing:
            fname = f"pred_{e['id']}_{c}.csv"
            if os.path.exists(os.path.join(out, fname)):
                continue
            t0 = time.perf_counter()
            rows = []
            for ds, cache, sel in jobs[c]:
                rr = [r for r in mans[ds] if r["cache_name"] == cache and sel(r)]
                lg, pr, _ = infer(cfg, model, mans[ds], cache, [r["record_uid"] for r in rr])
                for r, l, p in zip(rr, lg, pr):
                    if not (math.isfinite(l) and 0.0 <= p <= 1.0):
                        raise RuntimeError(f"invalid score {r['record_uid']}")
                    rows.append({"record_uid": r["record_uid"], "image_uid": r["image_uid"], "pair_id": r["pair_id"],
                                 "cluster_id": r["cluster_id"], "dataset": r["dataset"], "modality": r["modality"],
                                 "label_mel": r["label_mel"], "split": r["split"], "rq5_role": r["rq5_role"],
                                 "in_derm7pt_primary": r["in_derm7pt_primary"], "same_file_pair": r["same_file_pair"],
                                 "logit": repr(l), "prob": repr(p), "threshold": repr(t), "pred": TH.predict(p, t),
                                 "status": "ok", "checkpoint_sha256": e["sha256"], "threshold_sha256": reg_sha,
                                 "model_id": e["id"], "prep_version": "prep_v1_cached256", "precision_policy": "xai_precision_v1",
                                 "batch": 16})
            publish_csv(out, fname, rows, list(rows[0]))
            timing[f"{e['id']}|{c}"] = time.perf_counter() - t0
            print(e["id"], c, len(rows), round(timing[f"{e['id']}|{c}"], 1), flush=True)
    publish_json(out, "inference_log.json", {"started_utc": started, "timing_s": timing, "status": statuses})
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "infer"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))
                          if n.startswith("pred_") or n == "inference_log.json"}
    P.write_receipt(receipt, out, name="receipt_infer.json")


# ------------------------------------------------------------------ analysis
def _rows_for(cfg, out, e, cohort, reg_sha):
    if cohort in e["reuse"]:
        rows = _read(_abs(cfg, e["reuse"][cohort]["path"]))
    else:
        rows = _read(os.path.join(out, f"pred_{e['id']}_{cohort}.csv"))
    if any(r["checkpoint_sha256"] != e["sha256"] for r in rows):
        raise RuntimeError(f"{e['id']}/{cohort}: checkpoint identity")
    return rows


def _cluster_boot(rows, key, fn, seed):
    from . import rq1_stats as RS
    cl = {}
    for r in rows:
        cl.setdefault(r[key], []).append(r)
    ids = sorted(cl)
    b = RS.pair_bootstrap(cl, ids, lambda cs: fn([x for c in cs for x in c]), B, seed)
    return b


def _bin(rows, t):
    from . import rq2_stats as R2
    p = np.array([float(r["prob"]) for r in rows])
    y = np.array([r["label_mel"] == "1" for r in rows])
    sens = float((p[y] >= t).mean()) if y.any() else None
    spec = float((p[~y] < t).mean()) if (~y).any() else None
    return sens, spec, R2.auc_ranks(p, y)


def analyse_model(cfg, out, e, t, seeds):
    from . import auroc_paired as AP, desc_stats as DS, rq1_stats as RS, rq2_stats as R2, threshold as TH
    res = {"id": e["id"], "arch": e["arch"], "seed": e["seed"], "epoch": e["epoch"], "threshold": t, "sha256": e["sha256"]}
    # HAM TEST
    ht = _rows_for(cfg, out, e, "ham_test", None)
    c = TH.counts([float(r["prob"]) for r in ht], [int(r["label_mel"]) for r in ht], t)
    s_, sp_, au = _bin(ht, t)
    b = _cluster_boot(ht, "cluster_id", lambda rr: (lambda v: None if None in v else v)(_bin(rr, t)), seeds["ham_test"])
    res["ham_test"] = dict(c, auroc=au, n=len(ht), ci95={"sens": b["ci"][0], "spec": b["ci"][1], "auroc": b["ci"][2]},
                           undefined=b["n_undefined"])
    # PAD all
    pa = _rows_for(cfg, out, e, "pad_all", None)
    c = TH.counts([float(r["prob"]) for r in pa], [int(r["label_mel"]) for r in pa], t)
    _, _, au = _bin(pa, t)
    b = _cluster_boot(pa, "cluster_id", lambda rr: (lambda v: None if None in v else v)(_bin(rr, t)), seeds["pad_all"])
    res["pad_all"] = dict(c, auroc=au, n=len(pa), ci95={"sens": b["ci"][0], "spec": b["ci"][1], "auroc": b["ci"][2]},
                          undefined=b["n_undefined"])
    # Derm7pt
    dr = _rows_for(cfg, out, e, "derm7pt", None)
    th_sha = dr[0]["threshold_sha256"]
    pairs = RS.build_pairs(dr, t, th_sha, e["sha256"])
    for name, coh, ss in (("PRIMARY", {k: v for k, v in pairs.items() if v["primary"]}, seeds["derm_primary_sens"]),
                          ("ALL", pairs, seeds["derm_all_sens"])):
        pl = list(coh.values())
        mel = RS.paired_table(pl, 1)
        mc = RS.mcnemar_exact(mel["first_only"], mel["second_only"])
        sd, sc, dd = RS.sens_diff(pl, 1)
        ids = sorted(k for k, v in coh.items() if v["label"] == 1)
        bs = RS.pair_bootstrap(coh, ids, lambda ps: RS.sens_diff(ps, 1), B, ss)
        blk = {"n_pairs": len(pl), "n_mel_pairs": len(ids), "b": mel["first_only"], "c": mel["second_only"],
               "both": mel["both"], "neither": mel["neither"], "sens_derm": sd, "sens_clin": sc, "diff_pp": 100 * dd,
               "diff_ci_pp": [100 * x for x in bs["ci"][2]], "mcnemar_p_descriptive": mc["p_value"]}
        if name == "PRIMARY":
            ps = [coh[k] for k in sorted(coh)]
            y = [p["label"] for p in ps]
            dl = AP.delong_paired([p["prob_d"] for p in ps], [p["prob_c"] for p in ps], y)

            def aucs(pp):
                yy = np.array([q["label"] == 1 for q in pp])
                if yy.all() or not yy.any():
                    return None
                ad = R2.auc_ranks([q["prob_d"] for q in pp], yy)
                ac = R2.auc_ranks([q["prob_c"] for q in pp], yy)
                return ad, ac, ad - ac
            ba = RS.pair_bootstrap(coh, sorted(coh), aucs, B, seeds["derm_auc"])
            neg_ids = sorted(k for k, v in coh.items() if v["label"] == 0)
            bp = RS.pair_bootstrap(coh, neg_ids, lambda q: RS.sens_diff(q, 0), B, seeds["derm_spec"])
            spd, spc, spdd = RS.sens_diff(pl, 0)
            corr = DS.correctness_table(pl)
            cm = RS.mcnemar_exact(corr["dermoscopy_only_correct"], corr["clinical_only_correct"])
            blk.update(auroc_derm=dl["auc_a"], auroc_clin=dl["auc_b"], auroc_diff=dl["diff"], auroc_diff_ci=ba["ci"][2],
                       auroc_ci={"derm": ba["ci"][0], "clin": ba["ci"][1]}, delong_p_descriptive=dl.get("p_value"),
                       spec_derm=spd, spec_clin=spc, spec_diff_pp=100 * spdd, spec_diff_ci_pp=[100 * x for x in bp["ci"][2]],
                       correctness=corr, correctness_mcnemar_p_descriptive=cm["p_value"],
                       ece_derm=DS.ece_bins([p["prob_d"] for p in ps], y)["ece"],
                       ece_clin=DS.ece_bins([p["prob_c"] for p in ps], y)["ece"],
                       undefined={"auc": ba["n_undefined"], "spec": bp["n_undefined"], "sens": bs["n_undefined"]})
        res[f"derm7pt_{name}"] = blk
    return res


def phase_analyze(cfg, a):
    from . import provenance as P
    from .desc_4b3 import publish_csv, publish_json
    t0 = time.perf_counter()
    out = os.path.abspath(a.run_dir)
    spec = _load_spec(cfg, out)
    reg = _hashed_json(out, "threshold_registry.json")
    with open(os.path.join(out, "receipt_infer.json"), encoding="utf-8") as f:
        for n, h in json.load(f)["outputs"].items():
            if P.sha256_file(os.path.join(out, n)) != h:
                raise RuntimeError(f"changed {n}")
    seeds = spec["bootstrap"]["seeds"]
    allres, rows = [], []
    for e in spec["models"]:
        t = reg["thresholds"][e["id"]]["threshold"]
        r = analyse_model(cfg, out, e, t, seeds)
        r["reuse_status"] = {c: ("reused" if c in e["reuse"] else "new") for c in COHORTS}
        r["threshold_source"] = reg["thresholds"][e["id"]]["source"]
        allres.append(r)
        h, pd_, d = r["ham_test"], r["pad_all"], r["derm7pt_PRIMARY"]
        rows.append({"model": e["id"], "arch": e["arch"], "seed": e["seed"], "epoch": e["epoch"], "threshold": t,
                     "threshold_source": r["threshold_source"],
                     "ham_sens": h["sensitivity"], "ham_sens_ci": json.dumps(h["ci95"]["sens"]), "ham_spec": h["specificity"],
                     "ham_auroc": h["auroc"], "ham_auroc_ci": json.dumps(h["ci95"]["auroc"]),
                     "derm_sens_derm": d["sens_derm"], "derm_sens_clin": d["sens_clin"], "derm_b": d["b"], "derm_c": d["c"],
                     "derm_sens_diff_pp": d["diff_pp"], "derm_sens_diff_ci": json.dumps(d["diff_ci_pp"]),
                     "derm_mcnemar_p_desc": d["mcnemar_p_descriptive"], "derm_auroc_derm": d["auroc_derm"],
                     "derm_auroc_clin": d["auroc_clin"], "derm_auroc_diff": d["auroc_diff"], "derm_auroc_diff_ci": json.dumps(d["auroc_diff_ci"]),
                     "derm_spec_derm": d["spec_derm"], "derm_spec_clin": d["spec_clin"],
                     "derm_acc_derm": d["correctness"]["accuracy_dermoscopy"], "derm_acc_clin": d["correctness"]["accuracy_clinical"],
                     "ece_derm": d["ece_derm"], "ece_clin": d["ece_clin"],
                     "all_sens_diff_pp": r["derm7pt_ALL"]["diff_pp"],
                     "pad_sens": pd_["sensitivity"], "pad_spec": pd_["specificity"], "pad_auroc": pd_["auroc"],
                     "pad_auroc_ci": json.dumps(pd_["ci95"]["auroc"]), "status": json.dumps(r["reuse_status"])})
        print(e["id"], round(h["auroc"], 3), round(d["diff_pp"], 1), round(pd_["auroc"], 3), flush=True)
    ranges = []
    for arch in ("resnet50", "efficientnet_b3", "densenet121"):
        rr = [x for x in rows if x["arch"] == arch]
        for k in ("ham_sens", "ham_spec", "ham_auroc", "derm_sens_derm", "derm_sens_clin", "derm_sens_diff_pp", "derm_auroc_derm",
                  "derm_auroc_clin", "derm_auroc_diff", "pad_sens", "pad_spec", "pad_auroc", "ece_derm", "ece_clin"):
            v = [x[k] for x in rr]
            ranges.append({"arch": arch, "metric": k, "s0": v[0], "s1": v[1], "s2": v[2], "min": min(v), "max": max(v),
                           "note": "range across 3 seeds; not a CI"})
    publish_json(out, "per_model_results.json", {"models": allres, "runtime_s": time.perf_counter() - t0})
    publish_csv(out, "comparison_table.csv", rows, list(rows[0]))
    publish_csv(out, "seed_ranges.csv", ranges, list(ranges[0]))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("per_model_results.json", "comparison_table.csv",
                                                                           "seed_ranges.csv", "spec.json", "threshold_registry.json")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    print("runtime", round(time.perf_counter() - t0, 1), "verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


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
