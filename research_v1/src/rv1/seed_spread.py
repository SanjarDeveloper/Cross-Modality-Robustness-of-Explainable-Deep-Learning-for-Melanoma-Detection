"""Stage 6B: RQ1 E1.1/E1.2 seed spread for ResNet-50 seeds 0/1/2. Separate processes:
  spec -> threshold (seeds 1, 2 on HAM VAL) -> infer (seeds 1, 2 on Derm7pt ALL) -> analyze.
Seed 0 results are reused from Stage 4A/4B1 (verified by hash, never recomputed or altered).

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.seed_spread --config ../configs/seed_spread_6b_v1.json --phase spec --out-root ../runs/seed_spread_6b
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

SELECTED = {1: ("ckpt_epoch7.pt", 7, "b70ff0e77908d62f56f5643f44d9b29a60ca2b03650bb71766ff7b9a272bc781"),
            2: ("ckpt_epoch8.pt", 8, "fecd74a2a96a2de9c8c4de775150aa30e0f6636469a3b30de1407b553fd25404")}
SEED0_SHA = "303d21e71986dc8f26dec84c6822a301f696060e5f08f613c90fc33c41a003dd"
B = 2000
BATCH = 16
SAME_FILE_TOL = 1e-5
VAL_TOL = 1e-6


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_selected(seed, best, file_sha):
    ck, ep, sha = SELECTED[seed]
    if not (best["checkpoint"] == ck and best["epoch"] == ep and best["sha256"] == sha == file_sha):
        raise RuntimeError(f"seed {seed}: selected checkpoint identity mismatch")


def bind_check(thr, model_sha):
    """A threshold artifact may only be applied to the model it was estimated with."""
    if thr["model_sha256"] != model_sha:
        raise ValueError("threshold bound to a different model")
    return thr["threshold"]


def _read(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _runs(cfg):
    from . import config as C
    return {s: C.resolve(cfg, cfg["supporting_inputs"][f"run_s{s}"].rsplit("/", 1)[0]) for s in (1, 2)}


def _spec(cfg):
    from . import config as C, provenance as P
    si = cfg["supporting_inputs"]
    ins = {k: {"path": v, "sha256": P.sha256_file(C.resolve(cfg, v))} for k, v in si.items()}
    for m in ("ham10000", "derm7pt"):
        ins[f"manifest_{m}"] = {"path": cfg["manifests"][m], "sha256": P.sha256_file(C.resolve(cfg, cfg["manifests"][m]))}
    s0 = cfg["seeds"]["bootstrap_seed"]
    return {"spec_version": "seed_spread_6b_v1", "created_utc": _now(), "inputs": ins,
            "models": {"seed0": {"sha256": SEED0_SHA, "epoch": 8, "role": "PRIMARY (Stage 3A); results reused"},
                       **{f"seed{s}": {"checkpoint": v[0], "epoch": v[1], "sha256": v[2],
                                       "role": "descriptive seed spread"} for s, v in SELECTED.items()}},
            "threshold_rule": "v1_highest_t_img_sens95 (rv1.threshold.estimate), per model, full HAM VAL (val-list order); "
                              "seed-0 threshold unchanged (Stage 4A)",
            "inference": {"prep": "prep_v1_cached256", "precision": "xai_precision_v1", "batch": BATCH,
                          "order": "manifest order within each cache (as Stage 4A)",
                          "val_reproduction_tolerance": VAL_TOL, "same_file_tolerance_prob": SAME_FILE_TOL},
            "cohorts": {"PRIMARY": "in_derm7pt_primary=='1' (1002 pairs, 249 mel)", "ALL": "1011 pairs (sensitivity analysis)"},
            "statistics": {"E1.1": "two-sided exact McNemar on melanoma pairs (rq1_stats.mcnemar_exact)",
                           "E1.2": "sensitivity difference dermoscopy-clinical, pp; percentile pair bootstrap B=2000",
                           "bootstrap_seeds": {"PRIMARY": s0, "ALL": s0 + 2},
                           "bootstrap_note": "identical to the Stage 4B1 seed-0 sensitivity bootstrap seeds (20261001 / +2), "
                                             "so all three models are resampled with the same pair draws",
                           "rng_quantiles": "numpy Generator(PCG64); quantile method linear; undefined counted, not redrawn"},
            "reporting": "seed 0 is the sole primary RQ1 test; seeds 1-2 descriptive spread; no combined p-values, no "
                         "winning seed, no ensembles; min-max range is not a CI"}


def _hashed(out, name, obj=None):
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


def _load_spec(cfg, out):
    from . import config as C, provenance as P
    spec = _hashed(out, "spec.json")
    for k, v in spec["inputs"].items():
        if P.sha256_file(C.resolve(cfg, v["path"])) != v["sha256"]:
            raise RuntimeError(f"input changed: {k}")
    return spec


def _model(cfg, seed):
    import timm
    from . import provenance as P, trainer as TR
    rd = _runs(cfg)[seed]
    best = json.load(open(os.path.join(rd, "best.json"), encoding="utf-8"))
    ck = os.path.join(rd, best["checkpoint"])
    check_selected(seed, best, P.sha256_file(ck))
    st = TR._load_verified(ck, best["sha256"])
    if st["epoch_completed"] != best["epoch"] or st["config"]["seeds"]["train_seed"] != seed:
        raise RuntimeError("checkpoint content mismatch")
    m = timm.create_model("resnet50", pretrained=False, num_classes=1)
    m.load_state_dict(st["model"], strict=True)
    return m.to("cuda").eval(), best


def phase_spec(cfg, a):
    from . import provenance as P
    r = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, r["run_id"])
    os.makedirs(out)
    r["outputs"] = {"spec.json": _hashed(out, "spec.json", _spec(cfg))}
    P.write_receipt(r, out, name="receipt_spec.json")
    print("run dir", out)


def phase_threshold(cfg, a):
    from . import config as C, decode as DC, provenance as P, threshold as TH, trainer as TR
    from .desc_4b3 import publish_csv
    from .predict_v1 import infer
    out = os.path.abspath(a.run_dir)
    _load_spec(cfg, out)
    ham = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    by = {r["record_uid"]: r for r in ham}
    uids = [r["record_uid"] for r in _read(C.resolve(cfg, cfg["supporting_inputs"]["val_list"]))]
    T, outs = {}, {}
    for s in (1, 2):
        t0 = time.perf_counter()
        model, best = _model(cfg, s)
        logits, probs, bi = infer(cfg, model, ham, "ham_val", uids)
        labels = [int(by[u]["label_mel"]) for u in uids]
        saved = json.load(open(glob.glob(os.path.join(_runs(cfg)[s], f"val_epoch{best['epoch']}_*.json"))[0], encoding="utf-8"))
        d = max(abs(x - y) for x, y in zip(saved["probs"], probs))
        repro = {"record_uids_equal": saved["record_uids"] == uids, "labels_equal": saved["labels"] == labels,
                 "max_abs_prob_diff": d, "tolerance": VAL_TOL}
        repro["pass"] = repro["record_uids_equal"] and repro["labels_equal"] and d <= VAL_TOL
        if not repro["pass"]:
            raise RuntimeError(f"seed {s}: HAM VAL reproduction failed {repro}")
        est = TH.estimate(probs, labels)
        thr = dict(est, created_utc=_now(), model_sha256=best["sha256"], seed=s, checkpoint=best["checkpoint"],
                   dataset="HAM10000 VAL (1529)", batching=bi, val_reproduction=repro, auroc_ham_val=TR.auroc(labels, probs),
                   frozen_before_derm7pt_inference=True)
        outs[f"threshold_s{s}.json"] = _hashed(out, f"threshold_s{s}.json", thr)
        publish_csv(out, f"ham_val_pred_s{s}.csv",
                    [{"record_uid": u, "label_mel": y, "logit": repr(lg), "prob": repr(p), "checkpoint_sha256": best["sha256"]}
                     for u, y, lg, p in zip(uids, labels, logits, probs)],
                    ["record_uid", "label_mel", "logit", "prob", "checkpoint_sha256"])
        T[s] = time.perf_counter() - t0
        print(s, {k: thr[k] for k in ("status", "threshold", "TP", "FN", "TN", "FP", "sensitivity", "specificity")}, repro)
    r = P.build_receipt(cfg, a.config, "completed", details={"phase": "threshold", "seconds": T})
    r["outputs"] = dict(outs, **{f"ham_val_pred_s{s}.csv": P.sha256_file(os.path.join(out, f"ham_val_pred_s{s}.csv")) for s in (1, 2)})
    P.write_receipt(r, out, name="receipt_threshold.json")


def phase_infer(cfg, a):
    import numpy as np
    from . import config as C, decode as DC, provenance as P, threshold as TH
    from .desc_4b3 import publish_csv, publish_json
    from .predict_v1 import infer
    out = os.path.abspath(a.run_dir)
    _load_spec(cfg, out)
    derm = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["derm7pt"]), "derm7pt")[0]
    started = _now()
    T, ver = {}, {}
    for s in (1, 2):
        thr = _hashed(out, f"threshold_s{s}.json")
        if not thr["created_utc"] < started:
            raise RuntimeError("threshold not frozen first")
        th_sha = P.sha256_file(os.path.join(out, f"threshold_s{s}.json"))
        t0 = time.perf_counter()
        model, best = _model(cfg, s)
        t = bind_check(thr, best["sha256"])
        rows_out = []
        for cache in ("derm7pt_dermoscopy", "derm7pt_clinical"):
            rr = [r for r in derm if r["cache_name"] == cache]
            logits, probs, bi = infer(cfg, model, derm, cache, [r["record_uid"] for r in rr])
            for r, lg, p in zip(rr, logits, probs):
                if not (math.isfinite(lg) and 0.0 <= p <= 1.0):
                    raise RuntimeError(f"invalid score {r['record_uid']}")
                rows_out.append({"record_uid": r["record_uid"], "image_uid": r["image_uid"], "pair_id": r["pair_id"],
                                 "cluster_id": r["cluster_id"], "dataset": r["dataset"], "modality": r["modality"],
                                 "label_mel": r["label_mel"], "in_derm7pt_all": r["in_derm7pt_all"],
                                 "in_derm7pt_primary": r["in_derm7pt_primary"], "same_file_pair": r["same_file_pair"],
                                 "cohort_flags": r["cohort_flags"], "logit": repr(lg), "prob": repr(p),
                                 "threshold": repr(t), "pred": TH.predict(p, t), "status": "ok",
                                 "checkpoint_sha256": best["sha256"], "threshold_sha256": th_sha,
                                 "prep_version": "prep_v1_cached256", "precision_policy": "xai_precision_v1",
                                 "batch": BATCH, "cache_name": cache, "seed": s})
        if sorted(r["record_uid"] for r in rows_out) != sorted(r["record_uid"] for r in derm) or \
                len({r["record_uid"] for r in rows_out}) != len(derm):
            raise RuntimeError("coverage")
        pairs = {}
        for r in rows_out:
            pairs.setdefault(r["pair_id"], {})[r["modality"]] = r
        sf = []
        for pid, d in pairs.items():
            if d["dermoscopy"]["same_file_pair"] == "1":
                dp = abs(float(d["dermoscopy"]["prob"]) - float(d["clinical"]["prob"]))
                sf.append({"pair_id": pid, "abs_prob_diff": dp, "pred_equal": d["dermoscopy"]["pred"] == d["clinical"]["pred"],
                           "pass": dp <= SAME_FILE_TOL and d["dermoscopy"]["pred"] == d["clinical"]["pred"]})
        ver[s] = {"n_rows": len(rows_out), "n_pairs": len(pairs), "same_file": sf, "same_file_all_pass": all(x["pass"] for x in sf)}
        if not ver[s]["same_file_all_pass"]:
            raise RuntimeError(f"same-file check failed seed {s}")
        publish_csv(out, f"derm7pt_pred_s{s}.csv", rows_out, list(rows_out[0]))
        T[s] = time.perf_counter() - t0
    publish_json(out, "infer_verification.json", {"started_utc": started, "seconds": T, "checks": ver})
    r = P.build_receipt(cfg, a.config, "completed", details={"phase": "infer"})
    r["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("derm7pt_pred_s1.csv", "derm7pt_pred_s2.csv",
                                                                    "infer_verification.json")}
    P.write_receipt(r, out, name="receipt_infer.json")
    print(T, {s: (v["n_rows"], v["same_file_all_pass"]) for s, v in ver.items()})


def per_seed(rows, t, th_sha, ck_sha, seeds_boot):
    """E1.1/E1.2 for one model from its persisted rows (PRIMARY + ALL)."""
    from . import rq1_stats as RS
    pairs = RS.build_pairs(rows, t, th_sha, ck_sha)
    res, reps = {}, {}
    for name, coh, bseed in (("PRIMARY", {k: v for k, v in pairs.items() if v["primary"]}, seeds_boot["PRIMARY"]),
                             ("ALL", pairs, seeds_boot["ALL"])):
        plist = list(coh.values())
        mel = RS.paired_table(plist, 1)
        mc = RS.mcnemar_exact(mel["first_only"], mel["second_only"])
        sd, sc, dd = RS.sens_diff(plist, 1)
        ids = sorted(k for k, v in coh.items() if v["label"] == 1)
        bs = RS.pair_bootstrap(coh, ids, lambda ps: RS.sens_diff(ps, 1), B, bseed)
        reps[name] = bs.pop("replicates")
        n = sum(mel.values())
        res[name] = {"n_pairs": len(coh), "n_mel_pairs": n,
                     "table": {"both_detected": mel["both"], "dermoscopy_only_b": mel["first_only"],
                               "clinical_only_c": mel["second_only"], "both_missed": mel["neither"]},
                     "sens_dermoscopy": sd, "sens_clinical": sc,
                     "num_den": {"dermoscopy": [mel["both"] + mel["first_only"], n], "clinical": [mel["both"] + mel["second_only"], n]},
                     "diff_pp": 100 * dd, "ci95_diff_pp": [100 * x for x in bs["ci"][2]],
                     "mcnemar_p": mc["p_value"], "mcnemar_p_sci": mc.get("p_value_sci", "1"), "log10_p": mc.get("log10_p"),
                     "bootstrap": bs}
    return res, reps


def phase_analyze(cfg, a):
    import io
    import numpy as np
    from . import config as C, provenance as P
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    t_start = time.perf_counter()
    out = os.path.abspath(a.run_dir)
    spec = _load_spec(cfg, out)
    with open(os.path.join(out, "receipt_infer.json"), encoding="utf-8") as f:
        for n, h in json.load(f)["outputs"].items():
            if P.sha256_file(os.path.join(out, n)) != h:
                raise RuntimeError(f"changed {n}")
    si = cfg["supporting_inputs"]
    sb = spec["statistics"]["bootstrap_seeds"]
    results, reps_all = {}, {}
    # seed 0: reuse Stage 4B1 (verified), and recompute from Stage 4A rows only to cross-check
    r4b1 = json.load(open(C.resolve(cfg, si["seed0_results_4b1"]), encoding="utf-8"))
    thr0 = json.load(open(C.resolve(cfg, si["seed0_threshold"]), encoding="utf-8"))
    rows0 = _read(C.resolve(cfg, si["seed0_pred_derm7pt"]))
    chk0, _ = per_seed(rows0, thr0["threshold"], spec["inputs"]["seed0_threshold"]["sha256"], SEED0_SHA, sb)
    pub = r4b1["PRIMARY"]
    agree = (pub["melanoma_paired_table"]["dermoscopy_only_b"], pub["melanoma_paired_table"]["clinical_only_c"],
             pub["sensitivity"]["ci95_pp"]["diff"]) == (chk0["PRIMARY"]["table"]["dermoscopy_only_b"],
                                                       chk0["PRIMARY"]["table"]["clinical_only_c"], chk0["PRIMARY"]["ci95_diff_pp"])
    if not agree:
        raise RuntimeError("seed-0 recomputation disagrees with the published Stage 4B1 result")
    results["seed0"] = {"source": "Stage 4B1 results.json (published, unchanged); recomputation agrees exactly",
                        "epoch": 8, "threshold": thr0["threshold"], **{k: {kk: vv for kk, vv in v.items() if kk != "bootstrap"}
                                                                          for k, v in chk0.items()}}
    for s in (1, 2):
        thr = _hashed(out, f"threshold_s{s}.json")
        th_sha = P.sha256_file(os.path.join(out, f"threshold_s{s}.json"))
        rows = _read(os.path.join(out, f"derm7pt_pred_s{s}.csv"))
        t = bind_check(thr, SELECTED[s][2])
        res, reps = per_seed(rows, t, th_sha, SELECTED[s][2], sb)
        reps_all.update({f"s{s}_{k}": v for k, v in reps.items()})
        results[f"seed{s}"] = {"epoch": SELECTED[s][1], "threshold": t, **res}
    membership = {}
    for s, rows in ((0, rows0), (1, _read(os.path.join(out, "derm7pt_pred_s1.csv"))), (2, _read(os.path.join(out, "derm7pt_pred_s2.csv")))):
        membership[s] = sorted((r["record_uid"], r["pair_id"], r["label_mel"], r["in_derm7pt_primary"]) for r in rows)
    results["identical_cohort_membership"] = membership[0] == membership[1] == membership[2]
    if not results["identical_cohort_membership"]:
        raise RuntimeError("cohort membership differs across seeds")
    table = []
    for name in ("PRIMARY", "ALL"):
        for s in (0, 1, 2):
            x = results[f"seed{s}"][name]
            table.append({"cohort": name, "seed": s, "role": "PRIMARY RQ1 test" if (s == 0 and name == "PRIMARY") else
                          ("sensitivity analysis" if name == "ALL" else "descriptive seed spread"),
                          "epoch": results[f"seed{s}"]["epoch"], "threshold": results[f"seed{s}"]["threshold"],
                          "n_mel_pairs": x["n_mel_pairs"],
                          "sens_dermoscopy": f"{x['num_den']['dermoscopy'][0]}/{x['n_mel_pairs']}",
                          "sens_clinical": f"{x['num_den']['clinical'][0]}/{x['n_mel_pairs']}",
                          "diff_pp": x["diff_pp"], "ci_lo": x["ci95_diff_pp"][0], "ci_hi": x["ci95_diff_pp"][1],
                          "b": x["table"]["dermoscopy_only_b"], "c": x["table"]["clinical_only_c"],
                          "mcnemar_p": x["mcnemar_p"]})
    rng = {n: {"min_diff_pp": min(r["diff_pp"] for r in table if r["cohort"] == n),
               "max_diff_pp": max(r["diff_pp"] for r in table if r["cohort"] == n),
               "all_same_direction": len({r["diff_pp"] > 0 for r in table if r["cohort"] == n}) == 1,
               "note": "range across 3 runs; NOT a confidence interval"} for n in ("PRIMARY", "ALL")}
    results["seed_spread_range"] = rng
    results["analysis_runtime_s"] = time.perf_counter() - t_start
    publish_json(out, "results.json", results)
    publish_csv(out, "seed_spread_table.csv", table, list(table[0]))
    buf = io.BytesIO()
    np.savez_compressed(buf, **reps_all, columns=np.array(["dermoscopy", "clinical", "diff"]))
    publish_bytes(out, "bootstrap_replicates_s1_s2.npz", buf.getvalue())
    r = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze"})
    r["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("results.json", "seed_spread_table.csv",
                                                                    "bootstrap_replicates_s1_s2.npz", "spec.json")}
    rp = P.write_receipt(r, out, name="receipt_analyze.json")
    for x in table:
        print({k: (round(v, 4) if isinstance(v, float) and k != "mcnemar_p" else v) for k, v in x.items()})
    print(rng, "\nverify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


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
