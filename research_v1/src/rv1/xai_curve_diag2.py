"""Stage 2D2.1 bounded diagnostic under xai_precision_v1 (SOFTWARE DIAGNOSTIC ONLY).

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_curve_diag2 --config ../configs/xai_curve_diag2_v1.json --out-root ../runs/xai_curve_diag2

Declared BEFORE running (Addendum B):
  ENDPOINT_TOL = 1e-4 (probability)            -- curve endpoint vs direct prediction
  AUC_BATCH_TOL = 1e-3 (engineering tolerance) -- |AUC(eval_batch=k) - AUC(eval_batch=1)|
  IG criteria (protocol 02_rev2 §6.6): failure rate <= 5%, p95 e_rel <= 0.20,
  median |dAUC| (insertion and deletion) vs 2m < 0.005 (metric-convergence threshold).
Same 8 HAM TRAIN images and Stage 2C3 pilot checkpoint as Stages 2D1/2D2. IG internal
batch fixed at 16. Curve evaluation for the m comparison fixed at eval_batch = 1.
"""
import argparse
import json
import os
import time

AUC_BATCH_TOL = 1e-3
EVAL_BATCHES = (1, 11, 33)
IG_IB = 16
M_LIST = (128, 256, 512)
CONV_EB = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from . import config as C, decode as DC, provenance as P, torch_data as TD, trainer as TR
    from . import xai_attr as XA, xai_curves as XC
    from .seeds import seed_everything, stable_seed
    from .smoke_train import build_resnet50
    cfg = C.load(a.config)
    if cfg["purpose"] != "xai_diag":
        raise C.ConfigError("requires purpose 'xai_diag'")
    t_all = time.perf_counter()
    dev = "cuda"
    seed_everything(0, cfg["determinism"])
    ck_path = C.resolve(cfg, cfg["supporting_inputs"]["pilot_checkpoint"])
    with open(os.path.join(os.path.dirname(ck_path), "latest.json"), encoding="utf-8") as f:
        ptr = json.load(f)
    ck = TR._load_verified(ck_path, ptr["sha256"])
    model, _, _, _, _ = build_resnet50(dict(ck["config"], project_root=cfg["project_root"]), dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    root = cfg["project_root"]
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    sel = []
    for lab in ("1", "0"):
        rr = sorted((r for r in rows if r["split"] == "train" and r["label_mel"] == lab),
                    key=lambda r: (stable_seed("xai_diag_v1", r["record_uid"]), r["record_uid"]))
        sel += [r["record_uid"] for r in rr[:4]]
    ds = TD.ManifestDataset(rows, "ham_train", os.path.join(root, "cache"), os.path.join(root, "DATASETS"),
                            sel, mode="eval")
    x = torch.stack([ds[i]["x"] for i in range(len(ds))]).to(dev)
    N = len(sel)
    with XA.xai_precision():
        policy = XA.precision_settings()
    timings, mem = {}, {}

    def timed(name, fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        timings[name] = round(time.perf_counter() - t0, 3)
        mem[name] = [round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                     round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)]
        return out

    base = XA.gaussian_baseline(x)
    cam = timed("gradcam_b8", lambda: XA.grad_cam(model, model.layer4, x))
    ig = {m: timed(f"ig_m{m}_ib{IG_IB}", lambda m=m: XA.integrated_gradients(model, x, base, m, IG_IB))
          for m in M_LIST}

    def curve_all(maps, eb, tag):
        return timed(f"curves_{tag}_eb{eb}", lambda: [XC.curves(model, x[i], base[i], maps[i], eval_batch=eb)
                                                      for i in range(N)])

    # ---------------- A. curve-evaluation batching (reference eval_batch = 1)
    A = {}
    for tag, maps in (("gradcam", cam["map"]), ("ig_m128", ig[128]["primary"])):
        res = {eb: curve_all(maps, eb, tag) for eb in EVAL_BATCHES}
        per = []
        for i in range(N):
            r1 = res[1][i]
            d = {"record_uid": sel[i],
                 "endpoint": {f"eb{eb}": {"max_abs": res[eb][i]["endpoint_prediction_max_abs"],
                                          "status": res[eb][i]["endpoint_status"]} for eb in EVAL_BATCHES}}
            for eb in EVAL_BATCHES[1:]:
                r = res[eb][i]
                d[f"eb{eb}_vs_eb1"] = {
                    "max_point_diff": max(abs(p - q) for k in ("insertion", "deletion")
                                          for p, q in zip(r[k], r1[k])),
                    "d_auc_ins": r["auc_insertion_v1"] - r1["auc_insertion_v1"],
                    "d_auc_del": r["auc_deletion_v1"] - r1["auc_deletion_v1"]}
                d[f"eb{eb}_vs_eb1"]["within_engineering_tol"] = (
                    abs(d[f"eb{eb}_vs_eb1"]["d_auc_ins"]) <= AUC_BATCH_TOL and
                    abs(d[f"eb{eb}_vs_eb1"]["d_auc_del"]) <= AUC_BATCH_TOL)
            per.append(d)
        A[tag] = {"per_image": per,
                  "endpoint_failures": {f"eb{eb}": sum(res[eb][i]["endpoint_status"] != "pass" for i in range(N))
                                        for eb in EVAL_BATCHES},
                  "batch_tolerance_failures": {f"eb{eb}": sum(not p[f"eb{eb}_vs_eb1"]["within_engineering_tol"]
                                                              for p in per) for eb in EVAL_BATCHES[1:]},
                  "max_abs_d_auc": {f"eb{eb}": max(max(abs(p[f"eb{eb}_vs_eb1"]["d_auc_ins"]),
                                                       abs(p[f"eb{eb}_vs_eb1"]["d_auc_del"])) for p in per)
                                    for eb in EVAL_BATCHES[1:]},
                  "max_point_diff": {f"eb{eb}": max(p[f"eb{eb}_vs_eb1"]["max_point_diff"] for p in per)
                                     for eb in EVAL_BATCHES[1:]},
                  "aucs_eb1": {"ins": [r["auc_insertion_v1"] for r in res[1]],
                               "del": [r["auc_deletion_v1"] for r in res[1]]}}
        if tag == "ig_m128":
            cv128 = res[CONV_EB]
    # ---------------- B. IG convergence (curves at eval_batch = 1)
    cv = {128: cv128}
    for m in (256, 512):
        cv[m] = curve_all(ig[m]["primary"], CONV_EB, f"ig_m{m}")
    B = {"m": {}}
    for m in M_LIST:
        comp = ig[m]["completeness"]
        er = [c["e_rel"] for c in comp if c["e_rel"] is not None]
        B["m"][str(m)] = {"completeness": comp, "failures": sum(c["failure"] for c in comp),
                          "failure_rate": sum(c["failure"] for c in comp) / N,
                          "p95_e_rel": float(np.percentile(er, 95)) if er else None,
                          "n_rel_defined": len(er),
                          "eligibility": [XA.eligibility(XA.validity(ig[m]["primary"][i]), comp[i]) for i in range(N)],
                          "endpoint_failures": sum(r["endpoint_status"] != "pass" for r in cv[m]),
                          "auc_ins": [r["auc_insertion_v1"] for r in cv[m]],
                          "auc_del": [r["auc_deletion_v1"] for r in cv[m]]}
    B["candidates"] = {}
    for m in (128, 256):
        d_ins = [cv[m][i]["auc_insertion_v1"] - cv[2 * m][i]["auc_insertion_v1"] for i in range(N)]
        d_del = [cv[m][i]["auc_deletion_v1"] - cv[2 * m][i]["auc_deletion_v1"] for i in range(N)]
        rho = [float(spearmanr(ig[m]["primary"][i].flatten().cpu().numpy(),
                               ig[2 * m]["primary"][i].flatten().cpu().numpy())[0]) for i in range(N)]
        mb = B["m"][str(m)]
        crit = {"a_failure_rate_le_0.05": mb["failure_rate"] <= 0.05,
                "b_p95_e_rel_le_0.20": mb["p95_e_rel"] is not None and mb["p95_e_rel"] <= 0.20,
                "c_median_abs_d_auc_ins_lt_0.005": float(np.median(np.abs(d_ins))) < 0.005,
                "c_median_abs_d_auc_del_lt_0.005": float(np.median(np.abs(d_del))) < 0.005}
        B["candidates"][f"m{m}_vs_m{2 * m}"] = {
            "per_image_d_auc_ins": d_ins, "per_image_d_auc_del": d_del, "per_image_spearman_primary": rho,
            "median_abs_d_auc_ins": float(np.median(np.abs(d_ins))),
            "median_abs_d_auc_del": float(np.median(np.abs(d_del))),
            "criteria": crit, "all_criteria_met_on_this_sample": all(crit.values())}
    B["verdict"] = ("candidate(s) meeting all diagnostic criteria on this sample: " +
                    ", ".join(k for k, v in B["candidates"].items() if v["all_criteria_met_on_this_sample"])
                    if any(v["all_criteria_met_on_this_sample"] for v in B["candidates"].values())
                    else "UNRESOLVED: neither candidate meets all criteria on this sample")
    B["note"] = "m=512 is not called converged: no m=1024 comparison (out of scope). No production m is selected."
    R = {"policy": policy, "declared": {"endpoint_tol": XC.ENDPOINT_TOL, "auc_batch_engineering_tol": AUC_BATCH_TOL,
                                        "ig_internal_batch": IG_IB, "conv_curve_eval_batch": CONV_EB,
                                        "ig_convergence_threshold_c": 0.005},
         "A_curve_eval_batching": A, "B_ig_convergence": B,
         "gradcam_eligibility": [XA.eligibility(XA.validity(cam["map"][i])) for i in range(N)],
         "timings_s": timings, "peak_memory_alloc_reserved_mib": mem, "records": sel,
         "model": {"checkpoint": cfg["supporting_inputs"]["pilot_checkpoint"], "sha256": ptr["sha256"],
                   "note": "pilot checkpoint; software diagnostic only; trained on these images"},
         "total_elapsed_s": round(time.perf_counter() - t_all, 2)}
    receipt = P.build_receipt(cfg, a.config, "completed", details={"policy": policy})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    np.savez_compressed(os.path.join(out, "maps.npz"), record_uids=np.array(sel), x=x.cpu().numpy(),
                        baseline=base.cpu().numpy(), gradcam_map=cam["map"].cpu().numpy(),
                        **{f"ig_attr_m{m}": ig[m]["attr"].cpu().numpy() for m in M_LIST},
                        **{f"ig_primary_m{m}": ig[m]["primary"].cpu().numpy() for m in M_LIST})
    with open(os.path.join(out, "diagnostics.json"), "x", encoding="utf-8") as f:
        json.dump(R, f, indent=1)
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("maps.npz", "diagnostics.json")}
    rp = P.write_receipt(receipt, out)
    print(rp)
    print(json.dumps({"policy": policy,
                      "A": {t: {k: v for k, v in A[t].items() if k not in ("per_image", "aucs_eb1")} for t in A},
                      "B": {"m": {m: {k: v for k, v in b.items() if k in ("failures", "p95_e_rel", "n_rel_defined",
                                                                          "endpoint_failures")}
                                  for m, b in B["m"].items()},
                            "candidates": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("per_image")}
                                           for k, v in B["candidates"].items()},
                            "verdict": B["verdict"]},
                      "timings_s": timings, "total_elapsed_s": R["total_elapsed_s"]}, indent=1))
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


if __name__ == "__main__":
    main()
