"""Stage 3B.1 sequential IG extension (Addendum E): candidate m=512 vs m=1024 on the frozen
Stage 3B pilot. Reuses verified Stage 3B outputs; never writes into the Stage 3B run.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_pilot_ext --config ../configs/xai_pilot_stage3b1_v1.json --out-root ../runs/xai_pilot_ext
"""
import argparse
import csv
import json
import os
import time

from .xai_pilot import EXPECTED_SHA, IG_IB, _read_csv, _sidecar_ok

M_CAND, M_REF = 512, 1024
EVAL_BATCH = 11
TOL = 1e-6
N_FULL_REFS = 996


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    t_all = time.perf_counter()
    import numpy as np
    import timm
    import torch
    from . import config as C, decode as DC, provenance as P, torch_data as TD, trainer as TR
    from . import xai_attr as XA, xai_controls as XK, xai_curves as XC, xai_pairs as XP
    from .seeds import seed_everything
    cfg = C.load(a.config)
    si = cfg["supporting_inputs"]
    dev = "cuda"
    seed_everything(0, cfg["determinism"])
    T, M = {}, {}

    def timed(name, fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        T[name] = round(time.perf_counter() - t0, 3)
        M[name] = [round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                   round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)]
        print(f"{name}: {T[name]}s peak {M[name]}", flush=True)
        return r

    # ---------------- model
    ck_path = C.resolve(cfg, si["model_checkpoint"])
    with open(C.resolve(cfg, si["model_best_pointer"]), encoding="utf-8") as f:
        best = json.load(f)
    fsha = P.sha256_file(ck_path)
    if not (fsha == best["sha256"] == EXPECTED_SHA and _sidecar_ok(ck_path)):
        raise RuntimeError("checkpoint verification failed")
    ck = TR._load_verified(ck_path, EXPECTED_SHA)
    model = timm.create_model("resnet50", pretrained=False, num_classes=1)
    model.load_state_dict(ck["model"], strict=True)
    model = model.to(dev).eval()
    del ck
    # ---------------- Stage 3B reuse checks
    rp3 = C.resolve(cfg, si["stage3b_receipt"])
    d3 = os.path.dirname(rp3)
    probs = P.verify_receipt(rp3, cfg["project_root"])
    with open(rp3, encoding="utf-8") as f:
        rec3 = json.load(f)
    bad = [n for n, h in rec3["outputs"].items() if P.sha256_file(os.path.join(d3, n)) != h]
    if probs or bad or rec3["status"] != "completed":
        raise RuntimeError(f"Stage 3B verification failed: {probs} {bad}")
    with open(os.path.join(d3, "summary.json"), encoding="utf-8") as f:
        s3 = json.load(f)
    with XA.xai_precision():
        policy = XA.precision_settings()
    compat = {"policy_equal": s3["policy"] == policy,
              "settings": s3["settings"], "eval_batch_used": s3["batch_equivalence"]["eval_batch_used"]}
    if not (compat["policy_equal"] and s3["settings"]["ig_internal_batch"] == IG_IB and
            s3["batch_equivalence"]["eval_batch_used"] == EVAL_BATCH and s3["settings"]["S"] == 32 and
            s3["settings"]["prep"] == "prep_v1_cached256" and s3["stage_status"] == "completed"):
        raise RuntimeError(f"settings incompatible: {compat}")
    refs_f = _read_csv(C.resolve(cfg, si["pilot_refs"]))
    pairs_f = _read_csv(C.resolve(cfg, si["pilot_pairs"]))
    for k in ("pilot_pairs", "pilot_refs"):
        if not _sidecar_ok(C.resolve(cfg, si[k])):
            raise RuntimeError("pilot list sidecar mismatch")
    refs = [r["ref_key"] for r in refs_f]
    z = np.load(os.path.join(d3, "maps.npz"))
    if list(z["ref_keys"]) != refs:
        raise RuntimeError("Stage 3B ref order differs")
    with open(os.path.join(d3, "ig_convergence.json"), encoding="utf-8") as f:
        conv3 = json.load(f)
    with open(os.path.join(d3, "curves.json"), encoding="utf-8") as f:
        cv3 = json.load(f)
    with open(os.path.join(d3, "mismatch_assignment.json"), encoding="utf-8") as f:
        assign3 = json.load(f)["assignment"]
    rec512 = conv3["per_ref"][str(M_CAND)]
    # ---------------- inputs
    derm_rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["derm7pt"]), "derm7pt")[0]
    root = cfg["project_root"]
    xs = []
    for r in refs_f:
        ds = TD.ManifestDataset(derm_rows, r["cache_name"], os.path.join(root, "cache"),
                                os.path.join(root, "DATASETS"), [r["record_uid"]], mode="eval")
        xs.append(ds[0]["x"])
    x = torch.stack(xs).to(dev)
    N = len(refs)
    R = {r["ref_key"]: r for r in refs_f}
    pairs = [{"pair_id": p["pair_id"], "label_mel": p["label_mel"], "record_uid_dermoscopy": p["record_uid_dermoscopy"],
              "record_uid_clinical": p["record_uid_clinical"]} for p in pairs_f]
    receipt = P.build_receipt(cfg, a.config, "completed",
                              details={"policy": policy, "stage": "3B.1", "stage3b_run": os.path.basename(d3),
                                       "model_sha256": fsha})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)

    def dump(name, obj):
        with open(os.path.join(out, name), "x", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=float)

    base = XA.gaussian_baseline(x)
    lb = z["baseline_logit"]
    # ---------------- compatibility recheck of m=512 on the frozen subset
    sub = [i for i, k in enumerate(refs) if any(p["pair_id"] == R[k]["pair_id"] and p["batch_check_subset"] == "1"
                                                for p in pairs_f)]

    def recheck():
        rows, ok = [], True
        ig = XA.integrated_gradients(model, x[sub], base[sub], M_CAND, IG_IB)
        for j, i in enumerate(sub):
            k = refs[i]
            old_map = torch.from_numpy(z[f"ig{M_CAND}_primary"][i])
            new_map = ig["primary"][j].cpu()
            c = ig["completeness"][j]
            o = rec512[k]
            d_map = float((new_map - old_map).abs().max())
            el = XA.eligibility(XA.validity(ig["primary"][j]), c)
            r = {"ref": k, "max_abs_map_diff": d_map, "map_scale": float(old_map.abs().max()),
                 "failure_old": o["completeness_failure"], "failure_new": c["failure"],
                 "d_e_rel": abs(c["e_rel"] - o["e_rel"]) if c["e_rel"] is not None and o["e_rel"] is not None else None}
            if el["eligible_for_primary_analysis"]:
                cv = XC.analysis_curves(model, x[i], base[i], ig["primary"][j], el, "ig", eval_batch=EVAL_BATCH)
                r.update(d_ins=abs(cv["auc_insertion_v1"] - o["auc_ins"]), d_del=abs(cv["auc_deletion_v1"] - o["auc_del"]),
                         endpoint_new=cv["endpoint_status"], endpoint_old=o["endpoint_status"])
            good = (r["failure_old"] == r["failure_new"] and (r["d_e_rel"] is None or r["d_e_rel"] <= TOL) and
                    d_map <= TOL * max(r["map_scale"], 1e-30) and
                    ("d_ins" not in r or (r["d_ins"] <= TOL and r["d_del"] <= TOL and r["endpoint_new"] == r["endpoint_old"])))
            if el["eligible_for_primary_analysis"] != o["eligible"]:
                good = False
            r["pass"] = good
            ok &= good
            rows.append(r)
        return {"pass": bool(ok), "tolerance": TOL, "rows": rows}
    rc = timed("recheck_m512_subset", recheck)
    dump("compat_recheck_m512.json", rc)
    if not rc["pass"]:
        dump("summary.json", {"stage_status": "stopped", "reason": "m=512 recheck discrepancy", "timings_s": T,
                              "peak_memory_alloc_reserved_mib": M})
        receipt["status"] = "failed"
        receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))}
        P.write_receipt(receipt, out)
        print("STOPPED: recheck discrepancy")
        return
    # ---------------- m=1024 on all 48
    ig1 = timed("attr_ig_m1024", lambda: XA.integrated_gradients(model, x, base, M_REF, IG_IB))
    el1 = [XA.eligibility(XA.validity(ig1["primary"][i]), ig1["completeness"][i]) for i in range(N)]
    ex1 = [None] * N

    def curves1024():
        for i in range(N):
            try:
                cv = XC.analysis_curves(model, x[i], base[i], ig1["primary"][i], el1[i], "ig", eval_batch=EVAL_BATCH)
                ex1[i] = cv
            except XC.IneligibleMap as e:
                ex1[i] = {"analysis_status": "ineligible", "reason": str(e)}
    timed("curves_ig1024", curves1024)
    rec1024 = {}
    for i, k in enumerate(refs):
        c, cv = ig1["completeness"][i], ex1[i]
        prim = cv["analysis_status"] == "primary"
        rec1024[k] = {"completeness_failure": bool(c["failure"]), "e_rel": c["e_rel"], "e_abs": c["e_abs"],
                      "eligible": el1[i]["eligible_for_primary_analysis"], "reasons": el1[i]["ineligibility_reasons"],
                      "map_flags": el1[i]["map_flags"], "curve_status": cv.get("curve_status"),
                      "endpoint_status": cv.get("endpoint_status"),
                      "auc_ins": cv.get("auc_insertion_v1") if prim else None,
                      "auc_del": cv.get("auc_deletion_v1") if prim else None}
    # ---------------- assessment (unchanged rule)
    groups = {"pooled": refs}
    for mod in ("dermoscopy", "clinical"):
        groups[f"modality={mod}"] = [k for k in refs if R[k]["modality"] == mod]
    for lab in ("1", "0"):
        groups[f"label={lab}"] = [k for k in refs if R[k]["label_mel"] == lab]
        for mod in ("dermoscopy", "clinical"):
            groups[f"label={lab},modality={mod}"] = [k for k in refs if R[k]["label_mel"] == lab and R[k]["modality"] == mod]
    assess = {g: XP.assess_candidate({k: rec512[k] for k in ks}, {k: rec1024[k] for k in ks}, ks) for g, ks in groups.items()}
    chosen = XP.choose_m({M_CAND: assess["pooled"]}, candidates=(M_CAND,))
    per_image = []
    for k in refs:
        a5, a1 = rec512[k], rec1024[k]
        both = XP.curve_usable(a5) and XP.curve_usable(a1)
        per_image.append({"ref_key": k, "pair_id": R[k]["pair_id"], "label_mel": R[k]["label_mel"],
                          "modality": R[k]["modality"],
                          "fail_512": a5["completeness_failure"], "e_rel_512": a5["e_rel"],
                          "fail_1024": a1["completeness_failure"], "e_rel_1024": a1["e_rel"],
                          "usable_512": XP.curve_usable(a5), "usable_1024": XP.curve_usable(a1),
                          "auc_ins_512": a5["auc_ins"], "auc_ins_1024": a1["auc_ins"],
                          "auc_del_512": a5["auc_del"], "auc_del_1024": a1["auc_del"],
                          "d_ins": a1["auc_ins"] - a5["auc_ins"] if both else None,
                          "d_del": a1["auc_del"] - a5["auc_del"] if both else None,
                          "comparison": "both_usable" if both else "missing:" + ";".join(
                              assess["pooled"]["c"]["not_eligible_both"].get(k, []))})
    decision = {"candidate": M_CAND, "reference": M_REF, "pooled_overall": assess["pooled"]["overall"],
                "chosen_m": chosen, "status": "selected" if chosen else "UNRESOLVED",
                "scope": "seed-0 ResNet-50 (Stage 3A epoch 8), unperturbed Derm7pt, current pipeline only",
                "note": "m=1024 is not called converged (no 2048 comparison)"}
    dump("ig_assessment_512_vs_1024.json", {"rule": "Addendum D5 / E3 (unchanged)", "decision": decision,
                                            "assessments": assess, "per_ref_1024": rec1024})
    with open(os.path.join(out, "per_image_512_vs_1024.csv"), "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(per_image)
    print("decision:", decision, flush=True)
    # ---------------- Grad-CAM zero-map counts (from Stage 3B, unchanged rule)
    gz = []
    for k in refs:
        e = cv3[k]["explainers"]["gradcam"]
        gz.append({"ref": k, "modality": R[k]["modality"], "label": R[k]["label_mel"], "status": e["status"],
                   "reason": e.get("reason")})
    gcounts = {}
    for g in ("modality", "label"):
        for v in sorted({r[g] for r in gz}):
            rr = [r for r in gz if r[g] == v]
            gcounts[f"{g}={v}"] = {"n": len(rr), "ineligible_all_zero": sum(r["status"] != "primary" for r in rr)}
    # ---------------- controls / pair table at m=512 (only if selected)
    ig_m = chosen
    expl_status, ctrl_status, score_rows = {}, {}, []
    explainers = ["gradcam"] + ([f"ig{ig_m}"] if ig_m else [])
    if ig_m:
        mm = XK.mismatch_assignment([{"record_uid": R[k]["record_uid"], "case_id": R[k]["pair_id"],
                                      "label": R[k]["label_mel"], "modality": R[k]["modality"],
                                      "explainer": f"ig{ig_m}", "condition": "none"} for k in refs])
        for k in refs:
            u, mod = R[k]["record_uid"], R[k]["modality"]
            new = mm[XK.assignment_key(u, mod, f"ig{ig_m}", "none")]["donor"]
            old = assign3[XK.assignment_key(u, mod, "ig256", "none")]["donor"]
            if new != old:
                raise RuntimeError(f"planned donor changed for {k}")
        dump("mismatch_assignment_ig512.json", {"written_before_mismatch_curves": True,
                                                "same_donors_as_stage3b_ig256": True, "assignment": mm})
        maps512 = torch.from_numpy(z[f"ig{ig_m}_primary"])
        uid_i = {R[k]["record_uid"]: i for i, k in enumerate(refs)}
        mres = {}

        def mismatch_curves():
            for i, k in enumerate(refs):
                mod = R[k]["modality"]
                donor_maps = {R[kk]["record_uid"]: maps512[j] for j, kk in enumerate(refs) if R[kk]["modality"] == mod}
                donor_ok = {R[kk]["record_uid"]: XP.curve_usable(rec512[kk]) for kk in refs if R[kk]["modality"] == mod}
                dm, st = XK.resolve_mismatch(XK.assignment_key(R[k]["record_uid"], mod, f"ig{ig_m}", "none"), mm,
                                             donor_maps, donor_ok)
                if dm is None:
                    mres[k] = dict(st)
                    continue
                c = XC.curves(model, x[i], base[i], dm.to(dev), eval_batch=EVAL_BATCH)
                good = c["curve_status"] == "ok" and c["endpoint_status"] == "pass"
                mres[k] = dict(st, status="available" if good else "unavailable",
                               reason=None if good else "curve invalid or endpoint failure",
                               auc_ins=c["auc_insertion_v1"], auc_del=c["auc_deletion_v1"],
                               endpoint_status=c["endpoint_status"])
        timed("curves_mismatch_ig512", mismatch_curves)
    for k in refs:
        u = R[k]["record_uid"]
        for e in explainers:
            ex = cv3[k]["explainers"][e]
            expl = {"auc_ins": ex["auc_ins"], "auc_del": ex["auc_del"]}
            usable = ex["status"] == "primary"
            expl_status[(u, e)] = {"usable": usable, "reasons": [] if usable else
                                   (rec512[k]["reasons"] if e != "gradcam" else [ex.get("reason") or ex["status"]])}
            ctl = {"random": cv3[k]["controls"]["random"], "centre": cv3[k]["controls"]["centre"],
                   "sobel": cv3[k]["controls"]["sobel"],
                   "mismatch": cv3[k]["controls"]["mismatch_gradcam"] if e == "gradcam" else mres[k]}
            for cn, cd in ctl.items():
                sc = XK.control_scores(expl, cd)
                ctrl_status[(u, e, cn)] = {"available": sc["status"] == "available", "reason": sc.get("reason")}
                score_rows.append({"ref_key": k, "pair_id": R[k]["pair_id"], "label_mel": R[k]["label_mel"],
                                   "modality": R[k]["modality"], "explainer": e, "control": cn,
                                   "scores_role": "production_setting_pilot" if e != "gradcam" else "pilot",
                                   "control_status": sc["status"], "reason": sc.get("reason"),
                                   "insertion_advantage": sc["insertion_advantage"],
                                   "deletion_advantage": sc["deletion_advantage"]})
    ptab = XP.pair_table(pairs, expl_status, ctrl_status, explainers, ["random", "centre", "sobel", "mismatch"])
    psum = {}
    for e in explainers:
        for c in ("explainer", "random", "centre", "sobel", "mismatch"):
            rr = [r for r in ptab if r["explainer"] == e and r["contrast"] == c]
            n = sum(r["complete_pair"] for r in rr)
            psum[f"{e}|{c}"] = {"complete_pairs": n, "of": len(rr),
                                "by_label": {lab: sum(r["complete_pair"] for r in rr if r["label_mel"] == lab)
                                             for lab in ("1", "0")},
                                "not_estimable_cells": [lab for lab in ("1", "0") if
                                                        sum(r["complete_pair"] for r in rr if r["label_mel"] == lab) == 0],
                                "exclusions": {r["pair_id"]: r["reasons"] for r in rr if not r["complete_pair"]}}
    with open(os.path.join(out, "pair_eligibility_v2.csv"), "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "label_mel", "explainer", "contrast", "complete_pair", "reasons"],
                           lineterminator="\n")
        w.writeheader()
        for r in ptab:
            w.writerow(dict(r, reasons=";".join(r["reasons"])))
    with open(os.path.join(out, "scores_v2.csv"), "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(score_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(score_rows)
    # ---------------- persistence + projection
    def persist():
        np.savez_compressed(os.path.join(out, "maps_ig1024.npz"), ref_keys=np.array(refs),
                            ig1024_attr=ig1["attr"].cpu().numpy(), ig1024_primary=ig1["primary"].cpu().numpy())
        dump("curves_ig1024.json", {k: {kk: ex1[i].get(kk) for kk in ("analysis_status", "reason", "insertion", "deletion",
                                                                     "fractions", "auc_insertion_v1", "auc_deletion_v1",
                                                                     "endpoint_status", "endpoint_prediction_max_abs",
                                                                     "curve_status", "eval_batch")}
                                    for i, k in enumerate(refs)})
        if ig_m:
            dump("mismatch_ig512_results.json", mres)
    timed("persistence", persist)
    t3 = s3["timings_s"]
    t_curve = s3["projection"]["per_ref_s"]["curves_per_map_pair_ins_del"]
    per_ref = {"baseline": t3["baseline"] / N, "gradcam": t3["attr_gradcam"] / N, "ig_m512": t3["attr_ig_m512"] / N,
               "control_generation": t3["control_generation"] / N, "curve_pair_per_map": t_curve,
               "persistence": t3["persistence"] / N}
    s_ref = sum(v for kk, v in per_ref.items() if kk != "curve_pair_per_map") + 11 * t_curve
    projection = {"label": "PROJECTION (not a measurement) from Stage 3B/3B.1 pilot timings; same GPU, seed-0 ResNet-50, "
                           "unperturbed Derm7pt, eval_batch 11, IG m=512, 11 curve pairs/ref, single process",
                  "per_ref_s": per_ref, "projected_s_per_ref": s_ref, "n_refs": N_FULL_REFS,
                  "projected_total_h": s_ref * N_FULL_REFS / 3600,
                  "measured_gpu_peak_mib_ig512_stage3b": s3["peak_memory_alloc_reserved_mib"]["attr_ig_m512"],
                  "measured_gpu_peak_mib_ig1024_stage3b1": M["attr_ig_m1024"],
                  "projected_disk": {"float32_uncompressed_maps_GB": 1204224 * N_FULL_REFS / 1e9,
                                     "curves_json_GB": s3["projection"]["storage"]["curves_json_total_GB_upper"],
                                     "note": "disk storage, NOT GPU memory"}}
    summary = {"stage_status": "completed", "decision": decision, "compat_recheck_pass": rc["pass"],
               "gradcam_zero_counts": gcounts, "pair_summary": psum, "timings_s": T,
               "peak_memory_alloc_reserved_mib": M, "projection": projection,
               "incremental_measured_s": round(time.perf_counter() - t_all, 2)}
    dump("summary.json", summary)
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))}
    rp = P.write_receipt(receipt, out)
    print(json.dumps({k: summary[k] for k in ("decision", "gradcam_zero_counts", "timings_s",
                                              "peak_memory_alloc_reserved_mib", "projection")}, indent=1, default=float))
    for kk, v in psum.items():
        print(kk, v["complete_pairs"], "/", v["of"], v["by_label"], "NOT ESTIMABLE:" if v["not_estimable_cells"] else "",
              v["not_estimable_cells"] or "")
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


if __name__ == "__main__":
    main()
