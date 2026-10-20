"""Stage 3B.2 (Addendum F): fresh IG m=512 and m=1024 on all 48 frozen refs with ONE outer batch
(48, frozen order) for baseline, completeness endpoint logits and both step counts. Candidate 512
vs 1024 under the unchanged rule. m-independent Stage 3B outputs are reused only after exact
re-verification. Never writes into earlier runs.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_pilot_fresh --config ../configs/xai_pilot_stage3b2_v1.json --out-root ../runs/xai_pilot_fresh
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
BOUND_S = 30 * 60


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
        if time.perf_counter() - t_all > BOUND_S:
            raise RuntimeError(f"30-minute bound exceeded before {name}")
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
    # ---------------- Stage 3B (for m-independent reuse) verification
    rp3 = C.resolve(cfg, si["stage3b_receipt"])
    d3 = os.path.dirname(rp3)
    with open(rp3, encoding="utf-8") as f:
        rec3 = json.load(f)
    bad = [n for n, h in rec3["outputs"].items() if P.sha256_file(os.path.join(d3, n)) != h]
    probs = P.verify_receipt(rp3, cfg["project_root"])
    if probs or bad:
        raise RuntimeError(f"Stage 3B verification failed {probs} {bad}")
    with open(os.path.join(d3, "summary.json"), encoding="utf-8") as f:
        s3 = json.load(f)
    with XA.xai_precision():
        policy = XA.precision_settings()
    if s3["policy"] != policy or s3["batch_equivalence"]["eval_batch_used"] != EVAL_BATCH:
        raise RuntimeError("Stage 3B settings incompatible")
    for k in ("pilot_pairs", "pilot_refs"):
        if not _sidecar_ok(C.resolve(cfg, si[k])):
            raise RuntimeError("pilot list sidecar mismatch")
    refs_f = _read_csv(C.resolve(cfg, si["pilot_refs"]))
    pairs_f = _read_csv(C.resolve(cfg, si["pilot_pairs"]))
    refs = [r["ref_key"] for r in refs_f]
    z = np.load(os.path.join(d3, "maps.npz"))
    if list(z["ref_keys"]) != refs:
        raise RuntimeError("ref order differs")
    with open(os.path.join(d3, "curves.json"), encoding="utf-8") as f:
        cv3 = json.load(f)
    with open(os.path.join(d3, "mismatch_assignment.json"), encoding="utf-8") as f:
        assign3 = json.load(f)["assignment"]
    # ---------------- inputs: ONE outer batch of 48 in frozen order
    derm_rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["derm7pt"]), "derm7pt")[0]
    root = cfg["project_root"]
    x = torch.stack([TD.ManifestDataset(derm_rows, r["cache_name"], os.path.join(root, "cache"),
                                        os.path.join(root, "DATASETS"), [r["record_uid"]], mode="eval")[0]["x"]
                     for r in refs_f]).to(dev)
    N = len(refs)
    R = {r["ref_key"]: r for r in refs_f}
    pairs = [{"pair_id": p["pair_id"], "label_mel": p["label_mel"], "record_uid_dermoscopy": p["record_uid_dermoscopy"],
              "record_uid_clinical": p["record_uid_clinical"]} for p in pairs_f]
    batching = {"outer_batch": N, "outer_batch_order": "frozen ref_order 0..47", "partial_batch": None,
                "ig_internal_batch": IG_IB, "curve_eval_batch": EVAL_BATCH, "gradcam_batch_reuse_check": 8}
    receipt = P.build_receipt(cfg, a.config, "completed",
                              details={"policy": policy, "stage": "3B.2", "batching": batching, "model_sha256": fsha})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)

    def dump(name, obj):
        with open(os.path.join(out, name), "x", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=float)

    def finish(status, extra):
        summary = dict({"stage_status": status, "batching": batching, "timings_s": T,
                        "peak_memory_alloc_reserved_mib": M,
                        "measured_elapsed_s": round(time.perf_counter() - t_all, 2)}, **extra)
        dump("summary.json", summary)
        receipt["status"] = "completed" if status == "completed" else "failed"
        receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))}
        rp = P.write_receipt(receipt, out)
        print(json.dumps(summary, indent=1, default=float)[:5000])
        print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")

    base = timed("baseline", lambda: XA.gaussian_baseline(x))
    # ---------------- fresh IG, both m, same outer batch
    ig = {}
    for m in (M_CAND, M_REF):
        ig[m] = timed(f"attr_ig_m{m}", lambda m=m: XA.integrated_gradients(model, x, base, m, IG_IB))
    dlx = float((ig[M_CAND]["logit"] - ig[M_REF]["logit"]).abs().max())
    dlb = float((ig[M_CAND]["logit_baseline"] - ig[M_REF]["logit_baseline"]).abs().max())
    endpoint_agree = {"max_abs_logit_x": dlx, "max_abs_logit_baseline": dlb, "tolerance": TOL,
                      "pass": dlx <= TOL and dlb <= TOL}
    dump("endpoint_logit_agreement.json", endpoint_agree)
    if not endpoint_agree["pass"]:
        finish("stopped", {"reason": "endpoint logits differ between fresh IG calls", "endpoint_agree": endpoint_agree})
        return
    # ---------------- reuse verification (m-independent)
    def reuse_check():
        cams = [XA.grad_cam(model, model.layer4, x[s:s + 8]) for s in range(0, N, 8)]
        cam = torch.cat([c["map"] for c in cams]).cpu().numpy()
        sob = np.stack([XK.sobel_map(x[i]) for i in range(N)]).astype(np.float32)
        r = {"gradcam_max_abs_diff": float(np.abs(cam - z["gradcam"]).max()),
             "sobel_max_abs_diff": float(np.abs(sob - z["sobel"]).max()),
             "baseline_logit_max_abs_diff": float(np.abs(ig[M_CAND]["logit_baseline"].cpu().numpy() -
                                                         z["baseline_logit"]).max()),
             "stage3b_receipt_and_outputs_verified": True}
        r["pass"] = all(r[k] == 0.0 for k in ("gradcam_max_abs_diff", "sobel_max_abs_diff",
                                              "baseline_logit_max_abs_diff"))
        return r
    reuse = timed("reuse_verification", reuse_check)
    dump("reuse_verification.json", reuse)
    # ---------------- curves for both m
    recs, curves = {}, {}

    def do_curves():
        for m in (M_CAND, M_REF):
            recs[m], curves[m] = {}, {}
            for i, k in enumerate(refs):
                c = ig[m]["completeness"][i]
                el = XA.eligibility(XA.validity(ig[m]["primary"][i]), c)
                try:
                    cv = XC.analysis_curves(model, x[i], base[i], ig[m]["primary"][i], el, "ig", eval_batch=EVAL_BATCH)
                except XC.IneligibleMap as e:
                    cv = {"analysis_status": "ineligible", "reason": str(e)}
                prim = cv["analysis_status"] == "primary"
                curves[m][k] = cv
                recs[m][k] = {"completeness_failure": bool(c["failure"]), "e_rel": c["e_rel"], "e_abs": c["e_abs"],
                              "sum_attr": c["sum_attr"], "delta_logit": c["delta_logit"],
                              "eligible": el["eligible_for_primary_analysis"], "reasons": el["ineligibility_reasons"],
                              "map_flags": el["map_flags"], "analysis_status": cv["analysis_status"],
                              "curve_status": cv.get("curve_status"), "endpoint_status": cv.get("endpoint_status"),
                              "auc_ins": cv.get("auc_insertion_v1") if prim else None,
                              "auc_del": cv.get("auc_deletion_v1") if prim else None}
    timed("curves_ig", do_curves)
    # ---------------- assessment
    groups = {"pooled": refs}
    for mod in ("dermoscopy", "clinical"):
        groups[f"modality={mod}"] = [k for k in refs if R[k]["modality"] == mod]
    for lab in ("1", "0"):
        groups[f"label={lab}"] = [k for k in refs if R[k]["label_mel"] == lab]
        for mod in ("dermoscopy", "clinical"):
            groups[f"label={lab},modality={mod}"] = [k for k in refs if R[k]["label_mel"] == lab and R[k]["modality"] == mod]
    assess = {g: XP.assess_candidate({k: recs[M_CAND][k] for k in ks}, {k: recs[M_REF][k] for k in ks}, ks)
              for g, ks in groups.items()}
    chosen = XP.choose_m({M_CAND: assess["pooled"]}, candidates=(M_CAND,))
    decision = {"candidate": M_CAND, "reference": M_REF, "pooled_overall": assess["pooled"]["overall"],
                "chosen_m": chosen, "status": "selected" if chosen else "UNRESOLVED",
                "scope": "seed-0 ResNet-50 (Stage 3A epoch 8), unperturbed Derm7pt, current pipeline, outer batch "
                         "specified per Addendum F" if chosen else None,
                "note": "m=1024 is not called converged (no 2048 comparison)"}
    counts = {}
    for m in (M_CAND, M_REF):
        counts[m] = {}
        for g, ks in groups.items():
            rr = [recs[m][k] for k in ks]
            counts[m][g] = {"n": len(rr), "completeness_failures": sum(r["completeness_failure"] for r in rr),
                            "e_rel_undefined": sum(r["e_rel"] is None for r in rr),
                            "all_zero": sum("all_zero" in r["map_flags"] for r in rr),
                            "constant": sum("constant" in r["map_flags"] for r in rr),
                            "nonfinite": sum("nonfinite" in r["map_flags"] for r in rr),
                            "endpoint_failures": sum(r["endpoint_status"] == "fail" for r in rr),
                            "usable": sum(XP.curve_usable(r) for r in rr)}
    per_image = []
    for k in refs:
        a5, a1 = recs[M_CAND][k], recs[M_REF][k]
        both = XP.curve_usable(a5) and XP.curve_usable(a1)
        per_image.append({"ref_key": k, "pair_id": R[k]["pair_id"], "label_mel": R[k]["label_mel"],
                          "modality": R[k]["modality"], "delta_logit": a5["delta_logit"],
                          "fail_512": a5["completeness_failure"], "e_rel_512": a5["e_rel"], "e_abs_512": a5["e_abs"],
                          "fail_1024": a1["completeness_failure"], "e_rel_1024": a1["e_rel"], "e_abs_1024": a1["e_abs"],
                          "endpoint_512": a5["endpoint_status"], "endpoint_1024": a1["endpoint_status"],
                          "auc_ins_512": a5["auc_ins"], "auc_ins_1024": a1["auc_ins"],
                          "auc_del_512": a5["auc_del"], "auc_del_1024": a1["auc_del"],
                          "d_ins": a1["auc_ins"] - a5["auc_ins"] if both else None,
                          "d_del": a1["auc_del"] - a5["auc_del"] if both else None,
                          "comparison": "both_usable" if both else "excluded:" + ";".join(
                              assess["pooled"]["c"]["not_eligible_both"].get(k, []))})
    dump("ig_assessment_512_vs_1024.json", {"rule": "Addendum D5/E3/F4 (unchanged)", "decision": decision,
                                            "assessments": assess, "counts": counts,
                                            "per_ref": {str(m): recs[m] for m in recs}})
    with open(os.path.join(out, "per_image_512_vs_1024.csv"), "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(per_image)
    print("decision:", decision, flush=True)
    # ---------------- paired eligibility (m=512: production if chosen, else DIAGNOSTIC)
    role = "production_setting_pilot" if chosen else "DIAGNOSTIC_ONLY (m=512 not selected)"
    e_ig = "ig512"
    mm = XK.mismatch_assignment([{"record_uid": R[k]["record_uid"], "case_id": R[k]["pair_id"], "label": R[k]["label_mel"],
                                  "modality": R[k]["modality"], "explainer": e_ig, "condition": "none"} for k in refs])
    for k in refs:
        u, mod = R[k]["record_uid"], R[k]["modality"]
        if mm[XK.assignment_key(u, mod, e_ig, "none")]["donor"] != assign3[XK.assignment_key(u, mod, "ig256", "none")]["donor"]:
            raise RuntimeError(f"planned donor changed for {k}")
    dump("mismatch_assignment_ig512.json", {"written_before_mismatch_curves": True, "role": role,
                                            "same_donor_identities_as_stage3b": True, "assignment": mm})
    mres = {}

    def mismatch_curves():
        for i, k in enumerate(refs):
            mod = R[k]["modality"]
            same = [j for j, kk in enumerate(refs) if R[kk]["modality"] == mod]
            dm, st = XK.resolve_mismatch(XK.assignment_key(R[k]["record_uid"], mod, e_ig, "none"), mm,
                                         {R[refs[j]]["record_uid"]: ig[M_CAND]["primary"][j] for j in same},
                                         {R[refs[j]]["record_uid"]: XP.curve_usable(recs[M_CAND][refs[j]]) for j in same})
            if dm is None:
                mres[k] = dict(st)
                continue
            c = XC.curves(model, x[i], base[i], dm, eval_batch=EVAL_BATCH)
            good = c["curve_status"] == "ok" and c["endpoint_status"] == "pass"
            mres[k] = dict(st, status="available" if good else "unavailable",
                           reason=None if good else "curve invalid or endpoint failure",
                           auc_ins=c["auc_insertion_v1"], auc_del=c["auc_deletion_v1"], endpoint_status=c["endpoint_status"])
    timed("curves_mismatch_ig512", mismatch_curves)
    expl_status, ctrl_status, score_rows = {}, {}, []
    for k in refs:
        u = R[k]["record_uid"]
        for e in ("gradcam", e_ig):
            if e == "gradcam":
                ex = cv3[k]["explainers"]["gradcam"]
                expl = {"auc_ins": ex["auc_ins"], "auc_del": ex["auc_del"]}
                usable = ex["status"] == "primary"
                reasons = [] if usable else ["map:all_zero/constant (Stage 3B rule, unchanged)"]
                mis = cv3[k]["controls"]["mismatch_gradcam"]
            else:
                r5 = recs[M_CAND][k]
                expl = {"auc_ins": r5["auc_ins"], "auc_del": r5["auc_del"]}
                usable = r5["analysis_status"] == "primary"
                reasons = [] if usable else (r5["reasons"] or [r5["analysis_status"]])
                mis = mres[k]
            expl_status[(u, e)] = {"usable": usable, "reasons": reasons}
            ctl = {"random": cv3[k]["controls"]["random"], "centre": cv3[k]["controls"]["centre"],
                   "sobel": cv3[k]["controls"]["sobel"], "mismatch": mis}
            if not reuse["pass"]:
                ctl = {cn: ({"status": "unavailable", "reason": "m-independent reuse not verified"}
                            if cn != "mismatch" or e == "gradcam" else cd) for cn, cd in ctl.items()}
            for cn, cd in ctl.items():
                sc = XK.control_scores(expl, cd)
                ctrl_status[(u, e, cn)] = {"available": sc["status"] == "available", "reason": sc.get("reason")}
                score_rows.append({"ref_key": k, "pair_id": R[k]["pair_id"], "label_mel": R[k]["label_mel"],
                                   "modality": R[k]["modality"], "explainer": e, "control": cn,
                                   "role": "pilot_gradcam" if e == "gradcam" else role,
                                   "control_status": sc["status"], "reason": sc.get("reason"),
                                   "insertion_advantage": sc["insertion_advantage"],
                                   "deletion_advantage": sc["deletion_advantage"]})
    ptab = XP.pair_table(pairs, expl_status, ctrl_status, ["gradcam", e_ig], ["random", "centre", "sobel", "mismatch"])
    psum = {}
    for e in ("gradcam", e_ig):
        for c in ("explainer", "random", "centre", "sobel", "mismatch"):
            rr = [r for r in ptab if r["explainer"] == e and r["contrast"] == c]
            bl = {lab: sum(r["complete_pair"] for r in rr if r["label_mel"] == lab) for lab in ("1", "0")}
            psum[f"{e}|{c}"] = {"role": "pilot_gradcam" if e == "gradcam" else role,
                                "complete_pairs": sum(r["complete_pair"] for r in rr), "of": len(rr), "by_label": bl,
                                "NOT_ESTIMABLE_label_cells": [lab for lab, v in bl.items() if v == 0],
                                "exclusions": {r["pair_id"]: r["reasons"] for r in rr if not r["complete_pair"]}}
    with open(os.path.join(out, "pair_eligibility_ig512.csv"), "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "label_mel", "explainer", "contrast", "complete_pair", "reasons"],
                           lineterminator="\n")
        w.writeheader()
        for r in ptab:
            w.writerow(dict(r, reasons=";".join(r["reasons"])))
    with open(os.path.join(out, "scores_ig512.csv"), "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(score_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(score_rows)

    def persist():
        np.savez_compressed(os.path.join(out, "maps_fresh.npz"), ref_keys=np.array(refs),
                            logit=ig[M_CAND]["logit"].cpu().numpy(), baseline_logit=ig[M_CAND]["logit_baseline"].cpu().numpy(),
                            **{f"ig{m}_attr": ig[m]["attr"].cpu().numpy() for m in ig},
                            **{f"ig{m}_primary": ig[m]["primary"].cpu().numpy() for m in ig})
        keep = ("analysis_status", "reason", "insertion", "deletion", "fractions", "auc_insertion_v1", "auc_deletion_v1",
                "p_full", "p_base", "endpoint_status", "endpoint_prediction_max_abs", "curve_status", "eval_batch")
        dump("curves_fresh.json", {str(m): {k: {kk: curves[m][k].get(kk) for kk in keep} for k in refs} for m in curves})
        dump("mismatch_ig512_results.json", mres)
    timed("persistence", persist)
    t3 = s3["timings_s"]
    t_curve = s3["projection"]["per_ref_s"]["curves_per_map_pair_ins_del"]
    per_ref = {"baseline": T["baseline"] / N, "gradcam": t3["attr_gradcam"] / N, "ig_m512": T["attr_ig_m512"] / N,
               "control_generation": t3["control_generation"] / N, "curve_pair_per_map_stage3b": t_curve,
               "persistence": t3["persistence"] / N}
    s_ref = sum(v for kk, v in per_ref.items() if not kk.startswith("curve")) + 11 * t_curve
    projection = {"label": "PROJECTION, not a measurement: 996 refs, same GPU, seed-0 ResNet-50, unperturbed Derm7pt, "
                           "IG m=512 with outer batch 48 (+ one final partial batch of 36 -> its endpoint batching must "
                           "be specified), eval_batch 11, 11 curve pairs/ref; excludes other models/seeds/shifts",
                  "per_ref_s": per_ref, "projected_s_per_ref": s_ref, "projected_total_h": s_ref * N_FULL_REFS / 3600,
                  "projected_disk_GB": {"float32_maps_uncompressed": 1204224 * N_FULL_REFS / 1e9,
                                        "curves_json": s3["projection"]["storage"]["curves_json_total_GB_upper"],
                                        "note": "DISK storage, not GPU memory"}}
    finish("completed", {"decision": decision, "endpoint_agree": endpoint_agree, "reuse": reuse,
                         "counts": counts, "pair_summary": psum, "projection": projection})


if __name__ == "__main__":
    main()
