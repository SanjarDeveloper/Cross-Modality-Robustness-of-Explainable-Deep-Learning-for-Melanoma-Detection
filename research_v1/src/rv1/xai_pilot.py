"""Stage 3B bounded real-model XAI pilot (Addendum D). NUMERICAL-SETTINGS PILOT ONLY.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_pilot --config ../configs/xai_pilot_stage3b_v1.json --out-root ../runs/xai_pilot

Model: selected v1 checkpoint (best.json; epoch 8), built as timm resnet50 (num_classes=1, no
pretrained download) and loaded strictly from the checkpoint state dict.
Sample: frozen 24 pairs / 48 refs (configs/xai_pilot_stage3b_v1_{pairs,refs}.csv), re-derived
and compared before inference. Order of work: attribution (Grad-CAM; IG m=128/256/512) ->
batch-equivalence check (eval_batch 11 vs 1) -> explainer curves -> IG convergence decision ->
mismatch assignment persisted -> control generation -> control curves -> scores -> paired
eligibility -> persistence. Stops safely after 30 min, keeping partial diagnostics.
"""
import argparse
import csv
import json
import os
import time

EXPECTED_SHA = "303d21e71986dc8f26dec84c6822a301f696060e5f08f613c90fc33c41a003dd"
IG_MS = (128, 256, 512)
CANDIDATES = (128, 256)
IG_IB = 16
EVAL_BATCH_PROVISIONAL = 11
DIAG_M = 256
BOUND_S = 30 * 60
N_FULL_REFS = 996


class PilotStop(RuntimeError):
    pass


def _read_csv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sidecar_ok(path):
    from . import provenance as P
    with open(path + ".sha256", encoding="utf-8") as f:
        h = f.read().split()[0]
    return h == P.sha256_file(path)


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
    if cfg["purpose"] != "xai_diag":
        raise C.ConfigError("requires purpose 'xai_diag'")
    si = cfg["supporting_inputs"]
    dev = "cuda"
    seed_everything(0, cfg["determinism"])
    T, M, log = {}, {}, []

    def check_time(where):
        el = time.perf_counter() - t_all
        if el > BOUND_S:
            raise PilotStop(f"30-minute bound exceeded at {where} ({el:.0f}s)")

    def timed(name, fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        T[name] = round(time.perf_counter() - t0, 3)
        M[name] = [round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                   round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)]
        log.append(f"{name}: {T[name]}s peak {M[name]}")
        print(log[-1], flush=True)
        return out

    # ------------------------------------------------ model verification BEFORE loading
    ck_path = C.resolve(cfg, si["model_checkpoint"])
    with open(C.resolve(cfg, si["model_best_pointer"]), encoding="utf-8") as f:
        best = json.load(f)
    file_sha = P.sha256_file(ck_path)
    with open(ck_path + ".sha256", encoding="utf-8") as f:
        side_sha = f.read().split()[0]
    model_check = {"expected": EXPECTED_SHA, "file": file_sha, "sidecar": side_sha, "best_json": best["sha256"],
                   "best_json_checkpoint": best["checkpoint"], "best_epoch": best["epoch"]}
    if not (file_sha == side_sha == best["sha256"] == EXPECTED_SHA and
            best["checkpoint"] == os.path.basename(ck_path)):
        raise RuntimeError(f"checkpoint verification failed: {model_check}")
    ck = TR._load_verified(ck_path, EXPECTED_SHA)
    model = timm.create_model("resnet50", pretrained=False, num_classes=1)
    model.load_state_dict(ck["model"], strict=True)
    model = model.to(dev).eval()
    model_check.update(verified=True, loaded_strict=True, ckpt_epoch=ck.get("epoch"))
    del ck
    # ------------------------------------------------ frozen sample, re-derived and compared
    for k in ("pilot_pairs", "pilot_refs"):
        if not _sidecar_ok(C.resolve(cfg, si[k])):
            raise RuntimeError(f"{k} sidecar mismatch")
    pairs_f = _read_csv(C.resolve(cfg, si["pilot_pairs"]))
    refs_f = _read_csv(C.resolve(cfg, si["pilot_refs"]))
    xai_rows = _read_csv(C.resolve(cfg, cfg["manifests"]["xai_derm7pt"]))
    derm_rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["derm7pt"]), "derm7pt")[0]
    pairs = XP.select_pilot_pairs(xai_rows, derm_rows, 12)
    if [p["pair_id"] for p in pairs] != [p["pair_id"] for p in pairs_f]:
        raise RuntimeError("frozen pair list differs from re-derived selection")
    sub_pairs = set(XP.batch_subset(pairs, 2))
    if sub_pairs != {p["pair_id"] for p in pairs_f if p["batch_check_subset"] == "1"}:
        raise RuntimeError("batch subset differs")
    by_uid = {r["record_uid"]: r for r in derm_rows}
    refs = [r["ref_key"] for r in refs_f]
    R = {r["ref_key"]: dict(r, row=by_uid[r["record_uid"]]) for r in refs_f}
    for r in refs_f:
        d = by_uid[r["record_uid"]]
        if (d["in_derm7pt_primary"], d["image_uid"], d["sha256"], d["modality"]) != \
                ("1", r["image_uid"], r["sha256"], r["modality"]):
            raise RuntimeError(f"ref mismatch {r['ref_key']}")
    root = cfg["project_root"]
    xs = []
    for r in refs_f:
        ds = TD.ManifestDataset(derm_rows, r["cache_name"], os.path.join(root, "cache"),
                                os.path.join(root, "DATASETS"), [r["record_uid"]], mode="eval")
        it = ds[0]
        if it["record"]["record_uid"] != r["record_uid"] or int(it["label"]) != int(r["label_mel"]):
            raise RuntimeError("dataset row mismatch")
        xs.append(it["x"])
    x = torch.stack(xs).to(dev)
    N = len(refs)
    with XA.xai_precision():
        policy = XA.precision_settings()
    receipt = P.build_receipt(cfg, a.config, "completed",
                              details={"policy": policy, "controls_version": XK.CONTROLS_VERSION,
                                       "model_check": model_check, "stage": "3B"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    status = {"stage_status": "running", "stopped_reason": None}
    S = {}                                                     # summary pieces

    def dump(name, obj):
        with open(os.path.join(out, name), "x", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=float)

    try:
        # ------------------------------------------------ attribution
        base = timed("baseline", lambda: XA.gaussian_baseline(x))
        cams = []

        def do_cam():
            for s in range(0, N, 8):
                cams.append(XA.grad_cam(model, model.layer4, x[s:s + 8]))
            return {k: torch.cat([c[k] for c in cams]) for k in ("map", "logit")}
        cam = timed("attr_gradcam", do_cam)
        p_full = torch.sigmoid(cam["logit"]).cpu().tolist()
        ig = {}
        for m in IG_MS:
            check_time(f"ig m={m}")
            ig[m] = timed(f"attr_ig_m{m}", lambda m=m: XA.integrated_gradients(model, x, base, m, IG_IB))
        expl_maps = {"gradcam": cam["map"].cpu()}
        elig = {"gradcam": [XA.eligibility(XA.validity(cam["map"][i])) for i in range(N)]}
        for m in IG_MS:
            expl_maps[f"ig{m}"] = ig[m]["primary"].cpu()
            elig[f"ig{m}"] = [XA.eligibility(XA.validity(ig[m]["primary"][i]), ig[m]["completeness"][i])
                              for i in range(N)]
        attr_maps = list(expl_maps)
        # ------------------------------------------------ batch equivalence (fixed maps, subset)
        check_time("batch check")
        sub_idx = [i for i, k in enumerate(refs) if R[k]["pair_id"] in sub_pairs]

        def batch_check():
            comps, skipped = [], []
            for i in sub_idx:
                for e in attr_maps:
                    mp = expl_maps[e][i].to(dev)
                    if not elig[e][i]["map_shape_valid"]:
                        skipped.append({"ref": refs[i], "map": e, "flags": elig[e][i]["map_flags"]})
                        continue
                    c11 = XC.curves(model, x[i], base[i], mp, eval_batch=EVAL_BATCH_PROVISIONAL)
                    c1 = XC.curves(model, x[i], base[i], mp, eval_batch=1)
                    comps.append((f"{refs[i]}|{e}", c11, c1))
            r = XP.batch_equivalence(comps)
            r["skipped_ineligible_maps"] = skipped
            r["attribution_batching"] = {"gradcam_batch": 8, "ig_internal_batch": IG_IB}
            return r
        beq = timed("batch_equivalence_check", batch_check)
        EVAL_BATCH = EVAL_BATCH_PROVISIONAL if beq["pass"] else 1
        beq["eval_batch_used"] = EVAL_BATCH
        dump("batch_equivalence.json", beq)
        print("batch equivalence:", beq["pass"], "max dAUC", beq["max_d_auc"], flush=True)
        # ------------------------------------------------ explainer curves
        ex_res = {e: [None] * N for e in attr_maps}

        def expl_curves():
            for i in range(N):
                check_time("explainer curves")
                for e in attr_maps:
                    meth = "gradcam" if e == "gradcam" else "ig"
                    try:
                        cv = XC.analysis_curves(model, x[i], base[i], expl_maps[e][i].to(dev), elig[e][i], meth,
                                                eval_batch=EVAL_BATCH)
                        ok = cv["analysis_status"] == "primary"
                        ex_res[e][i] = {"status": cv["analysis_status"], "curve": cv,
                                        "auc_ins": cv["auc_insertion_v1"] if ok else None,
                                        "auc_del": cv["auc_deletion_v1"] if ok else None}
                    except XC.IneligibleMap as exc:
                        ex_res[e][i] = {"status": "ineligible", "reason": str(exc), "auc_ins": None,
                                        "auc_del": None}
        timed("curves_explainers", expl_curves)
        n_expl_curves = sum(1 for e in attr_maps for i in range(N) if "curve" in ex_res[e][i])
        # ------------------------------------------------ IG convergence
        def ig_rec(m, i):
            c = ig[m]["completeness"][i]
            er = ex_res[f"ig{m}"][i]
            cv = er.get("curve") or {}
            return {"completeness_failure": bool(c["failure"]), "e_rel": c["e_rel"], "e_abs": c["e_abs"],
                    "eligible": elig[f"ig{m}"][i]["eligible_for_primary_analysis"],
                    "reasons": elig[f"ig{m}"][i]["ineligibility_reasons"],
                    "map_flags": elig[f"ig{m}"][i]["map_flags"],
                    "curve_status": cv.get("curve_status"), "endpoint_status": cv.get("endpoint_status"),
                    "auc_ins": cv.get("auc_insertion_v1") if er["status"] == "primary" else None,
                    "auc_del": cv.get("auc_deletion_v1") if er["status"] == "primary" else None}
        per_m = {m: {refs[i]: ig_rec(m, i) for i in range(N)} for m in IG_MS}
        groups = {"pooled": refs}
        for mod in ("dermoscopy", "clinical"):
            groups[f"modality={mod}"] = [k for k in refs if R[k]["modality"] == mod]
        for lab in ("1", "0"):
            groups[f"label={lab}"] = [k for k in refs if R[k]["label_mel"] == lab]
            for mod in ("dermoscopy", "clinical"):
                groups[f"label={lab},modality={mod}"] = [k for k in refs if R[k]["label_mel"] == lab and
                                                         R[k]["modality"] == mod]
        conv = {}
        for m in CANDIDATES:
            conv[m] = {g: XP.assess_candidate({k: per_m[m][k] for k in ks}, {k: per_m[2 * m][k] for k in ks}, ks)
                       for g, ks in groups.items()}
        chosen = XP.choose_m({m: conv[m]["pooled"] for m in CANDIDATES})
        per_m_counts = {}
        for m in IG_MS:
            per_m_counts[m] = {}
            for g, ks in groups.items():
                rr = [per_m[m][k] for k in ks]
                per_m_counts[m][g] = {
                    "n": len(rr), "completeness_failures": sum(r["completeness_failure"] for r in rr),
                    "e_rel_undefined": sum(r["e_rel"] is None for r in rr),
                    "all_zero": sum("all_zero" in r["map_flags"] for r in rr),
                    "constant": sum("constant" in r["map_flags"] for r in rr),
                    "nonfinite": sum("nonfinite" in r["map_flags"] for r in rr),
                    "endpoint_failures": sum(r["endpoint_status"] == "fail" for r in rr),
                    "curve_nonfinite": sum(r["curve_status"] == "nonfinite" for r in rr),
                    "eligible_primary": sum(r["eligible"] for r in rr),
                    "usable_curves": sum(XP.curve_usable(r) for r in rr),
                    "p95_e_rel": float(np.percentile([r["e_rel"] for r in rr if r["e_rel"] is not None], 95))
                    if any(r["e_rel"] is not None for r in rr) else None,
                    "max_e_rel": max((r["e_rel"] for r in rr if r["e_rel"] is not None), default=None),
                    "max_e_abs_undefined_rel": max((r["e_abs"] for r in rr if r["e_rel"] is None), default=None)}
        gc_counts = {g: {"n": len(ks), "nonfinite_or_constant": sum(not elig["gradcam"][refs.index(k)]["map_shape_valid"]
                                                                    for k in ks),
                         "endpoint_failures": sum((ex_res["gradcam"][refs.index(k)].get("curve") or {}).get(
                             "endpoint_status") == "fail" for k in ks),
                         "usable": sum(ex_res["gradcam"][refs.index(k)]["status"] == "primary" for k in ks)}
                     for g, ks in groups.items()}
        ig_ctrl_m = chosen if chosen is not None else DIAG_M
        ig_label = f"ig{ig_ctrl_m}"
        decision = {"chosen_m": chosen, "status": "selected" if chosen else "UNRESOLVED",
                    "controls_ig_m": ig_ctrl_m,
                    "controls_ig_role": "production_candidate_pilot" if chosen else "DIAGNOSTIC_ONLY (no candidate passed)",
                    "note": "m=512 is never a candidate (no 1024 comparison); 512 is not called converged"}
        dump("ig_convergence.json", {"rule": "Addendum D5", "decision": decision,
                                     "assessments": {str(m): conv[m] for m in CANDIDATES},
                                     "per_m_counts": {str(m): per_m_counts[m] for m in IG_MS},
                                     "gradcam_counts": gc_counts,
                                     "per_ref": {str(m): per_m[m] for m in IG_MS}})
        print("IG decision:", decision, flush=True)
        # ------------------------------------------------ mismatch assignment, persisted BEFORE control curves
        ctrl_expl = ["gradcam", ig_label]
        mm_recs = [{"record_uid": R[k]["record_uid"], "case_id": R[k]["pair_id"], "label": R[k]["label_mel"],
                    "modality": R[k]["modality"], "explainer": e, "condition": "none"}
                   for e in ctrl_expl for k in refs]
        assign = XK.mismatch_assignment(mm_recs)
        dump("mismatch_assignment.json", {"written_before_control_curves": True, "explainers": ctrl_expl,
                                          "assignment": assign})
        # ------------------------------------------------ control generation
        check_time("control generation")

        def gen():
            c = {}
            for i, k in enumerate(refs):
                c[k] = {"random": XK.random_maps(R[k]["image_uid"]), "centre": XK.centre_map(),
                        "sobel": XK.sobel_map(x[i])}
            return c
        ctrl = timed("control_generation", gen)
        # ------------------------------------------------ control curves
        def to_t(mp):
            return torch.as_tensor(np.asarray(mp), dtype=torch.float64, device=dev)
        uid_idx = {R[k]["record_uid"]: i for i, k in enumerate(refs)}
        c_res = {k: {} for k in refs}

        def ctrl_curves():
            n = 0
            for i, k in enumerate(refs):
                check_time("control curves")
                rc = [XC.curves(model, x[i], base[i], to_t(mp), eval_batch=EVAL_BATCH) for mp in ctrl[k]["random"]]
                n += len(rc)
                good = all(c["curve_status"] == "ok" and c["endpoint_status"] == "pass" for c in rc)
                c_res[k]["random"] = {"status": "available" if good else "unavailable",
                                      "reason": None if good else "random curve invalid or endpoint failure",
                                      "curves": rc, "auc_ins_each": [c["auc_insertion_v1"] for c in rc],
                                      "auc_del_each": [c["auc_deletion_v1"] for c in rc]}
                for name in ("centre", "sobel"):
                    ok, flags = XK.map_valid(ctrl[k][name])
                    if not ok:
                        c_res[k][name] = {"status": "unavailable", "reason": f"{name} map invalid: {flags}"}
                        continue
                    c = XC.curves(model, x[i], base[i], to_t(ctrl[k][name]), eval_batch=EVAL_BATCH)
                    n += 1
                    good = c["curve_status"] == "ok" and c["endpoint_status"] == "pass"
                    c_res[k][name] = {"status": "available" if good else "unavailable",
                                      "reason": None if good else "curve invalid or endpoint failure",
                                      "curve": c, "auc_ins": c["auc_insertion_v1"], "auc_del": c["auc_deletion_v1"]}
                for e in ctrl_expl:
                    mod = R[k]["modality"]
                    same = [j for j, kk in enumerate(refs) if R[kk]["modality"] == mod]
                    donor_maps = {R[refs[j]]["record_uid"]: expl_maps[e][j] for j in same}
                    donor_ok = {R[refs[j]]["record_uid"]: ex_res[e][j]["status"] == "primary" for j in same}
                    dm, st = XK.resolve_mismatch(XK.assignment_key(R[k]["record_uid"], mod, e, "none"), assign,
                                                 donor_maps, donor_ok)
                    if dm is None:
                        c_res[k][f"mismatch_{e}"] = dict(st)
                        continue
                    c = XC.curves(model, x[i], base[i], dm.to(dev), eval_batch=EVAL_BATCH)
                    n += 1
                    good = c["curve_status"] == "ok" and c["endpoint_status"] == "pass"
                    c_res[k][f"mismatch_{e}"] = dict(st, status="available" if good else "unavailable",
                                                     reason=None if good else "curve invalid or endpoint failure",
                                                     curve=c, auc_ins=c["auc_insertion_v1"], auc_del=c["auc_deletion_v1"])
            return n
        n_ctrl_curves = timed("curves_controls", ctrl_curves)
        # ------------------------------------------------ scores + paired eligibility
        score_rows, expl_status, ctrl_status = [], {}, {}
        for i, k in enumerate(refs):
            u = R[k]["record_uid"]
            for e in ctrl_expl:
                ex = ex_res[e][i]
                usable = ex["status"] == "primary"
                reasons = [] if usable else (elig[e][i]["ineligibility_reasons"] or [ex["status"]])
                expl_status[(u, e)] = {"usable": usable, "reasons": reasons}
                for cn in ("random", "centre", "sobel", f"mismatch_{e}"):
                    sc = XK.control_scores(ex, c_res[k][cn])
                    cname = cn.split("_")[0]
                    ctrl_status[(u, e, cname)] = {"available": sc["status"] == "available", "reason": sc.get("reason")}
                    score_rows.append({"ref_key": k, "pair_id": R[k]["pair_id"], "label_mel": R[k]["label_mel"],
                                       "modality": R[k]["modality"], "explainer": e, "explainer_status": ex["status"],
                                       "control": cname, "control_status": sc["status"], "reason": sc.get("reason"),
                                       "donor": c_res[k][cn].get("donor") if cname == "mismatch" else None,
                                       "explainer_auc_ins": ex["auc_ins"], "explainer_auc_del": ex["auc_del"],
                                       "control_auc_ins": sc.get("control_auc_ins"),
                                       "control_auc_del": sc.get("control_auc_del"),
                                       "insertion_advantage": sc["insertion_advantage"],
                                       "deletion_advantage": sc["deletion_advantage"]})
        ptab = XP.pair_table(pairs, expl_status, ctrl_status, ctrl_expl, ["random", "centre", "sobel", "mismatch"])
        pair_summary = {}
        for e in ctrl_expl:
            for c in ("explainer", "random", "centre", "sobel", "mismatch"):
                rr = [r for r in ptab if r["explainer"] == e and r["contrast"] == c]
                pair_summary[f"{e}|{c}"] = {"complete_pairs": sum(r["complete_pair"] for r in rr), "of": len(rr),
                                            "by_label": {lab: sum(r["complete_pair"] for r in rr if r["label_mel"] == lab)
                                                         for lab in ("1", "0")},
                                            "exclusions": {r["pair_id"]: r["reasons"] for r in rr if not r["complete_pair"]}}
        # ------------------------------------------------ persistence
        def strip(c):
            return {kk: c[kk] for kk in ("insertion", "deletion", "fractions", "auc_insertion_v1", "auc_deletion_v1",
                                         "p_full", "p_base", "endpoint_status", "endpoint_prediction_max_abs",
                                         "curve_status", "eval_batch", "analysis_status") if kk in c}

        def persist():
            np.savez_compressed(os.path.join(out, "maps.npz"), ref_keys=np.array(refs),
                                baseline_logit=ig[IG_MS[0]]["logit_baseline"].cpu().numpy(),
                                logit=cam["logit"].cpu().numpy(), gradcam=cam["map"].cpu().numpy(),
                                **{f"ig{m}_attr": ig[m]["attr"].cpu().numpy() for m in IG_MS},
                                **{f"ig{m}_primary": ig[m]["primary"].cpu().numpy() for m in IG_MS},
                                sobel=np.stack([ctrl[k]["sobel"] for k in refs]).astype(np.float32))
            cv = {}
            for i, k in enumerate(refs):
                cv[k] = {"explainers": {e: dict({kk: vv for kk, vv in ex_res[e][i].items() if kk != "curve"},
                                                curve=strip(ex_res[e][i]["curve"]) if "curve" in ex_res[e][i] else None)
                                        for e in attr_maps}, "controls": {}}
                for cn, d in c_res[k].items():
                    dd = {kk: vv for kk, vv in d.items() if kk not in ("curve", "curves")}
                    if "curves" in d:
                        dd["curves"] = [strip(c) for c in d["curves"]]
                    if "curve" in d:
                        dd["curve"] = strip(d["curve"])
                    cv[k]["controls"][cn] = dd
            with open(os.path.join(out, "curves.json"), "x", encoding="utf-8") as f:
                json.dump(cv, f)
            with open(os.path.join(out, "scores.csv"), "x", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(score_rows[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(score_rows)
            with open(os.path.join(out, "pair_eligibility.csv"), "x", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["pair_id", "label_mel", "explainer", "contrast", "complete_pair",
                                                  "reasons"], lineterminator="\n")
                w.writeheader()
                for r in ptab:
                    w.writerow(dict(r, reasons=";".join(r["reasons"])))
            with open(os.path.join(out, "per_image_records.csv"), "x", newline="", encoding="utf-8") as f:
                fields = ["ref_key", "pair_id", "label_mel", "modality", "p_full", "explainer", "map_flags",
                          "eligible_primary", "ineligibility_reasons", "analysis_status", "endpoint_status",
                          "endpoint_max_abs", "auc_ins", "auc_del", "ig_sum_attr", "ig_delta_logit", "ig_e_abs",
                          "ig_e_rel", "ig_completeness_failure"]
                w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
                w.writeheader()
                for i, k in enumerate(refs):
                    for e in attr_maps:
                        er, el = ex_res[e][i], elig[e][i]
                        c = ig[int(e[2:])]["completeness"][i] if e != "gradcam" else {}
                        cvv = er.get("curve") or {}
                        w.writerow({"ref_key": k, "pair_id": R[k]["pair_id"], "label_mel": R[k]["label_mel"],
                                    "modality": R[k]["modality"], "p_full": p_full[i], "explainer": e,
                                    "map_flags": ";".join(el["map_flags"]),
                                    "eligible_primary": el["eligible_for_primary_analysis"],
                                    "ineligibility_reasons": ";".join(el["ineligibility_reasons"]),
                                    "analysis_status": er["status"], "endpoint_status": cvv.get("endpoint_status"),
                                    "endpoint_max_abs": cvv.get("endpoint_prediction_max_abs"),
                                    "auc_ins": er["auc_ins"], "auc_del": er["auc_del"],
                                    "ig_sum_attr": c.get("sum_attr"), "ig_delta_logit": c.get("delta_logit"),
                                    "ig_e_abs": c.get("e_abs"), "ig_e_rel": c.get("e_rel"),
                                    "ig_completeness_failure": c.get("failure")})
        timed("persistence", persist)
        # ------------------------------------------------ timing / storage projection
        n_curves_total = n_expl_curves + n_ctrl_curves
        t_curve = (T["curves_explainers"] + T["curves_controls"]) / n_curves_total
        per_ref = {"gradcam": T["attr_gradcam"] / N, "baseline": T["baseline"] / N,
                   f"ig_m{ig_ctrl_m}": T[f"attr_ig_m{ig_ctrl_m}"] / N,
                   "control_generation": T["control_generation"] / N,
                   "curves_per_map_pair_ins_del": t_curve,
                   "persistence_pilot_per_ref": T["persistence"] / N}
        curves_per_ref_prod = 2 + 5 + 2 + 2               # gradcam, IG, 5 random, centre, sobel, 2 mismatch
        prod_ref_s = (per_ref["gradcam"] + per_ref["baseline"] + per_ref[f"ig_m{ig_ctrl_m}"] +
                      per_ref["control_generation"] + curves_per_ref_prod * t_curve + per_ref["persistence_pilot_per_ref"])
        npz_size = os.path.getsize(os.path.join(out, "maps.npz"))
        curves_size = os.path.getsize(os.path.join(out, "curves.json"))
        prod_float32_per_ref = (1 + 3 + 1 + 1) * 224 * 224 * 4          # gradcam, IG attr(3), IG primary, sobel
        projection = {
            "label": "PROJECTION from pilot measurements (same GPU, ResNet-50 seed 0, unperturbed Derm7pt, "
                     "eval_batch %d); not a measurement; excludes other architectures, seeds, shifts" % EVAL_BATCH,
            "ig_m_used": ig_ctrl_m, "ig_m_status": decision["status"], "per_ref_s": per_ref,
            "curves_per_ref": curves_per_ref_prod, "projected_s_per_ref": prod_ref_s,
            "n_refs": N_FULL_REFS, "projected_total_s": prod_ref_s * N_FULL_REFS,
            "projected_total_h": prod_ref_s * N_FULL_REFS / 3600,
            "alt_ig_m": {str(m): (prod_ref_s - per_ref[f"ig_m{ig_ctrl_m}"] + T[f"attr_ig_m{m}"] / N) * N_FULL_REFS / 3600
                         for m in IG_MS},
            "storage": {"pilot_maps_npz_bytes": npz_size, "pilot_curves_json_bytes": curves_size,
                        "pilot_maps_npz_contains": "gradcam, IG attr+primary at 3 m, sobel (float32), compressed",
                        "production_float32_uncompressed_per_ref": prod_float32_per_ref,
                        "production_float32_uncompressed_total_GB": prod_float32_per_ref * N_FULL_REFS / 1e9,
                        "curves_json_per_ref_bytes_pilot": curves_size / N,
                        "curves_json_total_GB_upper": curves_size / N * N_FULL_REFS / 1e9,
                        "assumption": "random/centre maps regenerated from seeds, not stored; input tensors re-derived "
                                      "from cache; curves stored as JSON"}}
        status["stage_status"] = "completed"
        S.update(decision=decision, batch_equivalence={kk: beq[kk] for kk in ("pass", "n_compared", "max_d_auc",
                                                                                  "eval_batch_used")},
                 pair_summary=pair_summary, projection=projection)
    except PilotStop as exc:
        status.update(stage_status="stopped", stopped_reason=str(exc))
        print("STOPPED:", exc, flush=True)
    summary = dict(status, policy=policy, model_check=model_check, n_pairs=len(pairs), n_refs=N,
                   settings={"ig_ms": IG_MS, "ig_internal_batch": IG_IB, "eval_batch_provisional": EVAL_BATCH_PROVISIONAL,
                             "S": 32, "controls_version": XK.CONTROLS_VERSION, "prep": "prep_v1_cached256"},
                   timings_s=T, peak_memory_alloc_reserved_mib=M,
                   total_elapsed_s=round(time.perf_counter() - t_all, 2), **S)
    dump("summary.json", summary)
    receipt["status"] = "completed" if status["stage_status"] == "completed" else "failed"
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))}
    rp = P.write_receipt(receipt, out)
    print(rp)
    print(json.dumps({kk: summary[kk] for kk in summary if kk not in ("policy",)}, indent=1, default=float)[:6000])
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


if __name__ == "__main__":
    main()
