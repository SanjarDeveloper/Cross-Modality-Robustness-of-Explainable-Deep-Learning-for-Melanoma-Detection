"""Stage 10C: production simulated-HAM XAI arm (protocol 02_rev2 §6.7) at fixed IG m=2048 (selected by the
retrospective Stage 10B.1 extension). Computation + completeness/integrity audit only; no inferential statistics.

Contract inherited unchanged from the Stage 10B spec (full_arm): 340 images (170 mel / 170 non-mel), 11 conditions,
3,740 references, chunk plan = per condition, blocks of 48 in frozen record order (7 x 48 + final 4 per condition;
88 chunks). Batch composition and order are frozen; no assumption of batch-size invariance and no bitwise comparison
across batch shapes (the pilot used 20-reference chunks).

Phases (separate processes; resumable; hashed atomic chunk dirs):
  spec      freeze the production contract + full mismatch assignment (hashed before any scoring) + failure policy
  compute   per chunk: perturbed inputs + transformed masks, baseline, Grad-CAM, IG m=2048 (signed attribution, primary
            map, completeness), predictions, eligibility, primary curves (analysis_curves) and as-computed curves for
            finite non-primary maps, 5 random / centre / Sobel controls, localisation; severity-0 identity check in the
            Stage 4A batch layout
  mismatch  per chunk: mismatch curves + localisation from the frozen assignment (donor maps from the same condition)
  audit     coverage, hashes, endpoint checks, eligibility/failure tables, jointly eligible baseline-vs-condition
            counts, matched-population localisation, measured cost

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_sim_prod --config ../configs/xai_sim_prod_10c_v1.json --phase spec --out-root ../runs/xai_sim_prod_10c
  ... --phase compute|mismatch|audit --run-dir <dir> [--max-chunks N]
"""
import argparse
import io
import json
import math
import os
import time
import uuid

import numpy as np

from . import xai_sim as XS

R10B = "research_v1/runs/xai_sim_10b/20261004T180422Z_ad7eef76_cdce001b"
R10B1 = "research_v1/runs/xai_sim_ext_10b1/20261008T163049Z_5ce62e92_abd19dad"
M = 2048
EXPL = ("gradcam", "ig2048")


def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _abs(cfg, rel):
    from . import config as C
    return C.resolve(cfg, rel)


# ------------------------------------------------------------------ pure helpers (tested)
def chunk_dir_name(i):
    return f"chunk_{i:03d}"


def jointly_eligible(status_by_ref, record_uids, cond_key, explainer):
    """Images eligible (primary) for `explainer` at severity 0 AND at `cond_key`. status_by_ref: {ref_key: {expl: status}}."""
    out = []
    for u in record_uids:
        a = status_by_ref.get(XS.ref_key(u, "none", 0.0), {}).get(explainer)
        f, s = cond_key.split("|")
        b = status_by_ref.get(XS.ref_key(u, f, float(s)), {}).get(explainer)
        if a == "primary" and b == "primary":
            out.append(u)
    return out


# ------------------------------------------------------------------ spec
def phase_spec(cfg, a):
    from . import provenance as P, xai_controls as XK, xai_prod as XP
    from .desc_4b3 import publish_bytes, publish_json
    run10, run101 = _abs(cfg, R10B), _abs(cfg, R10B1)
    s10, s10sha = XP._load(run10, "spec.json")
    s101, s101sha = XP._load(run101, "spec.json")
    g = XP._read_json(os.path.join(run101, "gate2_results.json"))
    if g["readiness_2048_vs_4096"]["status"] != "READY" or XP._read_json(os.path.join(run101, "gate1_results.json"))["gate1"]["status"] != "READY":
        raise RuntimeError("Stage 10B.1 did not select m=2048")
    for rp in ("receipt_gate1.json", "receipt_gate2.json"):
        if P.verify_receipt(os.path.join(run101, rp), cfg["project_root"]):
            raise RuntimeError(f"10B.1 {rp} does not verify")
    fa = s10["full_arm"]
    recs = fa["records"]
    by = {r["record_uid"]: r for r in recs}
    conds = {c["key"]: c for c in s10["conditions"]}
    chunks = []
    for i, c in enumerate(fa["chunks"]):
        f, sv = c["condition"].split("|")
        chunks.append({"chunk": i, "condition": c["condition"], "factor": f, "severity": float(sv), "block": c["block"],
                       "record_uids": c["record_uids"]})
    if sum(len(c["record_uids"]) for c in chunks) != 3740 or len(chunks) != 88:
        raise RuntimeError("chunk plan")
    for k in conds:
        us = [u for c in chunks if c["condition"] == k for u in c["record_uids"]]
        if us != [r["record_uid"] for r in recs]:
            raise RuntimeError(f"condition {k} coverage/order")
    mm_recs = [{"record_uid": r["record_uid"], "case_id": r["cluster_id"], "label": r["label_mel"], "modality": "dermoscopy",
                "explainer": e, "condition": k} for e in EXPL for k in conds for r in recs]
    assign = XK.mismatch_assignment(mm_recs)
    unassigned = sum(v["status"] != "assigned" for v in assign.values())
    srcs = {n: P.sha256_file(os.path.join(os.path.dirname(__file__), n)) for n in ("xai_sim_prod.py", *s10["sources"].keys())}
    for n, h in s10["sources"].items():
        if srcs[n] != h:
            raise RuntimeError(f"source changed since Stage 10B: {n}")
    spec = {"spec_version": "xai_sim_prod_10c_v1", "created_utc": _now(),
            "model": s10["model"], "cohort": {k: fa[k] for k in ("selection", "n", "n_lesions", "by_class", "mask_flags")},
            "records": recs, "conditions": s10["conditions"], "cohort_note": s10["cohort_note"],
            "ig": {"m": M, "selected_by": "retrospective Stage 10B.1 extension (m=2048 vs 4096; READY under the unchanged 10B rule)",
                   "stage10b1": {"run": R10B1, "spec_sha256": s101sha}, "rule": "riemann_midpoint", "internal_batch": XS.IG_IB},
            "inherited_from_stage10b": {"run": R10B, "spec_sha256": s10sha,
                                        "unchanged": ["sample", "conditions", "simulator", "preprocessing", "precision", "baseline",
                                                      "controls", "localisation definitions", "eligibility rules", "tolerances",
                                                      "curve settings", "chunk plan"]},
            "settings": s10["settings"] | {"ig": {"m": M, "rule": "riemann_midpoint", "internal_batch": XS.IG_IB}},
            "localisation": s10["localisation"] | {"note_area_matched": "area-matched IoU uses the ground-truth mask area to choose k; it is "
                                                                      "not an independent segmentation prediction",
                                                   "note_extent": "masks = lesion extent; under hair/glare the visible lesion is smaller"},
            "batching": {"chunk_plan": "Stage 10B full_arm.chunks: per condition (condition-major in conditions order), blocks of 48 "
                                       "in frozen record order; final partial block of 4 per condition; 88 chunks",
                         "chunks": chunks, "batch_note": "baseline, Grad-CAM (batch 8 within chunk), IG logits and attributions are "
                                                         "computed per chunk; numerics are batch-context dependent; no bitwise "
                                                         "comparison to the 20-reference pilot",
                         "resume": "chunk boundaries fixed here; published chunks verified by manifest hashes and skipped; "
                                   "incomplete .tmp dirs quarantined; resume never regroups"},
            "s0_identity": "severity-0 probabilities recomputed in the Stage 4A layout (HAM TEST manifest order, batch 16) must equal "
                           f"Stage 4A within {XS.PROB_TOL_S0}; chunk probabilities recorded descriptively",
            "failure_policy": {"protocol": "02_rev2 §6.6: per-image failure definition; 'The failure rate is reported in every run.' "
                                           "No production-wide stop rule is specified.",
                               "decision": "the Stage 10B/10B.1 5% readiness criterion is a pilot gate for choosing m; it does NOT "
                                           "become a production stop rule. Individual maps follow §6.4 eligibility; failed IG maps "
                                           "are retained with reasons, excluded from primary analysis, never repaired, m never raised "
                                           "per image, images never replaced. Failures reported pooled, per condition, per class.",
                               "hard_stops": "only integrity violations: input/hash/coverage mismatch, severity-0 identity failure, "
                                             "baseline mismatch between phases"},
            "mismatch": {"assignment": assign, "unassigned": unassigned,
                         "rule": "fixed derangement within (label, condition, explainer), donor from another lesion; frozen before "
                                 "any scoring; planned donor only; ineligible donor -> unavailable"},
            "stored_per_reference": ["Grad-CAM map", "IG signed attribution (float32)", "IG primary map", "completeness", "eligibility",
                                     "logit/prob", "primary + as-computed curves", "random/centre/Sobel curves", "localisation",
                                     "mask", "perturbation metadata"],
            "sources": srcs,
            "inputs": {k: {"path": v, "sha256": P.sha256_file(_abs(cfg, v))} for k, v in cfg["supporting_inputs"].items()}}
    from . import provenance as Pv
    receipt = Pv.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "spec.json", spec)
    publish_bytes(out, "spec.json.sha256", f"{P.sha256_file(os.path.join(out, 'spec.json'))}  spec.json\n".encode())
    receipt["outputs"] = {"spec.json": P.sha256_file(os.path.join(out, "spec.json"))}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out, "chunks", len(chunks), "mismatch unassigned", unassigned)


def _load(cfg, out):
    from . import provenance as P, xai_prod as XP
    spec, ssha = XP._load(out, "spec.json")
    for n, h in spec["sources"].items():
        if P.sha256_file(os.path.join(os.path.dirname(__file__), n)) != h:
            raise RuntimeError(f"source changed {n} (resume refused)")
    for k, v in spec["inputs"].items():
        if P.sha256_file(_abs(cfg, v["path"])) != v["sha256"]:
            raise RuntimeError(f"input changed {k}")
    return spec, ssha


def phase_compute(cfg, a):
    import csv as _csv
    import torch
    from . import xai_attr as XA, xai_controls as XK, xai_curves as XC, xai_prod as XP
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load(cfg, out)
    q = XP.quarantine_tmp(out)
    if q:
        print("quarantined incomplete:", q, flush=True)
    by = {r["record_uid"]: r for r in spec["records"]}
    with open(_abs(cfg, cfg["supporting_inputs"]["stage4a_ham_test_predictions"]), encoding="utf-8", newline="") as f:
        p4a = {r["record_uid"]: float(r["prob"]) for r in _csv.DictReader(f)}
    origs = XS._origs(cfg, spec["records"])
    masks = XS._masks256(cfg, spec["records"])
    model = XS._model(cfg)
    done_now = 0
    log = os.path.join(out, "progress.jsonl")
    for ch in spec["batching"]["chunks"]:
        final = os.path.join(out, chunk_dir_name(ch["chunk"]))
        if XP.verify_dir(final, ssha):
            print(f"skip verified {chunk_dir_name(ch['chunk'])}", flush=True)
            continue
        if a.max_chunks is not None and done_now >= a.max_chunks:
            print("max-chunks reached; stopping cleanly", flush=True)
            break
        rows = [by[u] for u in ch["record_uids"]]
        T = {}

        def tm(name, fn):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = fn()
            torch.cuda.synchronize()
            T[name] = T.get(name, 0.0) + time.perf_counter() - t0
            return r
        s0_ref = tm("s0_reference", lambda: XS._s0_reference(cfg, model, rows)) if ch["severity"] == 0.0 else None
        torch.cuda.reset_peak_memory_stats()
        t_chunk = time.perf_counter()
        xn, mk, metas = tm("perturb_and_masks", lambda: XS._condition_inputs(origs, masks, rows, ch["factor"], ch["severity"]))
        x = torch.from_numpy(xn).to("cuda")
        base = tm("baseline", lambda: XA.gaussian_baseline(x))
        n = len(rows)
        cam = tm("gradcam", lambda: {k: torch.cat([XA.grad_cam(model, model.layer4, x[s:s + XS.CAM_B])[k] for s in range(0, n, XS.CAM_B)])
                                     for k in ("map", "logit")})
        ig = tm("ig", lambda: XA.integrated_gradients(model, x, base, M, XS.IG_IB))
        maps = {"gradcam": cam["map"], "ig2048": ig["primary"]}
        recs = []
        for i, r in enumerate(rows):
            comp = ig["completeness"][i]
            p = 1.0 / (1.0 + math.exp(-float(ig["logit"][i])))
            rec = {"ref_key": XS.ref_key(r["record_uid"], ch["factor"], ch["severity"]), "record_uid": r["record_uid"],
                   "label_mel": r["label_mel"], "cluster_id": r["cluster_id"], "condition": ch["condition"], "chunk": ch["chunk"],
                   "perturb_meta": metas[i], "baseline_sha256": XP._sha_bytes(base[i].detach().cpu().numpy().tobytes()),
                   "logit_x": float(ig["logit"][i]), "prob_x": p, "logit_baseline": float(ig["logit_baseline"][i]),
                   "gradcam_logit": float(cam["logit"][i]), "mask": {"status": XS.mask_status(mk[i]), "fraction": float(mk[i].mean())},
                   "ig_completeness": comp, "explainers": {}, "controls": {}, "localisation": {}}
            if s0_ref is not None:
                qv = s0_ref[r["record_uid"]]
                rec["s0_check"] = {"prob_4a": p4a[r["record_uid"]], "prob_4a_layout": qv, "abs_diff": abs(qv - p4a[r["record_uid"]]),
                                   "chunk_prob_abs_diff_descriptive": abs(p - p4a[r["record_uid"]])}
                if rec["s0_check"]["abs_diff"] > XS.PROB_TOL_S0:
                    raise RuntimeError(f"severity-0 identity failed {r['record_uid']}")
            for e in EXPL:
                mp = maps[e][i]
                el = XA.eligibility(XA.validity(mp), None if e == "gradcam" else comp)
                d = {"eligibility": el, "map_sha256": XP._sha_bytes(mp.detach().cpu().numpy().tobytes())}
                try:
                    cv = tm("curves_explainers", lambda: XC.analysis_curves(model, x[i], base[i], mp, el, "gradcam" if e == "gradcam" else "ig",
                                                                             eval_batch=XS.EVAL_BATCH, s=XS.S))
                    d["primary_curve"], d["status"] = XS._strip(cv), cv["analysis_status"]
                except XC.IneligibleMap as ex:
                    d["status"], d["ineligible_reason"] = "ineligible", str(ex)
                    if "nonfinite" not in el["map_flags"]:
                        ac = tm("curves_as_computed", lambda: XC.curves(model, x[i], base[i], mp, s=XS.S, eval_batch=XS.EVAL_BATCH,
                                                                         allow_constant=True))
                        d["as_computed_curve"] = dict(XS._strip(ac), analysis_status="as_computed_non_primary",
                                                      reasons=el["ineligibility_reasons"])
                rec["explainers"][e] = d
                extra = None if el["eligible_for_primary_analysis"] else ";".join(el["ineligibility_reasons"])
                rec["localisation"][e] = tm("localisation", lambda: XS.localisation(mp.detach().cpu().numpy(), mk[i], extra))
            rnd = XK.random_maps(r["image_uid"])
            rc = tm("curves_random", lambda: [XC.curves(model, x[i], base[i], torch.as_tensor(m_, dtype=torch.float64, device="cuda"),
                                                        s=XS.S, eval_batch=XS.EVAL_BATCH) for m_ in rnd])
            ok = all(c["curve_status"] == "ok" and c["endpoint_status"] == "pass" for c in rc)
            rec["controls"]["random"] = {"status": "available" if ok else "unavailable",
                                         "reason": None if ok else "random curve invalid or endpoint failure",
                                         "curves": [XS._strip(c) for c in rc],
                                         "auc_ins_each": [c["auc_insertion_v1"] for c in rc], "auc_del_each": [c["auc_deletion_v1"] for c in rc]}
            rl = tm("localisation", lambda: [XS.localisation(m_, mk[i]) for m_ in rnd])
            rec["localisation"]["random_mean"] = ({"status": "available", **{k: float(np.mean([z[k] for z in rl])) for k in ("iou_area", "pointing_hit")}}
                                                  if all(z["status"] == "available" for z in rl) else {"status": "unavailable", "reason": rl[0].get("reason")})
            for name, mp in (("centre", XK.centre_map()), ("sobel", XK.sobel_map(x[i]))):
                v, flags = XK.map_valid(mp)
                if not v:
                    rec["controls"][name] = {"status": "unavailable", "reason": f"{name} map invalid: {flags}"}
                    rec["localisation"][name] = {"status": "unavailable", "reason": "map_invalid"}
                    continue
                c = tm(f"curves_{name}", lambda: XC.curves(model, x[i], base[i], torch.as_tensor(np.asarray(mp), dtype=torch.float64, device="cuda"),
                                                           s=XS.S, eval_batch=XS.EVAL_BATCH))
                good = c["curve_status"] == "ok" and c["endpoint_status"] == "pass"
                rec["controls"][name] = {"status": "available" if good else "unavailable", "reason": None if good else "curve invalid or endpoint failure",
                                         "curve": XS._strip(c), "auc_ins": c["auc_insertion_v1"], "auc_del": c["auc_deletion_v1"]}
                rec["localisation"][name] = tm("localisation", lambda: XS.localisation(np.asarray(mp), mk[i]))
            recs.append(rec)
        tmp = f"{final}.tmp-{uuid.uuid4().hex[:8]}"
        os.makedirs(tmp)
        buf = io.BytesIO()
        np.savez_compressed(buf, ref_keys=np.array([r["ref_key"] for r in recs]), masks=mk,
                            gradcam=cam["map"].cpu().numpy().astype(np.float32),
                            ig2048_primary=ig["primary"].cpu().numpy().astype(np.float32),
                            ig2048_attr_signed=ig["attr"].cpu().numpy().astype(np.float32))
        with open(os.path.join(tmp, "maps.npz"), "xb") as f:
            f.write(buf.getvalue())
        with open(os.path.join(tmp, "records.json"), "x", encoding="utf-8") as f:
            json.dump(recs, f)
        meta = {"spec_sha256": ssha, "chunk": ch["chunk"], "condition": ch["condition"], "n": n, "timing_s": T,
                "chunk_seconds": time.perf_counter() - t_chunk,
                "peak_cuda_mib": [round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)]}
        XP.publish_dir(tmp, final, meta)
        done_now += 1
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"chunk": ch["chunk"], "condition": ch["condition"], "n": n, "seconds": round(meta["chunk_seconds"], 1),
                                "peak": meta["peak_cuda_mib"], "utc": _now()}) + "\n")
        print(f"chunk {ch['chunk']:03d} {ch['condition']} n={n} {meta['chunk_seconds']:.1f}s ig={T['ig']:.1f}s", flush=True)
        del x
    done = sum(XP.verify_dir(os.path.join(out, chunk_dir_name(c["chunk"])), ssha) for c in spec["batching"]["chunks"])
    print(f"COMPUTE STATUS {done}/{len(spec['batching']['chunks'])}", flush=True)


def _records(out, spec, ssha, prefix="chunk"):
    from . import xai_prod as XP
    recs = []
    for ch in spec["batching"]["chunks"]:
        d = os.path.join(out, f"{prefix}_{ch['chunk']:03d}")
        if not XP.verify_dir(d, ssha):
            raise RuntimeError(f"{d} incomplete")
        recs += XP._read_json(os.path.join(d, "records.json"))
    return recs


def phase_mismatch(cfg, a):
    import torch
    from . import xai_attr as XA, xai_controls as XK, xai_curves as XC, xai_prod as XP
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load(cfg, out)
    XP.quarantine_tmp(out)
    assign = spec["mismatch"]["assignment"]
    by = {r["record_uid"]: r for r in spec["records"]}
    origs = XS._origs(cfg, spec["records"])
    masks = XS._masks256(cfg, spec["records"])
    model = XS._model(cfg)
    cond_cache = {}
    for ch in spec["batching"]["chunks"]:
        final = os.path.join(out, f"mismatch_{ch['chunk']:03d}")
        if XP.verify_dir(final, ssha):
            continue
        if ch["condition"] not in cond_cache:
            cond_cache.clear()
            dm, dok = {e: {} for e in EXPL}, {e: {} for e in EXPL}
            for c2 in spec["batching"]["chunks"]:
                if c2["condition"] != ch["condition"]:
                    continue
                d = os.path.join(out, chunk_dir_name(c2["chunk"]))
                if not XP.verify_dir(d, ssha):
                    raise RuntimeError(f"{d} incomplete")
                z = np.load(os.path.join(d, "maps.npz"))
                rr = XP._read_json(os.path.join(d, "records.json"))
                for j, r in enumerate(rr):
                    dm["gradcam"][r["record_uid"]] = z["gradcam"][j]
                    dm["ig2048"][r["record_uid"]] = z["ig2048_primary"][j]
                    for e in EXPL:
                        dok[e][r["record_uid"]] = XP.donor_eligible(r["explainers"][e])
            cond_cache[ch["condition"]] = (dm, dok)
        dm, dok = cond_cache[ch["condition"]]
        recs = XP._read_json(os.path.join(out, chunk_dir_name(ch["chunk"]), "records.json"))
        rows = [by[u] for u in ch["record_uids"]]
        t0 = time.perf_counter()
        xn, mk, _ = XS._condition_inputs(origs, masks, rows, ch["factor"], ch["severity"])
        x = torch.from_numpy(xn).to("cuda")
        base = XA.gaussian_baseline(x)
        tc = tl = 0.0
        res = []
        for i, r in enumerate(rows):
            if XP._sha_bytes(base[i].detach().cpu().numpy().tobytes()) != recs[i]["baseline_sha256"]:
                raise RuntimeError("baseline differs from compute phase")
            row = {"ref_key": recs[i]["ref_key"]}
            for e in EXPL:
                key = XK.assignment_key(r["record_uid"], "dermoscopy", e, ch["condition"])
                m_, st = XK.resolve_mismatch(key, assign, dm[e], dok[e])
                if m_ is None:
                    row[e] = dict(st)
                    continue
                ts = time.perf_counter()
                c = XC.curves(model, x[i], base[i], torch.as_tensor(m_, device="cuda"), s=XS.S, eval_batch=XS.EVAL_BATCH)
                tc += time.perf_counter() - ts
                ts = time.perf_counter()
                loc = XS.localisation(m_, mk[i])
                tl += time.perf_counter() - ts
                good = c["curve_status"] == "ok" and c["endpoint_status"] == "pass"
                row[e] = dict(st, status="available" if good else "unavailable", reason=None if good else "curve invalid or endpoint failure",
                              curve=XS._strip(c), auc_ins=c["auc_insertion_v1"], auc_del=c["auc_deletion_v1"], localisation=loc,
                              donor_cluster=by[st["donor"]]["cluster_id"])
            res.append(row)
        tmp = f"{final}.tmp-{uuid.uuid4().hex[:8]}"
        os.makedirs(tmp)
        with open(os.path.join(tmp, "records.json"), "x", encoding="utf-8") as f:
            json.dump(res, f)
        XP.publish_dir(tmp, final, {"spec_sha256": ssha, "chunk": ch["chunk"], "seconds": time.perf_counter() - t0,
                                    "curve_seconds": tc, "localisation_seconds": tl})
        print(f"mismatch {ch['chunk']:03d} {time.perf_counter() - t0:.1f}s", flush=True)
        del x


def phase_audit(cfg, a):
    from . import provenance as P, xai_controls as XK, xai_prod as XP
    from .desc_4b3 import publish_csv, publish_json
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load(cfg, out)
    recs = _records(out, spec, ssha)
    mm = {r["ref_key"]: r for r in _records(out, spec, ssha, "mismatch")}
    keys = [r["ref_key"] for r in recs]
    expected = [XS.ref_key(u, c["factor"], c["severity"]) for c in spec["conditions"] for u in [r["record_uid"] for r in spec["records"]]]
    cov = {"n_records": len(recs), "unique": len(set(keys)), "expected": len(expected), "exact_set_match": set(keys) == set(expected),
           "mismatch_rows": len(mm), "mismatch_set_match": set(mm) == set(keys)}
    if not (cov["exact_set_match"] and cov["unique"] == 3740 and cov["mismatch_set_match"]):
        raise RuntimeError(f"coverage {cov}")
    # baseline consistency between phases is enforced in mismatch; s0 identity
    s0 = [r["s0_check"] for r in recs if "s0_check" in r]
    conds = [c["key"] for c in spec["conditions"]]
    elig, ctl, loc, ep = [], [], [], {"explainer_primary_endpoint_fail": 0, "controls_endpoint_or_invalid": 0}
    status = {r["ref_key"]: {e: r["explainers"][e]["status"] for e in EXPL} for r in recs}
    for c in conds:
        for lab in ("1", "0"):
            rr = [r for r in recs if r["condition"] == c and r["label_mel"] == lab]
            for e in EXPL:
                reasons = {}
                for r in rr:
                    s = r["explainers"][e]["status"]
                    if s != "primary":
                        k = ";".join(r["explainers"][e]["eligibility"]["ineligibility_reasons"]) or s
                        reasons[k] = reasons.get(k, 0) + 1
                elig.append({"condition": c, "label_mel": lab, "explainer": e, "n": len(rr),
                             "primary": sum(r["explainers"][e]["status"] == "primary" for r in rr),
                             "excluded": sum(r["explainers"][e]["status"] != "primary" for r in rr), "reasons": json.dumps(reasons),
                             "endpoint_failure": sum(r["explainers"][e]["status"] == "endpoint_failure" for r in rr),
                             "as_computed_curves": sum("as_computed_curve" in r["explainers"][e] for r in rr)})
            for cn in ("random", "centre", "sobel"):
                ctl.append({"condition": c, "label_mel": lab, "control": cn, "n": len(rr),
                            "available": sum(r["controls"][cn]["status"] == "available" for r in rr)})
            for e in EXPL:
                ctl.append({"condition": c, "label_mel": lab, "control": f"mismatch_{e}", "n": len(rr),
                            "available": sum(mm[r["ref_key"]][e].get("status") == "available" for r in rr),
                            "unavailable_reasons": json.dumps({k: sum(mm[r["ref_key"]][e].get("reason") == k for r in rr)
                                                               for k in {mm[r["ref_key"]][e].get("reason") for r in rr} if k})})
    ep["explainer_primary_endpoint_fail"] = sum(r["explainers"][e]["status"] == "endpoint_failure" for r in recs for e in EXPL)
    ep["controls_endpoint_or_invalid"] = sum(r["controls"][cn]["status"] != "available" for r in recs for cn in ("random", "centre", "sobel"))
    ep["mismatch_curve_failures"] = sum(v.get("reason") == "curve invalid or endpoint failure" for r in mm.values() for e in EXPL for v in [r[e]])
    # IG failure summaries
    igf = []
    for grp, sel in [("pooled", lambda r: True)] + [(c, (lambda r, c=c: r["condition"] == c)) for c in conds] + \
            [(f"class={lab}", (lambda r, lab=lab: r["label_mel"] == lab)) for lab in ("1", "0")]:
        rr = [r for r in recs if sel(r)]
        er = [r["ig_completeness"]["e_rel"] for r in rr if r["ig_completeness"]["e_rel"] is not None]
        igf.append({"group": grp, "n": len(rr), "ig_completeness_failures": sum(r["ig_completeness"]["failure"] for r in rr),
                    "rate": sum(r["ig_completeness"]["failure"] for r in rr) / len(rr),
                    "fail_mel": sum(r["ig_completeness"]["failure"] for r in rr if r["label_mel"] == "1"),
                    "fail_nonmel": sum(r["ig_completeness"]["failure"] for r in rr if r["label_mel"] == "0"),
                    "e_rel_median": float(np.median(er)) if er else None, "e_rel_p95": float(np.percentile(er, 95)) if er else None,
                    "e_rel_undefined": len(rr) - len(er),
                    "gradcam_all_zero_or_constant": sum(r["explainers"]["gradcam"]["status"] != "primary" for r in rr)})
    # jointly eligible baseline-vs-condition counts (images), per explainer and class
    uids = [r["record_uid"] for r in spec["records"]]
    lab = {r["record_uid"]: r["label_mel"] for r in spec["records"]}
    joint = []
    rbk = {r["ref_key"]: r for r in recs}
    cl = {r["record_uid"]: r["cluster_id"] for r in spec["records"]}
    for c in conds[1:]:
        for e in EXPL:
            j = jointly_eligible(status, uids, c, e)
            randok = [u for u in j if all(rbk[XS.ref_key(u, k.split("|")[0], float(k.split("|")[1]))]["controls"]["random"]["status"] == "available"
                                          for k in ("none|0.000", c))]
            joint.append({"condition": c, "explainer": e, "jointly_eligible_images": len(j),
                          "mel": sum(lab[u] == "1" for u in j), "nonmel": sum(lab[u] == "0" for u in j),
                          "jointly_eligible_with_random_control": len(randok),
                          "lesions": len({cl[u] for u in j})})
    # matched-population localisation (descriptive availability + medians)
    for comp in (("ig2048", "centre", "random_mean", "sobel"), ("gradcam", "centre", "random_mean", "sobel"), ("ig2048", "gradcam", "centre", "random_mean")):
        for c in conds:
            rr = [r for r in recs if r["condition"] == c]
            m = [r for r in rr if all(r["localisation"][k]["status"] == "available" for k in comp)]
            row = {"comparison": "+".join(comp), "condition": c, "n_refs": len(rr), "n_matched": len(m),
                   "mel": sum(r["label_mel"] == "1" for r in m), "nonmel": sum(r["label_mel"] == "0" for r in m),
                   "lesions": len({r["cluster_id"] for r in m})}
            for k in comp:
                row[f"{k}_iou_area_median"] = float(np.median([r["localisation"][k]["iou_area"] for r in m])) if m else None
                row[f"{k}_pointing_rate"] = float(np.mean([r["localisation"][k]["pointing_hit"] for r in m])) if m else None
            loc.append(row)
    man = [XP._read_json(os.path.join(out, chunk_dir_name(c["chunk"]), "chunk_manifest.json")) for c in spec["batching"]["chunks"]]
    mman = [XP._read_json(os.path.join(out, f"mismatch_{c['chunk']:03d}", "chunk_manifest.json")) for c in spec["batching"]["chunks"]]
    tsum = {}
    for m_ in man:
        for k, v in m_["timing_s"].items():
            tsum[k] = tsum.get(k, 0.0) + v
    meas = {"compute_chunk_seconds_total": sum(m_["chunk_seconds"] for m_ in man), "seconds_by_component": tsum,
            "mismatch_seconds_total": sum(m_["seconds"] for m_ in mman),
            "chunk_seconds_min_median_max": [min(m_["chunk_seconds"] for m_ in man if m_["n"] == 48),
                                             float(np.median([m_["chunk_seconds"] for m_ in man if m_["n"] == 48])),
                                             max(m_["chunk_seconds"] for m_ in man if m_["n"] == 48)],
            "peak_cuda_mib_max": [max(m_["peak_cuda_mib"][0] for m_ in man), max(m_["peak_cuda_mib"][1] for m_ in man)],
            "disk_bytes": sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)}
    res = {"coverage": cov, "s0_identity": {"n": len(s0), "max_abs_diff_4a_layout": max(x["abs_diff"] for x in s0),
                                            "max_abs_diff_chunk_layout_descriptive": max(x["chunk_prob_abs_diff_descriptive"] for x in s0)},
           "endpoint_and_control_failures": ep, "measured": meas, "mismatch_unassigned": spec["mismatch"]["unassigned"],
           "notes": {"area_matched_iou": spec["localisation"]["note_area_matched"], "extent": spec["localisation"]["note_extent"],
                     "units": "3,740 references = 340 images x 11 conditions (repeated images across conditions)"}}
    publish_json(out, "audit_results.json", res)
    publish_csv(out, "eligibility_table.csv", elig, list(elig[0]))
    publish_csv(out, "control_availability_table.csv", ctl, ["condition", "label_mel", "control", "n", "available", "unavailable_reasons"])
    publish_csv(out, "ig_failure_table.csv", igf, list(igf[0]))
    publish_csv(out, "joint_eligibility_table.csv", joint, list(joint[0]))
    publish_csv(out, "matched_localisation_table.csv", loc, sorted({k for r in loc for k in r}))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "audit"})
    receipt["outputs"] = {n_: P.sha256_file(os.path.join(out, n_)) for n_ in sorted(os.listdir(out))
                          if os.path.isfile(os.path.join(out, n_)) and not n_.startswith("receipt")}
    receipt["outputs"].update({f"{d}/chunk_manifest.json": P.sha256_file(os.path.join(out, d, "chunk_manifest.json"))
                               for d in sorted(os.listdir(out)) if (d.startswith("chunk_") or d.startswith("mismatch_")) and ".tmp" not in d
                               and ".quarantined" not in d})
    rp = P.write_receipt(receipt, out, name="receipt_audit.json")
    print(json.dumps({k: res[k] for k in ("coverage", "s0_identity", "endpoint_and_control_failures", "measured")}, indent=1, default=str))
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "compute", "mismatch", "audit"))
    ap.add_argument("--out-root")
    ap.add_argument("--run-dir")
    ap.add_argument("--max-chunks", type=int, default=None)
    a = ap.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from . import config as C
    from .seeds import seed_everything
    cfg = C.load(a.config)
    if cfg["purpose"] != "xai_diag":
        raise C.ConfigError("requires purpose 'xai_diag'")
    seed_everything(0, cfg["determinism"])
    {"spec": phase_spec, "compute": phase_compute, "mismatch": phase_mismatch, "audit": phase_audit}[a.phase](cfg, a)


if __name__ == "__main__":
    main()
