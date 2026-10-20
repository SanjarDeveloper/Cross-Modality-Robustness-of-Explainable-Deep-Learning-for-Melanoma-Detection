"""Stage 7A: production seed-0 XAI on the frozen 498-pair Derm7pt XAI cohort (996 references).

Phases (separate processes; resumable):
  spec      freeze reference order, chunk membership (outer batches of 48, final 36), full-cohort
            mismatch donor mapping, pilot exposure flags, settings and hashes
  compute   per chunk: baseline, Grad-CAM, IG m=512, eligibility, explainer curves (primary via
            analysis_curves; non-primary finite maps 'as computed' and labelled), control curves
            (5 random, centre, Sobel). Each chunk is written to <chunk>.tmp-<id>/ and published by
            an atomic directory rename after a manifest of file hashes is written.
  mismatch  per chunk: mismatch-control curves from the frozen donor mapping (donor eligibility from
            computed records; invalid donor -> unavailable; never re-selected)
  audit     verify all chunks, scores via controls_v1, coverage/eligibility tables, receipt

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_prod --config ../configs/xai_prod_7a_v1.json --phase spec --out-root ../runs/xai_prod_7a
  ... --phase compute|mismatch|audit --run-dir <dir>
"""
import argparse
import csv
import datetime
import glob
import hashlib
import io
import json
import os
import time
import uuid

CHUNK = 48
IG_M, IG_IB, CAM_B, EVAL_BATCH, S = 512, 16, 8, 11, 32
EXPLAINERS = ("gradcam", "ig512")
CONTROLS = ("random", "centre", "sobel", "mismatch")
MODEL_SHA = "303d21e71986dc8f26dec84c6822a301f696060e5f08f613c90fc33c41a003dd"


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _read_bytes(p):
    with open(p, "rb") as f:
        return f.read()


def _read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _read_csv_rows(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ pure helpers (tested)
def build_refs(xai_rows, derm_rows):
    by = {r["record_uid"]: r for r in derm_rows}
    if len({r["pair_id"] for r in xai_rows}) != len(xai_rows):
        raise ValueError("duplicate pair_id")
    refs = []
    for r in xai_rows:
        for mod in ("dermoscopy", "clinical"):
            u = r[f"record_uid_{mod}"]
            d = by.get(u)
            if d is None or d["in_derm7pt_primary"] != "1" or d["modality"] != mod or d["pair_id"] != r["pair_id"] \
                    or d["label_mel"] != r["label_mel"] or d["image_uid"] != r[f"image_uid_{mod}"]:
                raise ValueError(f"reference check failed: {u}")
            refs.append({"ref_key": f"{u}|{mod}", "record_uid": u, "pair_id": r["pair_id"], "label_mel": r["label_mel"],
                         "modality": mod, "image_uid": d["image_uid"], "cache_name": d["cache_name"],
                         "cache_index": d["cache_index"]})
    keys = [x["ref_key"] for x in refs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate reference keys")
    return refs


def chunk_plan(n, size=CHUNK):
    return [{"chunk": i, "start": s, "end": min(s + size, n)} for i, s in enumerate(range(0, n, size))]


def mismatch_records(refs):
    from . import xai_controls as XK
    recs = [{"record_uid": r["record_uid"], "case_id": r["pair_id"], "label": r["label_mel"], "modality": r["modality"],
             "explainer": e, "condition": "none"} for e in EXPLAINERS for r in refs]
    return XK.mismatch_assignment(recs)


def publish_dir(tmp, final, meta):
    """Write chunk_manifest.json (hashes of all files) then atomically rename tmp -> final."""
    if os.path.exists(final):
        raise FileExistsError(final)
    files = {n: _sha_bytes(_read_bytes(os.path.join(tmp, n))) for n in sorted(os.listdir(tmp))}
    with open(os.path.join(tmp, "chunk_manifest.json"), "x", encoding="utf-8") as f:
        json.dump(dict(meta, files=files, completed_utc=_now()), f, indent=1)
    os.rename(tmp, final)


def verify_dir(final, spec_sha):
    """True only for a published directory whose manifest matches the spec and all file hashes."""
    mp = os.path.join(final, "chunk_manifest.json")
    if not os.path.isdir(final) or not os.path.exists(mp) or ".tmp-" in os.path.basename(final):
        return False
    m = _read_json(mp)
    if m.get("spec_sha256") != spec_sha:
        raise RuntimeError(f"{final}: spec hash mismatch (resume refused)")
    for n, h in m["files"].items():
        p = os.path.join(final, n)
        if not os.path.exists(p) or _sha_bytes(_read_bytes(p)) != h:
            raise RuntimeError(f"{final}: file {n} hash mismatch (resume refused)")
    return True


def is_result_dir(name):
    import re
    return re.fullmatch(r"(chunk|%s)_\d\d" % MISMATCH_PREFIX, name) is not None


def quarantine_tmp(run_dir):
    moved = []
    for d in glob.glob(os.path.join(run_dir, "*.tmp-*")):
        q = d.replace(".tmp-", ".quarantined-")
        os.rename(d, q)
        moved.append(os.path.basename(q))
    return moved


def primary_ok(curve_rec):
    return curve_rec.get("analysis_status") == "primary"


def donor_eligible(explainer_rec):
    """Explainer record (records.json -> explainers[e]): eligible donor only if its primary curve exists.
    (v1 wrongly passed this record to primary_ok, which reads 'analysis_status' -> always False.)"""
    return explainer_rec.get("status") == "primary" and primary_ok(explainer_rec.get("primary_curve") or {})


MISMATCH_PREFIX = "mismatch2"          # v2 after the donor-eligibility bug; v1 outputs kept in invalid_v1_donor_bug/
AMENDMENT = "amendment_mismatch_v2.json"


# ------------------------------------------------------------------ spec
def _paths(cfg):
    from . import config as C
    return {k: C.resolve(cfg, v) for k, v in cfg["supporting_inputs"].items()}


def phase_spec(cfg, a):
    from . import config as C, decode as DC, provenance as P, xai_attr as XA, xai_controls as XK
    from .desc_4b3 import publish_bytes, publish_json
    xai = _read_csv_rows(C.resolve(cfg, cfg["manifests"]["xai_derm7pt"]))
    derm = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["derm7pt"]), "derm7pt")[0]
    refs = build_refs(xai, derm)
    if (len(xai), sum(r["label_mel"] == "1" for r in xai), len(refs)) != (498, 249, 996):
        raise RuntimeError("XAI cohort counts differ")
    pilot = {r["pair_id"] for r in _read_csv_rows(_paths(cfg)["pilot_pairs"])}
    for r in refs:
        r["stage3b_pilot_member"] = r["pair_id"] in pilot
    plan = chunk_plan(len(refs))
    assign = mismatch_records(refs)
    src = {n: P.sha256_file(os.path.join(os.path.dirname(__file__), n)) for n in
           ("xai_attr.py", "xai_curves.py", "xai_controls.py", "xai_prod.py", "predict_v1.py", "torch_data.py", "preprocess.py")}
    with XA.xai_precision():
        pol = XA.precision_settings()
    spec = {"spec_version": "xai_prod_7a_v1", "created_utc": _now(),
            "model": {"checkpoint": cfg["supporting_inputs"]["model_checkpoint"], "sha256": MODEL_SHA,
                      "role": "Stage 3A seed-0 primary model (NOT the PAD-adapted model)"},
            "inputs": {k: {"path": v, "sha256": P.sha256_file(C.resolve(cfg, v))} for k, v in cfg["supporting_inputs"].items()
                       if not k.endswith("_img.npy")},
            "manifests": {k: {"path": cfg["manifests"][k], "sha256": P.sha256_file(C.resolve(cfg, cfg["manifests"][k]))}
                          for k in ("derm7pt", "xai_derm7pt")},
            "source_sha256": src, "precision_policy": pol,
            "settings": {"prep": "prep_v1_cached256", "baseline": "attr_v1 blurred baseline (k=57, sigma=9.5)",
                         "gradcam": {"layer": "layer4", "batch": CAM_B},
                         "ig": {"m": IG_M, "rule": "riemann_midpoint", "internal_batch": IG_IB,
                                "scope_note": "m=512 selected on the 24-pair Stage 3B.2 pilot for this model/pipeline only; "
                                              "not universally converged; every image is checked individually"},
                         "outer_batch": {"size": CHUNK, "n_chunks": len(plan), "final_partial_batch": plan[-1]["end"] - plan[-1]["start"],
                                         "membership": "frozen ref order; chunk k = refs[48k:48k+48]; baseline, completeness "
                                                       "logits and IG computed per chunk; resume never regroups"},
                         "curves": {"S": S, "eval_batch": EVAL_BATCH, "endpoint_tol": 1e-4},
                         "validity": "attr_v1 validity (nonfinite / all_zero / constant <1e-12) + IG completeness (e_rel<=0.05 "
                                     "if |delta|>=0.1 else e_abs<=0.05)",
                         "controls": "controls_v1 (5 random stable_seed('ctrl',k,image_uid), centre sigma=56, Sobel BT.601) + "
                                     "full-cohort mismatch (fresh assignment, NOT the pilot's)",
                         "as_computed": "finite non-primary maps (constant or IG completeness failure) get curves via "
                                        "curves(allow_constant=True), labelled 'as_computed_non_primary'; NaN/Inf maps: no curves",
                         "secondary_sources": "IG absolute map stored; IG signed channel attribution stored; IG positive "
                                              "smoothed (sigma=2) DEFERRED (not computed)"},
            "refs": refs, "chunks": plan,
            "cohort_note": "class-enriched (249+249 pairs); not natural prevalence; seed-0 model only"}
    from . import provenance as Pv
    receipt = Pv.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    for name, obj in (("spec.json", spec), ("mismatch_assignment.json",
                                            {"written_before_any_curve": True, "assignment": assign})):
        publish_json(out, name, obj)
        publish_bytes(out, name + ".sha256", f"{P.sha256_file(os.path.join(out, name))}  {name}\n".encode())
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("spec.json", "mismatch_assignment.json")}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out, "chunks", len(plan), "last", plan[-1])


def _load(out, name):
    from . import provenance as P
    p = os.path.join(out, name)
    if _read_bytes(p + ".sha256").decode("utf-8").split()[0] != P.sha256_file(p):
        raise RuntimeError(f"{name} hash mismatch")
    return _read_json(p), P.sha256_file(p)


def _check_inputs(cfg, spec):
    from . import config as C, provenance as P
    for k, v in spec["inputs"].items():
        if P.sha256_file(C.resolve(cfg, v["path"])) != v["sha256"]:
            raise RuntimeError(f"input changed {k}")
    out = spec.get("_run_dir")
    amended = {}
    if out and os.path.exists(os.path.join(out, AMENDMENT)):
        am, _ = _load(out, AMENDMENT)
        amended = {am["file"]: am["new_sha256"]}
    for n, h in spec["source_sha256"].items():
        cur = P.sha256_file(os.path.join(os.path.dirname(__file__), n))
        if cur != h and amended.get(n) != cur:
            raise RuntimeError(f"source changed {n} (resume refused)")


def _model(cfg):
    import timm
    from . import trainer as TR
    st = TR._load_verified(_paths(cfg)["model_checkpoint"], MODEL_SHA)
    m = timm.create_model("resnet50", pretrained=False, num_classes=1)
    m.load_state_dict(st["model"], strict=True)
    return m.to("cuda").eval()


def _inputs(cfg, refs):
    import torch
    from . import config as C, decode as DC, torch_data as TD
    derm = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["derm7pt"]), "derm7pt")[0]
    root = cfg["project_root"]
    ds = {c: TD.ManifestDataset(derm, c, os.path.join(root, "cache"), os.path.join(root, "DATASETS"),
                                [r["record_uid"] for r in derm if r["cache_name"] == c], mode="eval")
          for c in ("derm7pt_dermoscopy", "derm7pt_clinical")}
    idx = {c: {r["record_uid"]: i for i, r in enumerate(d.rows)} for c, d in ds.items()}
    xs = []
    for r in refs:
        it = ds[r["cache_name"]][idx[r["cache_name"]][r["record_uid"]]]
        if it["record"]["record_uid"] != r["record_uid"]:
            raise RuntimeError("row mismatch")
        xs.append(it["x"])
    return torch.stack(xs).to("cuda")


def _strip(c):
    keep = ("insertion", "deletion", "fractions", "boundaries", "auc_insertion_v1", "auc_deletion_v1", "p_full", "p_base",
            "endpoint_status", "endpoint_prediction_max_abs", "curve_status", "eval_batch", "analysis_status",
            "ineligibility_reasons", "method")
    return {k: c[k] for k in keep if k in c}


def _chunk_compute(model, x, refs):
    import numpy as np
    import torch
    from . import xai_attr as XA, xai_controls as XK, xai_curves as XC
    T = {}

    def tm(name, fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        T[name] = time.perf_counter() - t0
        return r
    n = len(refs)
    base = tm("baseline", lambda: XA.gaussian_baseline(x))
    cam = tm("gradcam", lambda: {k: torch.cat([XA.grad_cam(model, model.layer4, x[s:s + CAM_B])[k]
                                               for s in range(0, n, CAM_B)]) for k in ("map", "logit")})
    ig = tm("ig", lambda: XA.integrated_gradients(model, x, base, IG_M, IG_IB))
    maps = {"gradcam": cam["map"], "ig512": ig["primary"]}
    elig = {"gradcam": [XA.eligibility(XA.validity(cam["map"][i])) for i in range(n)],
            "ig512": [XA.eligibility(XA.validity(ig["primary"][i]), ig["completeness"][i]) for i in range(n)]}
    recs = []

    def curves_all():
        for i, r in enumerate(refs):
            rec = {"ref_key": r["ref_key"], "record_uid": r["record_uid"], "pair_id": r["pair_id"],
                   "label_mel": r["label_mel"], "modality": r["modality"], "image_uid": r["image_uid"],
                   "stage3b_pilot_member": r["stage3b_pilot_member"],
                   "logit_x": float(ig["logit"][i]), "logit_baseline": float(ig["logit_baseline"][i]),
                   "gradcam_logit": float(cam["logit"][i]),
                   "baseline_sha256": _sha_bytes(base[i].detach().cpu().numpy().tobytes()),
                   "ig_completeness": ig["completeness"][i], "explainers": {}, "controls": {}}
            for e in EXPLAINERS:
                mp = maps[e][i]
                el = elig[e][i]
                d = {"eligibility": el, "map_sha256": _sha_bytes(mp.detach().cpu().numpy().tobytes())}
                try:
                    cv = XC.analysis_curves(model, x[i], base[i], mp, el, "gradcam" if e == "gradcam" else "ig",
                                            eval_batch=EVAL_BATCH, s=S)
                    d["primary_curve"] = _strip(cv)
                    d["status"] = cv["analysis_status"]
                except XC.IneligibleMap as ex:
                    d["status"], d["ineligible_reason"] = "ineligible", str(ex)
                    if "nonfinite" not in el["map_flags"]:
                        ac = XC.curves(model, x[i], base[i], mp, s=S, eval_batch=EVAL_BATCH, allow_constant=True)
                        d["as_computed_curve"] = dict(_strip(ac), analysis_status="as_computed_non_primary",
                                                      reasons=el["ineligibility_reasons"])
                rec["explainers"][e] = d
            rnd = XK.random_maps(r["image_uid"])
            rc = [XC.curves(model, x[i], base[i], torch.as_tensor(m_, dtype=torch.float64, device="cuda"),
                            s=S, eval_batch=EVAL_BATCH) for m_ in rnd]
            ok = all(c["curve_status"] == "ok" and c["endpoint_status"] == "pass" for c in rc)
            rec["controls"]["random"] = {"status": "available" if ok else "unavailable",
                                         "reason": None if ok else "random curve invalid or endpoint failure",
                                         "curves": [_strip(c) for c in rc],
                                         "auc_ins_each": [c["auc_insertion_v1"] for c in rc],
                                         "auc_del_each": [c["auc_deletion_v1"] for c in rc]}
            for name, mp in (("centre", XK.centre_map()), ("sobel", XK.sobel_map(x[i]))):
                v, flags = XK.map_valid(mp)
                if not v:
                    rec["controls"][name] = {"status": "unavailable", "reason": f"{name} map invalid: {flags}"}
                    continue
                c = XC.curves(model, x[i], base[i], torch.as_tensor(np.asarray(mp), dtype=torch.float64, device="cuda"),
                              s=S, eval_batch=EVAL_BATCH)
                good = c["curve_status"] == "ok" and c["endpoint_status"] == "pass"
                rec["controls"][name] = {"status": "available" if good else "unavailable",
                                         "reason": None if good else "curve invalid or endpoint failure",
                                         "curve": _strip(c), "auc_ins": c["auc_insertion_v1"], "auc_del": c["auc_deletion_v1"]}
            recs.append(rec)
    tm("curves", curves_all)
    arrays = {"ref_keys": np.array([r["ref_key"] for r in refs]),
              "gradcam": cam["map"].cpu().numpy().astype(np.float32), "gradcam_logit": cam["logit"].cpu().numpy(),
              "ig512_primary": ig["primary"].cpu().numpy().astype(np.float32),
              "ig512_absolute": ig["absolute"].cpu().numpy().astype(np.float32),
              "ig512_attr_signed": ig["attr"].cpu().numpy().astype(np.float32),
              "sobel": np.stack([XK.sobel_map(x[i]) for i in range(n)]).astype(np.float32)}
    return recs, arrays, T


def phase_compute(cfg, a):
    import numpy as np
    import torch
    from . import provenance as P
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load(out, "spec.json")
    spec["_run_dir"] = out
    _check_inputs(cfg, spec)
    q = quarantine_tmp(out)
    if q:
        print("quarantined incomplete:", q, flush=True)
    model = _model(cfg)
    refs = spec["refs"]
    log = os.path.join(out, "progress.jsonl")
    t_all = time.perf_counter()
    for ch in spec["chunks"]:
        final = os.path.join(out, f"chunk_{ch['chunk']:02d}")
        if verify_dir(final, ssha):
            continue
        if time.perf_counter() - t_all > a.max_seconds:
            print("time budget reached; stopping cleanly", flush=True)
            break
        rr = refs[ch["start"]:ch["end"]]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        x = _inputs(cfg, rr)
        recs, arrays, T = _chunk_compute(model, x, rr)
        tmp = f"{final}.tmp-{uuid.uuid4().hex[:8]}"
        os.makedirs(tmp)
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        with open(os.path.join(tmp, "maps.npz"), "xb") as f:
            f.write(buf.getvalue())
        with open(os.path.join(tmp, "records.json"), "x", encoding="utf-8") as f:
            json.dump(recs, f)
        meta = {"spec_sha256": ssha, "chunk": ch, "ref_keys": [r["ref_key"] for r in rr],
                "timing_s": T, "chunk_seconds": time.perf_counter() - t0,
                "peak_cuda_mib": [round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                                  round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)]}
        publish_dir(tmp, final, meta)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"chunk": ch["chunk"], "n": len(rr), "seconds": round(meta["chunk_seconds"], 1),
                                "peak": meta["peak_cuda_mib"], "utc": _now()}) + "\n")
        print(f"chunk {ch['chunk']:02d} done n={len(rr)} {meta['chunk_seconds']:.1f}s peak={meta['peak_cuda_mib']}", flush=True)
        del x
    done = sum(verify_dir(os.path.join(out, f"chunk_{c['chunk']:02d}"), ssha) for c in spec["chunks"])
    print(f"COMPUTE STATUS {done}/{len(spec['chunks'])} chunks complete", flush=True)


def _all_records(out, spec, ssha, prefix="chunk"):
    recs = []
    for ch in spec["chunks"]:
        d = os.path.join(out, f"{prefix}_{ch['chunk']:02d}")
        if not verify_dir(d, ssha):
            raise RuntimeError(f"{d} incomplete")
        recs += _read_json(os.path.join(d, "records.json"))
    return recs


def phase_mismatch(cfg, a):
    import numpy as np
    import torch
    from . import xai_attr as XA, xai_controls as XK, xai_curves as XC
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load(out, "spec.json")
    spec["_run_dir"] = out
    _check_inputs(cfg, spec)
    assign, _ = _load(out, "mismatch_assignment.json")
    assign = assign["assignment"]
    quarantine_tmp(out)
    recs = {r["ref_key"]: r for r in _all_records(out, spec, ssha)}
    maps = {}
    for ch in spec["chunks"]:
        z = np.load(os.path.join(out, f"chunk_{ch['chunk']:02d}", "maps.npz"))
        for i, k in enumerate(z["ref_keys"]):
            maps[str(k)] = {"gradcam": z["gradcam"][i], "ig512": z["ig512_primary"][i]}
    uid_ref = {r["record_uid"]: r["ref_key"] for r in spec["refs"]}
    donor_ok = {e: {r["record_uid"]: donor_eligible(recs[r["ref_key"]]["explainers"][e]) for r in spec["refs"]} for e in EXPLAINERS}
    model = _model(cfg)
    t_all = time.perf_counter()
    for ch in spec["chunks"]:
        final = os.path.join(out, f"{MISMATCH_PREFIX}_{ch['chunk']:02d}")
        if verify_dir(final, ssha):
            continue
        rr = spec["refs"][ch["start"]:ch["end"]]
        t0 = time.perf_counter()
        x = _inputs(cfg, rr)
        base = XA.gaussian_baseline(x)                         # same chunk membership -> identical baseline
        res = []
        for i, r in enumerate(rr):
            if _sha_bytes(base[i].detach().cpu().numpy().tobytes()) != recs[r["ref_key"]]["baseline_sha256"]:
                raise RuntimeError(f"baseline differs from compute phase for {r['ref_key']}")
            row = {"ref_key": r["ref_key"]}
            for e in EXPLAINERS:
                key = XK.assignment_key(r["record_uid"], r["modality"], e, "none")
                dm, st = XK.resolve_mismatch(key, assign, {u: maps[uid_ref[u]][e] for u in uid_ref}, donor_ok[e])
                if dm is None:
                    row[e] = dict(st)
                    continue
                c = XC.curves(model, x[i], base[i], torch.as_tensor(dm, device="cuda"), s=S, eval_batch=EVAL_BATCH)
                good = c["curve_status"] == "ok" and c["endpoint_status"] == "pass"
                row[e] = dict(st, status="available" if good else "unavailable",
                              reason=None if good else "curve invalid or endpoint failure",
                              donor_ref_key=uid_ref[st["donor"]], curve=_strip(c),
                              auc_ins=c["auc_insertion_v1"], auc_del=c["auc_deletion_v1"])
            res.append(row)
        tmp = f"{final}.tmp-{uuid.uuid4().hex[:8]}"
        os.makedirs(tmp)
        with open(os.path.join(tmp, "records.json"), "x", encoding="utf-8") as f:
            json.dump(res, f)
        publish_dir(tmp, final, {"spec_sha256": ssha, "chunk": ch, "seconds": time.perf_counter() - t0})
        print(f"mismatch {ch['chunk']:02d} done {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"MISMATCH STATUS done in {time.perf_counter() - t_all:.1f}s", flush=True)


def phase_audit(cfg, a):
    from . import provenance as P, xai_controls as XK
    from .desc_4b3 import publish_csv, publish_json
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load(out, "spec.json")
    spec["_run_dir"] = out
    _check_inputs(cfg, spec)
    recs = _all_records(out, spec, ssha)
    mm = {r["ref_key"]: r for r in _all_records(out, spec, ssha, prefix=MISMATCH_PREFIX)}
    if [r["ref_key"] for r in recs] != [r["ref_key"] for r in spec["refs"]] or set(mm) != {r["ref_key"] for r in recs}:
        raise RuntimeError("coverage mismatch")
    scores, img = [], []
    for r in recs:
        for e in EXPLAINERS:
            ex = r["explainers"][e]
            pc = ex.get("primary_curve") or {}
            prim = ex["status"] == "primary"
            expl = {"auc_ins": pc.get("auc_insertion_v1") if prim else None, "auc_del": pc.get("auc_deletion_v1") if prim else None}
            ac = ex.get("as_computed_curve") or {}
            img.append({"ref_key": r["ref_key"], "pair_id": r["pair_id"], "label_mel": r["label_mel"], "modality": r["modality"],
                        "explainer": e, "status": ex["status"], "map_flags": ";".join(ex["eligibility"]["map_flags"]),
                        "reasons": ";".join(ex["eligibility"]["ineligibility_reasons"]),
                        "ig_e_rel": r["ig_completeness"]["e_rel"] if e == "ig512" else None,
                        "ig_e_abs": r["ig_completeness"]["e_abs"] if e == "ig512" else None,
                        "endpoint_status": pc.get("endpoint_status") or ac.get("endpoint_status"),
                        "auc_ins_primary": expl["auc_ins"], "auc_del_primary": expl["auc_del"],
                        "auc_ins_as_computed": ac.get("auc_insertion_v1"), "auc_del_as_computed": ac.get("auc_deletion_v1"),
                        "p_full": pc.get("p_full", ac.get("p_full")), "p_base": pc.get("p_base", ac.get("p_base")),
                        "logit_x": r["logit_x"], "logit_baseline": r["logit_baseline"],
                        "pilot_member": r["stage3b_pilot_member"]})
            for cn in CONTROLS:
                cd = mm[r["ref_key"]][e] if cn == "mismatch" else r["controls"][cn]
                sc = XK.control_scores(expl, cd)
                scores.append({"ref_key": r["ref_key"], "pair_id": r["pair_id"], "label_mel": r["label_mel"],
                               "modality": r["modality"], "explainer": e, "explainer_status": ex["status"], "control": cn,
                               "control_status": sc["status"], "reason": sc.get("reason"),
                               "donor_ref_key": cd.get("donor_ref_key") if cn == "mismatch" else None,
                               "explainer_auc_ins": expl["auc_ins"], "explainer_auc_del": expl["auc_del"],
                               "control_auc_ins": sc.get("control_auc_ins"), "control_auc_del": sc.get("control_auc_del"),
                               "delta_ins": sc["insertion_advantage"], "delta_del": sc["deletion_advantage"]})
    # coverage tables
    cov = {}
    for e in EXPLAINERS:
        rows = [x for x in img if x["explainer"] == e]
        g = {}
        for x in rows:
            k = f"{x['modality']}|label={x['label_mel']}"
            gg = g.setdefault(k, {"n": 0, "primary": 0, "reasons": {}})
            gg["n"] += 1
            gg["primary"] += x["status"] == "primary"
            if x["status"] != "primary":
                rk = x["reasons"] or x["status"]
                gg["reasons"][rk] = gg["reasons"].get(rk, 0) + 1
        pairs = {}
        for x in rows:
            pairs.setdefault(x["pair_id"], {})[x["modality"]] = x
        elig_pairs = [p for p, d in pairs.items() if all(d[m]["status"] == "primary" for m in ("dermoscopy", "clinical"))]
        ctl = {}
        for cn in CONTROLS:
            sp = {(s["pair_id"], s["modality"]): s for s in scores if s["explainer"] == e and s["control"] == cn}
            ok = [p for p in elig_pairs if all(sp[(p, m)]["control_status"] == "available" for m in ("dermoscopy", "clinical"))]
            ctl[cn] = {"pairs_available": len(ok),
                       "by_label": {lab: sum(pairs[p]["dermoscopy"]["label_mel"] == lab for p in ok) for lab in ("1", "0")}}
        cov[e] = {"by_modality_class": g, "eligible_pairs": len(elig_pairs),
                  "eligible_pairs_by_label": {lab: sum(pairs[p]["dermoscopy"]["label_mel"] == lab for p in elig_pairs)
                                              for lab in ("1", "0")},
                  "eligible_pairs_pilot_members": sum(pairs[p]["dermoscopy"]["pilot_member"] for p in elig_pairs),
                  "control_pairs": ctl,
                  "endpoint_failures": sum(x["endpoint_status"] == "fail" for x in rows),
                  "as_computed_available": sum(x["auc_ins_as_computed"] is not None for x in rows),
                  "nonfinite_maps": sum("nonfinite" in x["map_flags"] for x in rows)}
    rnd_fail = sum(r["controls"]["random"]["status"] != "available" for r in recs)
    cov["controls_image_level"] = {c: sum(r["controls"][c]["status"] == "available" for r in recs) for c in ("random", "centre", "sobel")}
    cov["random_unavailable_images"] = rnd_fail
    cov["pilot_member_pairs_total"] = sum(r["stage3b_pilot_member"] for r in recs) // 2
    tim = [_read_json(os.path.join(out, f"chunk_{c['chunk']:02d}", "chunk_manifest.json"))
           for c in spec["chunks"]]
    cov["measured"] = {"compute_seconds_sum": sum(t["chunk_seconds"] for t in tim),
                       "timing_sum_s": {k: sum(t["timing_s"][k] for t in tim) for k in tim[0]["timing_s"]},
                       "peak_cuda_mib_max": [max(t["peak_cuda_mib"][0] for t in tim), max(t["peak_cuda_mib"][1] for t in tim)],
                       "mismatch_seconds_sum": sum(_read_json(os.path.join(out, f"{MISMATCH_PREFIX}_{c['chunk']:02d}", "chunk_manifest.json"))["seconds"] for c in spec["chunks"]),
                       "disk_bytes": sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)}
    publish_json(out, "coverage_audit_v2.json", cov)
    publish_csv(out, "image_status_v2.csv", img, list(img[0]))
    publish_csv(out, "scores_v2.csv", scores, list(scores[0]))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "audit", "spec_sha256": ssha})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("coverage_audit_v2.json", "image_status_v2.csv",
                                                                           "scores_v2.csv", "spec.json", "mismatch_assignment.json",
                                                                           AMENDMENT)}
    receipt["outputs"].update({f"{d}/chunk_manifest.json": P.sha256_file(os.path.join(out, d, "chunk_manifest.json"))
                               for d in sorted(os.listdir(out)) if is_result_dir(d) and os.path.isdir(os.path.join(out, d))})
    rp = P.write_receipt(receipt, out, name="receipt_audit.json")
    print(json.dumps(cov, indent=1)[:6000])
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "compute", "mismatch", "audit"))
    ap.add_argument("--out-root")
    ap.add_argument("--run-dir")
    ap.add_argument("--max-seconds", type=float, default=float("inf"))
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
