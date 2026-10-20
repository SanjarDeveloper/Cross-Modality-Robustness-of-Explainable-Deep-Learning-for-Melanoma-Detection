"""Stage 10B: simulated-HAM XAI arm (protocol 02_rev2 §6.7) -- full-arm contract, localisation definitions,
and a bounded numerical/runtime pilot. Seed-0 standard ResNet-50 only. The full arm is NOT run here.

Phases (separate processes; resumable):
  spec      full-arm manifest (170 mel + 170 deterministically selected non-mel HAM TEST images), pilot manifest
            (10 + 10, unique lesions), 11 conditions, chunk plans, mismatch assignment, frozen localisation and
            IG-readiness rules, perturbation/mask identity checks, alignment sheet (masks only; no model)
  compute   per pilot chunk (one condition x 20 images): baseline, Grad-CAM, IG m=512 and m=1024, curves, controls
            (5 random, centre, Sobel), localisation; timed separately; hashed atomic chunk dirs
  mismatch  mismatch-control curves + localisation from the frozen assignment (donor = other lesion, same label,
            same condition, planned donor only)
  analyze   IG convergence/readiness per condition and pooled, localisation availability, cost + projection

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_sim --config ../configs/xai_sim_10b_v1.json --phase spec --out-root ../runs/xai_sim_10b
  ... --phase compute|mismatch|analyze --run-dir <dir>
"""
import argparse
import io
import json
import math
import os
import time
import uuid

import numpy as np

FACTORS = ("blur", "illumination", "compression", "scale", "nuisance")
SEVS = (0.5, 1.0)
N_NONMEL = 170
PILOT_PER_CLASS = 10
CANDIDATES = (512, 1024)              # m and its 2m reference
IG_IB, CAM_B, EVAL_BATCH, S = 16, 8, 11, 32
FULL_CHUNK = 48
SEL_SALT, PILOT_SALT = "rq4_sim_nonmel_v1", "rq4_sim_pilot_v1"
PROB_TOL_S0 = 1e-6
MODEL_SHA = "303d21e71986dc8f26dec84c6822a301f696060e5f08f613c90fc33c41a003dd"
EXPLAINERS = ("gradcam", "ig512", "ig1024")
MISMATCH_EXPLAINERS = ("gradcam", "ig512")


# ------------------------------------------------------------------ pure helpers (tested)
def conditions():
    return [("none", 0.0)] + [(f, s) for f in FACTORS for s in SEVS]


def cond_key(factor, sev):
    return f"{factor}|{sev:.3f}"


def ref_key(record_uid, factor, sev):
    return f"{record_uid}|{cond_key(factor, sev)}"


def select_subset(test_rows, n_nonmel=N_NONMEL, salt=SEL_SALT):
    """All HAM TEST melanomas + the first n_nonmel non-melanomas in stable_seed(salt, record_uid) order.
    Uses identifiers only (no predictions, maps or masks). Repeated images of a lesion are retained."""
    from .seeds import stable_seed
    mel = sorted((r for r in test_rows if r["label_mel"] == "1"), key=lambda r: r["record_uid"])
    non = sorted((r for r in test_rows if r["label_mel"] == "0"), key=lambda r: (stable_seed(salt, r["record_uid"]), r["record_uid"]))
    return mel + non[:n_nonmel]


def select_pilot(subset, n_per_class=PILOT_PER_CLASS, salt=PILOT_SALT):
    """Per class: walk stable_seed(salt, record_uid) order and take the first image of each not-yet-used lesion."""
    from .seeds import stable_seed
    out = []
    for lab in ("1", "0"):
        used = set()
        for r in sorted((r for r in subset if r["label_mel"] == lab), key=lambda r: (stable_seed(salt, r["record_uid"]), r["record_uid"])):
            if r["cluster_id"] in used:
                continue
            used.add(r["cluster_id"])
            out.append(r)
            if len(used) == n_per_class:
                break
    return out


def sim_factor(factor):
    """The shared severity-0 baseline condition 'none' is produced by the simulator's identity (blur at s=0),
    exactly as the Stage 8B/9C2 baseline unit."""
    return "blur" if factor == "none" else factor


def ranking(m):
    """Same deterministic order as the curves: descending value, ties by ascending flat index."""
    flat = np.asarray(m, dtype=np.float64).reshape(-1)
    return np.argsort(-flat, kind="stable")


def mask_status(mask):
    mask = np.asarray(mask, bool)
    if not mask.any():
        return "empty_mask"
    if mask.all():
        return "all_foreground"
    return "ok"


def map_status(m):
    m = np.asarray(m, dtype=np.float64)
    if not np.isfinite(m).all():
        return "map_invalid:nonfinite"
    if (m == 0).all():
        return "map_invalid:all_zero"
    if m.max() - m.min() < 1e-12:
        return "map_invalid:constant"
    return "ok"


def localisation(m, mask, extra_unavailable=None):
    """Frozen definitions (retrospective, Stage 10B):
    iou_area   : top-|M| pixels by the curve ranking vs mask M (area-matched; primary)
    iou_half   : pixels with value >= 0.5*max (ties included) vs M; requires max > 0 (secondary)
    pointing   : peak = first pixel of the ranking (max; ties -> lowest flat index); hit = M[peak]
    Unavailable (never zero) for empty/all-foreground masks, invalid maps or an explicit extra reason."""
    mask = np.asarray(mask, bool)
    out = {"iou_area": None, "iou_half": None, "pointing_hit": None, "n_max_ties": None, "any_max_in_mask": None,
           "mask_fraction": float(mask.mean())}
    ms = mask_status(mask)
    reason = extra_unavailable or (None if ms == "ok" else ms)
    if reason is None:
        st = map_status(m)
        reason = None if st == "ok" else st
    if reason is not None:
        return dict(out, status="unavailable", reason=reason)
    a = np.asarray(m, dtype=np.float64)
    flat_mask = mask.reshape(-1)
    order = ranking(a)
    k = int(flat_mask.sum())
    top = np.zeros(flat_mask.size, bool)
    top[order[:k]] = True
    iou_area = float((top & flat_mask).sum() / (top | flat_mask).sum())
    mx = a.max()
    if mx > 0:
        sel = a.reshape(-1) >= 0.5 * mx
        iou_half = float((sel & flat_mask).sum() / (sel | flat_mask).sum())
    else:
        iou_half = None
    peak = int(order[0])
    ties = np.flatnonzero(a.reshape(-1) == mx)
    return dict(out, status="available", reason=None, iou_area=iou_area, iou_half=iou_half, pointing_hit=bool(flat_mask[peak]),
                n_max_ties=int(ties.size), any_max_in_mask=bool(flat_mask[ties].any()))


def readiness(pooled, per_condition):
    """Frozen before results: READY only if the pooled assessment AND every condition's assessment are 'pass'.
    Any condition 'fail' or 'UNRESOLVED' -> NOT_READY (full arm not launched with this m)."""
    bad = {c: a["overall"] for c, a in per_condition.items() if a["overall"] != "pass"}
    ok = pooled["overall"] == "pass" and not bad
    return {"status": "READY" if ok else "NOT_READY", "pooled": pooled["overall"], "conditions_not_pass": bad}


def planned_counts(n_images, n_conditions):
    refs = n_images * n_conditions
    return {"references": refs, "gradcam_maps": refs, "ig_maps_per_m": refs, "curve_sets": {
        "gradcam": refs, "ig_per_m": refs, "random": 5 * refs, "centre": refs, "sobel": refs, "mismatch": len(MISMATCH_EXPLAINERS) * refs}}


# ------------------------------------------------------------------ io helpers
def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _abs(cfg, rel):
    from . import config as C
    return C.resolve(cfg, rel)


def _masks256(cfg, rows):
    from . import augment as AU, decode as DC
    root = cfg["project_root"]
    out = {}
    for r in rows:
        les, meta = DC.interpret_mask(os.path.join(root, "DATASETS", *r["mask_path"].split("/")))
        out[r["record_uid"]] = AU.mask_to_cache_grid(les)
    return out


def _origs(cfg, rows):
    from . import decode as DC, preprocess as PP
    allrows = DC.read_manifest(_abs(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    reader = PP.EvalReader(allrows, "ham_test", os.path.join(cfg["project_root"], "cache"))
    return {r["record_uid"]: np.asarray(reader.img[int(r["cache_index"])]).copy() for r in rows}


def _condition_inputs(origs, masks, rows, factor, sev):
    """Perturbed model inputs and transformed masks. Images are produced by perturb.apply exactly as in Stage 8B
    (perturbed_inputs); the mask is transformed in the same call with the same geometry."""
    from . import perturb as PT
    xs, ms, metas = [], [], []
    for r in rows:
        img, m2, meta = PT.apply(origs[r["record_uid"]], sim_factor(factor), sev, r["image_uid"], mask256=masks[r["record_uid"]])
        xs.append(PT.to_model_input(img))
        ms.append(PT.mask_to_input(m2))
        metas.append(meta)
    return np.stack(xs), np.stack(ms), metas


# ------------------------------------------------------------------ spec
def phase_spec(cfg, a):
    from . import decode as DC, perturb as PT, provenance as P, xai_attr as XA, xai_controls as XK, xai_pairs as XPa
    from .desc_4b3 import publish_bytes, publish_json
    from .rq2_predict import perturbed_inputs
    rows = DC.read_manifest(_abs(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    test = [r for r in rows if r["split"] == "test"]
    subset = select_subset(test)
    pilot = select_pilot(subset)
    if (sum(r["label_mel"] == "1" for r in subset), sum(r["label_mel"] == "0" for r in subset)) != (170, 170):
        raise RuntimeError("subset counts")
    keep = ("record_uid", "image_uid", "label_mel", "cluster_id", "cache_index", "mask_path")
    full = [{k: r[k] for k in keep} for r in subset]
    pil = [{k: r[k] for k in keep} for r in pilot]
    conds = conditions()
    # mask flags (masks only; not used for selection) + identity checks
    origs = _origs(cfg, subset)
    masks = _masks256(cfg, subset)
    mflags = {u: mask_status(PT.mask_to_input(m)) for u, m in masks.items()}
    ident = {"with_mask_equals_without_mask": True, "n_checked": 0}
    for r in pil:
        for f, s in conds:
            a1 = PT.apply(origs[r["record_uid"]], sim_factor(f), s, r["image_uid"], mask256=masks[r["record_uid"]])[0]
            a2 = PT.apply(origs[r["record_uid"]], sim_factor(f), s, r["image_uid"])[0]
            ident["n_checked"] += 1
            if not np.array_equal(a1, a2):
                ident["with_mask_equals_without_mask"] = False
    xs, _, _ = _condition_inputs(origs, masks, [next(x for x in subset if x["record_uid"] == r["record_uid"]) for r in pil[:3]], "scale", 1.0)
    xr, _ = perturbed_inputs([origs[r["record_uid"]] for r in pil[:3]], [r["image_uid"] for r in pil[:3]], "scale", 1.0)
    ident["model_input_equals_stage8b_path"] = bool(np.array_equal(xs, xr))
    # Stage 8B audit digests (first 4 HAM TEST records per unit) at the arm's severities
    r8b = os.path.dirname(_abs(cfg, cfg["supporting_inputs"]["stage8b_spec"]))
    aud = {"checked": 0, "mismatch": 0}
    t_orig = _origs(cfg, test[:4])
    for f, s in [(f, s) for f in FACTORS for s in (0.0,) + SEVS]:
        z = np.load(os.path.join(r8b, f"u_{f}__{s:.3f}", "audit_images.npz"))
        for i in range(4):
            img = PT.apply(t_orig[test[i]["record_uid"]], f, s, test[i]["image_uid"], mask256=None)[0]
            aud["checked"] += 1
            aud["mismatch"] += (str(z["record_uids"][i]) != test[i]["record_uid"]) or not np.array_equal(img, z["images"][i])
    if aud["mismatch"] or not ident["with_mask_equals_without_mask"] or not ident["model_input_equals_stage8b_path"]:
        raise RuntimeError(f"perturbation identity failed {aud} {ident}")
    # pilot chunk plan: one chunk per condition, frozen image order (pilot order)
    pilot_chunks = [{"chunk": i, "condition": cond_key(f, s), "factor": f, "severity": s, "record_uids": [r["record_uid"] for r in pil]}
                    for i, (f, s) in enumerate(conds)]
    full_chunks = [{"condition": cond_key(f, s), "block": b, "record_uids": [r["record_uid"] for r in full[b * FULL_CHUNK:(b + 1) * FULL_CHUNK]]}
                   for f, s in conds for b in range(math.ceil(len(full) / FULL_CHUNK))]
    mm_recs = [{"record_uid": r["record_uid"], "case_id": r["cluster_id"], "label": r["label_mel"], "modality": "dermoscopy",
                "explainer": e, "condition": cond_key(f, s)} for e in MISMATCH_EXPLAINERS for f, s in conds for r in pil]
    assign = XK.mismatch_assignment(mm_recs)
    srcs = {n: P.sha256_file(os.path.join(os.path.dirname(__file__), n)) for n in
            ("xai_sim.py", "xai_attr.py", "xai_curves.py", "xai_controls.py", "xai_pairs.py", "perturb.py", "preprocess.py", "rq2_predict.py",
             "augment.py", "decode.py")}
    with XA.xai_precision():
        pol = XA.precision_settings()
    sp = cfg["supporting_inputs"]
    spec = {"spec_version": "xai_sim_10b_v1", "created_utc": _now(),
            "model": {"checkpoint": sp["model_checkpoint"], "sha256": MODEL_SHA, "role": "standard ResNet-50 seed 0 (primary)"},
            "cohort_note": "HAM TEST was evaluated in earlier stages (4A, 8B, 9C1/9C2, 9G1/9G2); not an untouched confirmatory cohort. "
                           "Class-enriched 170 + 170 subset; not natural prevalence.",
            "full_arm": {"selection": f"all 170 HAM TEST melanomas (record_uid order) + first {N_NONMEL} non-melanomas in "
                                      f"stable_seed('{SEL_SALT}', record_uid) order; identifiers only (no predictions/maps/masks)",
                         "records": full, "n": len(full), "n_lesions": len({r['cluster_id'] for r in full}),
                         "by_class": {"1": {"images": 170, "lesions": len({r['cluster_id'] for r in full if r['label_mel'] == '1'})},
                                      "0": {"images": 170, "lesions": len({r['cluster_id'] for r in full if r['label_mel'] == '0'})}},
                         "repeated_lesions_retained": True, "chunks": full_chunks, "planned": planned_counts(len(full), len(conds)),
                         "mask_flags": {k: sum(v == k for v in mflags.values()) for k in ("ok", "empty_mask", "all_foreground")}},
            "pilot": {"selection": f"per class, first image of each new lesion in stable_seed('{PILOT_SALT}', record_uid) order, "
                                   f"{PILOT_PER_CLASS} per class; identifiers only; no replacement of difficult/failed examples",
                      "records": pil, "chunks": pilot_chunks, "planned": planned_counts(len(pil), len(conds)),
                      "mismatch_assignment": assign},
            "conditions": [{"factor": f, "severity": s, "key": cond_key(f, s)} for f, s in conds],
            "condition_note": "severity 0 is the identity for every factor -> one shared baseline condition ('none|0.000'); "
                              "factors = the five RQ2 factors (perturb_msim_v1); never chained",
            "simulator": {"version": "perturb_msim_v1", "module_sha256": P.sha256_file(os.path.join(os.path.dirname(__file__), "perturb.py"))},
            "identity_checks": {"mask_path": ident, "stage8b_audit": aud},
            "settings": {"prep": "prep_v1_cached256 -> perturb.apply -> to_model_input (224, INTER_LINEAR)",
                         "precision": pol, "baseline": "attr_v1 blurred baseline per chunk",
                         "gradcam": {"layer": "layer4", "batch": CAM_B}, "ig": {"candidates": CANDIDATES, "rule": "riemann_midpoint",
                                                                                "internal_batch": IG_IB},
                         "curves": {"S": S, "eval_batch": EVAL_BATCH, "endpoint_tol": 1e-4},
                         "controls": "controls_v1: 5 random (stable_seed per image_uid), centre, Sobel; mismatch: fixed derangement "
                                     "within (label, condition, explainer), donor from another lesion; planned donor only",
                         "validity": "attr_v1 validity + IG completeness (§6.6)",
                         "s0_check": f"severity-0 probabilities vs Stage 4A HAM TEST predictions, |dp| <= {PROB_TOL_S0}"},
            "localisation": {"grid": "224x224 model-input grid; mask = perturb.mask_to_input(transformed 256 mask) (INTER_NEAREST_EXACT)",
                             "iou_area_primary": "top-|M| pixels by the curve ranking (desc value, ties by flat index) vs mask",
                             "iou_half_secondary": "pixels >= 0.5*max(map) (ties included) vs mask; requires max > 0",
                             "pointing": "peak = first ranked pixel (ties -> lowest flat index); hit = mask at peak; tie count recorded",
                             "unavailable": "empty mask, all-foreground mask (Addendum A3 primary exclusion), invalid map "
                                            "(non-finite/all-zero/constant), IG completeness failure; never scored as zero",
                             "random_control": "mean over the 5 random maps (all 5 required)",
                             "masks": "photometric factors keep the original mask; nuisance masks = lesion extent (hair/glare "
                                      "occlusion not removed); scale masks follow the inset geometry, zero surround; the synthetic "
                                      "skin surround and feathered seam are image artefacts outside the mask",
                             "status": "retrospective conventions frozen in Stage 10B before any map; the toy-test 0.5*max rule "
                                       "(03_rev2 §13) is a unit-test criterion, not a scientific threshold",
                             "claim_limit": "no localisation claim on Derm7pt (§6.8)"},
            "ig_readiness": {"criteria": {"fail_rate_max": XPa.FAIL_RATE_MAX, "p95_rel_max": XPa.P95_REL_MAX,
                                          "median_abs_dauc_max": XPa.MEDIAN_DAUC_MAX, "comparison": "m=512 vs m=1024 (xai_pairs.assess_candidate)"},
                             "rule": "READY only if the pooled 220-reference assessment AND each of the 11 condition assessments "
                                     "(20 refs each) pass all criteria; otherwise NOT_READY: no full arm, report evidence + next step",
                             "no_selective_m": True, "note": "completeness success alone does not establish map convergence"},
            "analysis_plan_full_arm": {"populations": "per explainer x condition: images with a valid map (IG: completeness) and an "
                                                      "available control; localisation population: available masks and valid maps",
                                       "uncertainty": "lesion-cluster bootstrap (repeated lesions kept with multiplicity)",
                                       "estimands": "raw Δ_ins/Δ_del vs random per condition; changes from severity 0; IoU/pointing "
                                                    "per condition; descriptive"},
            "sources": srcs, "inputs": {k: {"path": v, "sha256": P.sha256_file(_abs(cfg, v))} for k, v in sp.items()
                                        if os.path.isfile(_abs(cfg, v))}}
    from . import provenance as Pv
    receipt = Pv.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "spec.json", spec)
    publish_bytes(out, "spec.json.sha256", f"{P.sha256_file(os.path.join(out, 'spec.json'))}  spec.json\n".encode())
    publish_bytes(out, "alignment_sheet.png", _alignment_sheet(origs, masks, pil))
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("spec.json", "alignment_sheet.png")}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out, spec["full_arm"]["by_class"], spec["full_arm"]["mask_flags"], ident, aud)


def _alignment_sheet(origs, masks, pil):
    """Masks only (no model): 4 pilot images x conditions none, blur 1, nuisance 1, scale 0.5, scale 1 with mask contour."""
    from PIL import Image, ImageDraw
    from . import perturb as PT
    chosen = [r for r in pil if r["label_mel"] == "1"][:2] + [r for r in pil if r["label_mel"] == "0"][:2]
    conds = [("none", 0.0), ("blur", 1.0), ("nuisance", 1.0), ("scale", 0.5), ("scale", 1.0)]
    T = 224
    sheet = Image.new("RGB", (T * len(conds), (T + 16) * len(chosen)), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(chosen):
        for j, (f, s) in enumerate(conds):
            img, m2, _ = PT.apply(origs[r["record_uid"]], sim_factor(f), s, r["image_uid"], mask256=masks[r["record_uid"]])
            import cv2
            im = cv2.resize(img, (T, T), interpolation=cv2.INTER_LINEAR)
            mk = PT.mask_to_input(m2)
            edge = mk & ~(np.roll(mk, 1, 0) & np.roll(mk, -1, 0) & np.roll(mk, 1, 1) & np.roll(mk, -1, 1))
            im = im.copy()
            im[edge] = (0, 255, 0)
            sheet.paste(Image.fromarray(im), (j * T, i * (T + 16) + 16))
            d.text((j * T + 2, i * (T + 16) + 2), f"{r['record_uid'][9:]} y={r['label_mel']} {f} {s}", fill="black")
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


def _load_spec(cfg, out):
    from . import provenance as P, xai_prod as XP
    spec, ssha = XP._load(out, "spec.json")
    for n, h in spec["sources"].items():
        if P.sha256_file(os.path.join(os.path.dirname(__file__), n)) != h:
            raise RuntimeError(f"source changed {n} (resume refused)")
    for k, v in spec["inputs"].items():
        if P.sha256_file(_abs(cfg, v["path"])) != v["sha256"]:
            raise RuntimeError(f"input changed {k}")
    return spec, ssha


def _model(cfg):
    import timm
    from . import trainer as TR
    st = TR._load_verified(_abs(cfg, cfg["supporting_inputs"]["model_checkpoint"]), MODEL_SHA)
    m = timm.create_model("resnet50", pretrained=False, num_classes=1)
    m.load_state_dict(st["model"], strict=True)
    return m.to("cuda").eval()


def _strip(c):
    from .xai_prod import _strip as s
    return s(c)


def phase_compute(cfg, a):
    import torch
    from . import provenance as P, xai_attr as XA, xai_controls as XK, xai_curves as XC, xai_prod as XP
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load_spec(cfg, out)
    XP.quarantine_tmp(out)
    pil = spec["pilot"]["records"]
    full_rows = {r["record_uid"]: r for r in spec["full_arm"]["records"]}
    import csv as _csv
    with open(_abs(cfg, cfg["supporting_inputs"]["stage4a_ham_test_predictions"]), encoding="utf-8", newline="") as f:
        p4a = {r["record_uid"]: float(r["prob"]) for r in _csv.DictReader(f)}
    origs = _origs(cfg, pil)
    masks = _masks256(cfg, pil)
    model = _model(cfg)
    for ch in spec["pilot"]["chunks"]:
        final = os.path.join(out, f"chunk_{ch['chunk']:02d}")
        if XP.verify_dir(final, ssha):
            continue
        rows = [full_rows[u] for u in ch["record_uids"]]
        T = {}

        def tm(name, fn):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = fn()
            torch.cuda.synchronize()
            T[name] = T.get(name, 0.0) + time.perf_counter() - t0
            return r
        s0_ref = _s0_reference(cfg, model, rows) if ch["severity"] == 0.0 else None
        torch.cuda.reset_peak_memory_stats()
        t_chunk = time.perf_counter()
        xn, mk, metas = tm("perturb_and_masks", lambda: _condition_inputs(origs, masks, rows, ch["factor"], ch["severity"]))
        x = torch.from_numpy(xn).to("cuda")
        base = tm("baseline", lambda: XA.gaussian_baseline(x))
        n = len(rows)
        cam = tm("gradcam", lambda: {k: torch.cat([XA.grad_cam(model, model.layer4, x[s:s + CAM_B])[k] for s in range(0, n, CAM_B)])
                                     for k in ("map", "logit")})
        ig = {m: tm(f"ig{m}", lambda m=m: XA.integrated_gradients(model, x, base, m, IG_IB)) for m in CANDIDATES}
        maps = {"gradcam": cam["map"], "ig512": ig[512]["primary"], "ig1024": ig[1024]["primary"]}
        comp = {"ig512": ig[512]["completeness"], "ig1024": ig[1024]["completeness"]}
        recs = []
        for i, r in enumerate(rows):
            rec = {"ref_key": ref_key(r["record_uid"], ch["factor"], ch["severity"]), "record_uid": r["record_uid"], "label_mel": r["label_mel"],
                   "cluster_id": r["cluster_id"], "condition": ch["condition"], "perturb_meta": metas[i],
                   "baseline_sha256": XP._sha_bytes(base[i].detach().cpu().numpy().tobytes()),
                   "logit_x": float(ig[512]["logit"][i]), "logit_baseline": float(ig[512]["logit_baseline"][i]),
                   "logit_x_1024_run": float(ig[1024]["logit"][i]), "mask": {"status": mask_status(mk[i]), "fraction": float(mk[i].mean())},
                   "ig_completeness": {"ig512": comp["ig512"][i], "ig1024": comp["ig1024"][i]}, "explainers": {}, "controls": {}, "localisation": {}}
            p = 1.0 / (1.0 + math.exp(-rec["logit_x"]))
            if ch["severity"] == 0.0:
                q = s0_ref[r["record_uid"]]
                rec["s0_check"] = {"prob_4a": p4a[r["record_uid"]], "prob_4a_batch_composition": q,
                                   "abs_diff": abs(q - p4a[r["record_uid"]]),
                                   "xai_batch_prob": p, "xai_batch_abs_diff_descriptive": abs(p - p4a[r["record_uid"]])}
                if rec["s0_check"]["abs_diff"] > PROB_TOL_S0:
                    raise RuntimeError(f"severity-0 probability differs from Stage 4A {r['record_uid']}")
            for e in EXPLAINERS:
                mp = maps[e][i]
                el = XA.eligibility(XA.validity(mp), None if e == "gradcam" else comp[e][i])
                d = {"eligibility": el, "map_sha256": XP._sha_bytes(mp.detach().cpu().numpy().tobytes())}
                try:
                    cv = tm("curves_explainers", lambda: XC.analysis_curves(model, x[i], base[i], mp, el, "gradcam" if e == "gradcam" else "ig",
                                                                             eval_batch=EVAL_BATCH, s=S))
                    d["primary_curve"], d["status"] = _strip(cv), cv["analysis_status"]
                except XC.IneligibleMap as ex:
                    d["status"], d["ineligible_reason"] = "ineligible", str(ex)
                rec["explainers"][e] = d
                extra = None if el["eligible_for_primary_analysis"] else ";".join(el["ineligibility_reasons"])
                rec["localisation"][e] = tm("localisation", lambda: localisation(mp.detach().cpu().numpy(), mk[i], extra))
            rnd = XK.random_maps(r["image_uid"])
            rc = tm("curves_random", lambda: [XC.curves(model, x[i], base[i], torch.as_tensor(m_, dtype=torch.float64, device="cuda"),
                                                        s=S, eval_batch=EVAL_BATCH) for m_ in rnd])
            ok = all(c["curve_status"] == "ok" and c["endpoint_status"] == "pass" for c in rc)
            rec["controls"]["random"] = {"status": "available" if ok else "unavailable", "curves": [_strip(c) for c in rc],
                                         "auc_ins_each": [c["auc_insertion_v1"] for c in rc], "auc_del_each": [c["auc_deletion_v1"] for c in rc]}
            rl = tm("localisation", lambda: [localisation(m_, mk[i]) for m_ in rnd])
            rec["localisation"]["random_mean"] = ({"status": "available", **{k: float(np.mean([z[k] for z in rl])) for k in ("iou_area", "pointing_hit")}}
                                                  if all(z["status"] == "available" for z in rl) else {"status": "unavailable", "reason": rl[0].get("reason")})
            for name, mp in (("centre", XK.centre_map()), ("sobel", XK.sobel_map(x[i]))):
                v, flags = XK.map_valid(mp)
                if not v:
                    rec["controls"][name] = {"status": "unavailable", "reason": f"{name} map invalid: {flags}"}
                    rec["localisation"][name] = {"status": "unavailable", "reason": "map_invalid"}
                    continue
                c = tm(f"curves_{name}", lambda: XC.curves(model, x[i], base[i], torch.as_tensor(np.asarray(mp), dtype=torch.float64, device="cuda"),
                                                           s=S, eval_batch=EVAL_BATCH))
                good = c["curve_status"] == "ok" and c["endpoint_status"] == "pass"
                rec["controls"][name] = {"status": "available" if good else "unavailable", "curve": _strip(c),
                                         "auc_ins": c["auc_insertion_v1"], "auc_del": c["auc_deletion_v1"]}
                rec["localisation"][name] = tm("localisation", lambda: localisation(np.asarray(mp), mk[i]))
            recs.append(rec)
        tmp = f"{final}.tmp-{uuid.uuid4().hex[:8]}"
        os.makedirs(tmp)
        buf = io.BytesIO()
        np.savez_compressed(buf, ref_keys=np.array([r["ref_key"] for r in recs]), masks=mk,
                            gradcam=cam["map"].cpu().numpy().astype(np.float32),
                            ig512_primary=ig[512]["primary"].cpu().numpy().astype(np.float32),
                            ig1024_primary=ig[1024]["primary"].cpu().numpy().astype(np.float32),
                            ig512_attr_signed=ig[512]["attr"].cpu().numpy().astype(np.float32))
        with open(os.path.join(tmp, "maps.npz"), "xb") as f:
            f.write(buf.getvalue())
        with open(os.path.join(tmp, "records.json"), "x", encoding="utf-8") as f:
            json.dump(recs, f)
        meta = {"spec_sha256": ssha, "chunk": ch["chunk"], "condition": ch["condition"], "timing_s": T,
                "chunk_seconds": time.perf_counter() - t_chunk,
                "peak_cuda_mib": [round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)]}
        XP.publish_dir(tmp, final, meta)
        print(f"chunk {ch['chunk']:02d} {ch['condition']} {meta['chunk_seconds']:.1f}s {json.dumps({k: round(v, 1) for k, v in T.items()})}", flush=True)
    done = sum(XP.verify_dir(os.path.join(out, f"chunk_{c['chunk']:02d}"), ssha) for c in spec["pilot"]["chunks"])
    print(f"COMPUTE STATUS {done}/{len(spec['pilot']['chunks'])}", flush=True)


def _s0_reference(cfg, model, rows):
    """Severity-0 identity check in the Stage 4A batch composition (HAM TEST manifest order, batch 16, xai_precision):
    GPU outputs are not bitwise batch-invariant (Stage 2D1), so the check must reproduce 4A's composition. The XAI
    chunk's own batch-of-20 probabilities are recorded descriptively. (Stage 10B attempt 1 compared across
    compositions and stopped at 1.01e-6 > 1e-6; see runs/stage10b/diag_s0_check.json.)"""
    import torch
    from . import decode as DC, perturb as PT, xai_attr as XA
    allrows = DC.read_manifest(_abs(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    test = [r for r in allrows if r["split"] == "test"]
    idx = {r["record_uid"]: i for i, r in enumerate(test)}
    out = {}
    for r in rows:
        b0 = (idx[r["record_uid"]] // 16) * 16
        batch = test[b0:b0 + 16]
        og = _origs(cfg, batch)
        xb = torch.from_numpy(np.stack([PT.to_model_input(og[t["record_uid"]]) for t in batch])).to("cuda")
        with XA.xai_precision(), torch.no_grad():
            pr = torch.sigmoid(XA._logit(model, xb).float()).cpu().numpy()
        out[r["record_uid"]] = float(pr[idx[r["record_uid"]] - b0])
    return out


def _records(out, spec, ssha, prefix="chunk"):
    from . import xai_prod as XP
    recs = []
    for ch in spec["pilot"]["chunks"]:
        d = os.path.join(out, f"{prefix}_{ch['chunk']:02d}")
        if not XP.verify_dir(d, ssha):
            raise RuntimeError(f"{d} incomplete")
        recs += XP._read_json(os.path.join(d, "records.json"))
    return recs


def phase_mismatch(cfg, a):
    import torch
    from . import xai_attr as XA, xai_controls as XK, xai_curves as XC, xai_prod as XP
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load_spec(cfg, out)
    XP.quarantine_tmp(out)
    assign = spec["pilot"]["mismatch_assignment"]
    full_rows = {r["record_uid"]: r for r in spec["full_arm"]["records"]}
    origs = _origs(cfg, spec["pilot"]["records"])
    masks = _masks256(cfg, spec["pilot"]["records"])
    model = _model(cfg)
    for ch in spec["pilot"]["chunks"]:
        final = os.path.join(out, f"mismatch_{ch['chunk']:02d}")
        if XP.verify_dir(final, ssha):
            continue
        src = os.path.join(out, f"chunk_{ch['chunk']:02d}")
        if not XP.verify_dir(src, ssha):
            raise RuntimeError(f"{src} incomplete")
        recs = XP._read_json(os.path.join(src, "records.json"))
        z = np.load(os.path.join(src, "maps.npz"))
        rows = [full_rows[u] for u in ch["record_uids"]]
        t0 = time.perf_counter()
        xn, mk, _ = _condition_inputs(origs, masks, rows, ch["factor"], ch["severity"])
        x = torch.from_numpy(xn).to("cuda")
        base = XA.gaussian_baseline(x)
        tc = tl = 0.0
        res = []
        for i, r in enumerate(rows):
            if XP._sha_bytes(base[i].detach().cpu().numpy().tobytes()) != recs[i]["baseline_sha256"]:
                raise RuntimeError("baseline differs from compute phase")
            row = {"ref_key": recs[i]["ref_key"]}
            for e in MISMATCH_EXPLAINERS:
                arr = z["gradcam" if e == "gradcam" else "ig512_primary"]
                dmaps = {rr["record_uid"]: arr[j] for j, rr in enumerate(rows)}
                dok = {rr["record_uid"]: XP.donor_eligible(recs[j]["explainers"][e]) for j, rr in enumerate(rows)}
                key = XK.assignment_key(r["record_uid"], "dermoscopy", e, ch["condition"])
                dm, st = XK.resolve_mismatch(key, assign, dmaps, dok)
                if dm is None:
                    row[e] = dict(st)
                    continue
                ts = time.perf_counter()
                c = XC.curves(model, x[i], base[i], torch.as_tensor(dm, device="cuda"), s=S, eval_batch=EVAL_BATCH)
                tc += time.perf_counter() - ts
                ts = time.perf_counter()
                loc = localisation(dm, mk[i])
                tl += time.perf_counter() - ts
                good = c["curve_status"] == "ok" and c["endpoint_status"] == "pass"
                row[e] = dict(st, status="available" if good else "unavailable", curve=_strip(c), auc_ins=c["auc_insertion_v1"],
                              auc_del=c["auc_deletion_v1"], localisation=loc, donor_cluster=full_rows[st["donor"]]["cluster_id"])
            res.append(row)
        tmp = f"{final}.tmp-{uuid.uuid4().hex[:8]}"
        os.makedirs(tmp)
        with open(os.path.join(tmp, "records.json"), "x", encoding="utf-8") as f:
            json.dump(res, f)
        XP.publish_dir(tmp, final, {"spec_sha256": ssha, "chunk": ch["chunk"], "seconds": time.perf_counter() - t0,
                                    "curve_seconds": tc, "localisation_seconds": tl})
        print(f"mismatch {ch['chunk']:02d} {time.perf_counter() - t0:.1f}s", flush=True)


def _ig_rec(r, m):
    e = r["explainers"][f"ig{m}"]
    pc = e.get("primary_curve") or {}
    c = r["ig_completeness"][f"ig{m}"]
    return {"completeness_failure": c["failure"], "e_rel": c["e_rel"], "e_abs": c["e_abs"], "eligible": e["status"] == "primary",
            "curve_status": pc.get("curve_status"), "endpoint_status": pc.get("endpoint_status"),
            "auc_ins": pc.get("auc_insertion_v1"), "auc_del": pc.get("auc_deletion_v1"), "map_flags": e["eligibility"]["map_flags"],
            "reasons": e["eligibility"]["ineligibility_reasons"]}


def phase_analyze(cfg, a):
    from . import provenance as P, xai_pairs as XPa, xai_prod as XP
    from .desc_4b3 import publish_csv, publish_json
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load_spec(cfg, out)
    recs = _records(out, spec, ssha)
    mm = {r["ref_key"]: r for r in _records(out, spec, ssha, "mismatch")}
    keys = [r["ref_key"] for r in recs]
    if len(keys) != len(set(keys)) or len(keys) != 220 or set(mm) != set(keys):
        raise RuntimeError("coverage")
    at = {m: {r["ref_key"]: _ig_rec(r, m) for r in recs} for m in CANDIDATES}

    def assess(sel):
        ks = [r["ref_key"] for r in recs if sel(r)]
        A = XPa.assess_candidate({k: at[512][k] for k in ks}, {k: at[1024][k] for k in ks}, ks)
        return A
    pooled = assess(lambda r: True)
    per_cond = {c["key"]: assess(lambda r, c=c: r["condition"] == c["key"]) for c in spec["conditions"]}
    ready = readiness(pooled, per_cond)
    conv_rows = []
    for name, A, sel in [("pooled", pooled, lambda r: True)] + [(k, v, (lambda r, k=k: r["condition"] == k)) for k, v in per_cond.items()] + \
            [(f"class={lab}", assess(lambda r, lab=lab: r["label_mel"] == lab), (lambda r, lab=lab: r["label_mel"] == lab)) for lab in ("1", "0")]:
        rr = [r for r in recs if sel(r)]
        e512 = [r["ig_completeness"]["ig512"]["e_rel"] for r in rr if r["ig_completeness"]["ig512"]["e_rel"] is not None]
        conv_rows.append({"group": name, "n_refs": A["n_refs"], "fail_512": A["a"]["failures"], "fail_rate_512": A["a"]["rate"],
                          "fail_1024": sum(r["ig_completeness"]["ig1024"]["failure"] for r in rr),
                          "fail_512_mel": sum(r["ig_completeness"]["ig512"]["failure"] for r in rr if r["label_mel"] == "1"),
                          "fail_512_nonmel": sum(r["ig_completeness"]["ig512"]["failure"] for r in rr if r["label_mel"] == "0"),
                          "e_rel_512_median": float(np.median(e512)) if e512 else None, "e_rel_512_p95": A["b"].get("p95"),
                          "e_rel_512_max": float(max(e512)) if e512 else None, "e_rel_undefined": A["b"]["n_undefined"],
                          "n_eligible_both": A["c"]["n_eligible_both"],
                          "median_abs_dauc_ins": A["c"]["ins"]["median_abs"], "median_abs_dauc_del": A["c"]["del"]["median_abs"],
                          "max_abs_dauc_ins": A["c"]["ins"]["max_abs"], "max_abs_dauc_del": A["c"]["del"]["max_abs"],
                          "status_a": A["a"]["status"], "status_b": A["b"]["status"], "status_c_ins": A["c"]["ins"]["status"],
                          "status_c_del": A["c"]["del"]["status"], "overall": A["overall"]})
    # localisation + availability
    loc_rows = []
    for c in spec["conditions"]:
        rr = [r for r in recs if r["condition"] == c["key"]]
        for e in EXPLAINERS + ("random_mean", "centre", "sobel"):
            L = [r["localisation"][e] for r in rr]
            av = [z for z in L if z["status"] == "available"]
            reasons = {}
            for z in L:
                if z["status"] != "available":
                    reasons[z.get("reason")] = reasons.get(z.get("reason"), 0) + 1
            loc_rows.append({"condition": c["key"], "map": e, "n": len(L), "available": len(av), "unavailable_reasons": json.dumps(reasons),
                             "iou_area_median": float(np.median([z["iou_area"] for z in av])) if av else None,
                             "pointing_rate": float(np.mean([z["pointing_hit"] for z in av])) if av else None})
        for e in MISMATCH_EXPLAINERS:
            L = [mm[r["ref_key"]][e] for r in rr]
            av = [z["localisation"] for z in L if z.get("status") == "available" and z["localisation"]["status"] == "available"]
            loc_rows.append({"condition": c["key"], "map": f"mismatch_{e}", "n": len(L), "available": len(av),
                             "unavailable_reasons": json.dumps({z.get("reason"): 1 for z in L if z.get("status") != "available"}),
                             "iou_area_median": float(np.median([z["iou_area"] for z in av])) if av else None,
                             "pointing_rate": float(np.mean([z["pointing_hit"] for z in av])) if av else None})
    s0 = [r["s0_check"]["abs_diff"] for r in recs if "s0_check" in r]
    endpoint_fail = sum((r["explainers"][e].get("primary_curve") or {}).get("endpoint_status") == "fail" for r in recs for e in EXPLAINERS)
    # cost
    man = [XP._read_json(os.path.join(out, f"chunk_{c['chunk']:02d}", "chunk_manifest.json")) for c in spec["pilot"]["chunks"]]
    mman = [XP._read_json(os.path.join(out, f"mismatch_{c['chunk']:02d}", "chunk_manifest.json")) for c in spec["pilot"]["chunks"]]
    tsum = {}
    for m_ in man:
        for k, v in m_["timing_s"].items():
            tsum[k] = tsum.get(k, 0.0) + v
    n = len(recs)
    per_ref = {k: v / n for k, v in tsum.items()}
    per_ref["mismatch_curves"] = sum(m_["curve_seconds"] for m_ in mman) / n
    per_ref["mismatch_localisation"] = sum(m_["localisation_seconds"] for m_ in mman) / n
    disk = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)
    full_refs = spec["full_arm"]["planned"]["references"]
    common = sum(v for k, v in per_ref.items() if not k.startswith("ig"))
    proj = {"basis": "measured pilot seconds per reference x full-arm references (3740); ESTIMATE; same GPU, no competing load",
            "full_refs": full_refs,
            "hours_if_m512_only": (common + per_ref["ig512"]) * full_refs / 3600,
            "hours_if_m1024_only": (common + per_ref["ig1024"]) * full_refs / 3600,
            "hours_if_both_512_and_1024": (common + per_ref["ig512"] + per_ref["ig1024"]) * full_refs / 3600,
            "note": "curves_explainers in the pilot include 3 explainers (Grad-CAM, IG512, IG1024); a single-m arm would need ~2/3 of "
                    "that term; the projection keeps the measured value (conservative)",
            "disk_gb_estimate": disk / n * full_refs / 1e9}
    res = {"readiness": ready, "pooled_assessment": pooled, "per_condition_assessment": per_cond,
           "s0_check": {"n": len(s0), "max_abs_prob_diff_vs_4a": max(s0)}, "explainer_endpoint_failures": endpoint_fail,
           "measured": {"seconds_by_component_total": tsum, "seconds_per_reference": per_ref,
                        "chunk_seconds_total": sum(m_["chunk_seconds"] for m_ in man), "mismatch_seconds_total": sum(m_["seconds"] for m_ in mman),
                        "peak_cuda_mib_max": [max(m_["peak_cuda_mib"][0] for m_ in man), max(m_["peak_cuda_mib"][1] for m_ in man)],
                        "disk_bytes": disk},
           "projection_full_arm": proj}
    publish_json(out, "results.json", res)
    publish_csv(out, "ig_convergence_table.csv", conv_rows, list(conv_rows[0]))
    publish_csv(out, "localisation_pilot_table.csv", loc_rows, list(loc_rows[0]))
    img = [{"ref_key": r["ref_key"], "condition": r["condition"], "label_mel": r["label_mel"], "cluster_id": r["cluster_id"],
            **{f"{e}_status": r["explainers"][e]["status"] for e in EXPLAINERS},
            "e_rel_512": r["ig_completeness"]["ig512"]["e_rel"], "e_abs_512": r["ig_completeness"]["ig512"]["e_abs"],
            "e_rel_1024": r["ig_completeness"]["ig1024"]["e_rel"], "e_abs_1024": r["ig_completeness"]["ig1024"]["e_abs"],
            "auc_ins_512": at[512][r["ref_key"]]["auc_ins"], "auc_ins_1024": at[1024][r["ref_key"]]["auc_ins"],
            "auc_del_512": at[512][r["ref_key"]]["auc_del"], "auc_del_1024": at[1024][r["ref_key"]]["auc_del"],
            **{f"iou_{e}": r["localisation"][e].get("iou_area") for e in EXPLAINERS},
            **{f"point_{e}": r["localisation"][e].get("pointing_hit") for e in EXPLAINERS}} for r in recs]
    publish_csv(out, "per_reference_pilot.csv", img, list(img[0]))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze"})
    receipt["outputs"] = {n_: P.sha256_file(os.path.join(out, n_)) for n_ in sorted(os.listdir(out))
                          if os.path.isfile(os.path.join(out, n_)) and not n_.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    print(json.dumps({"readiness": ready, "s0": res["s0_check"], "per_ref": per_ref, "proj": proj}, indent=1, default=str))
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "compute", "mismatch", "analyze"))
    ap.add_argument("--out-root")
    ap.add_argument("--run-dir")
    a = ap.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from . import config as C
    from .seeds import seed_everything
    cfg = C.load(a.config)
    if cfg["purpose"] != "xai_diag":
        raise C.ConfigError("requires purpose 'xai_diag'")
    seed_everything(0, cfg["determinism"])
    {"spec": phase_spec, "compute": phase_compute, "mismatch": phase_mismatch, "analyze": phase_analyze}[a.phase](cfg, a)


if __name__ == "__main__":
    main()
