"""Stage 10A: secondary IG map variants on the frozen Stage 7A Derm7pt XAI cohort (seed-0 standard ResNet-50).

Variants (protocol 02_rev2 Table "Secondary attribution"; plan 03_rev2 map_type ig_abs / ig_pos_smooth):
  ig_abs        : sum over RGB of |signed attribution|                    (NOT |sum over RGB|)
  ig_pos_smooth : Gaussian(sigma=2 px) applied AFTER ReLU(sum over RGB of signed attribution)

Frozen retrospective conventions (not specified in earlier documents; fixed here before any curve):
  kernel  : sampled Gaussian exp(-x^2/(2 sigma^2)), radius = int(4*sigma + 0.5) = 8 (17 taps), normalised to sum 1,
            float64, separable (rows then columns)
  padding : symmetric half-sample reflection (numpy 'symmetric' == scipy.ndimage mode 'reflect')
  dtype   : maps computed and ranked in float64 from the stored float32 Stage 7A arrays
  sources : ig_abs = stored ig512_absolute; ig_pos_smooth = smooth(stored ig512_primary). Both stored maps are first
            validated against reconstructions from the stored signed attribution (relative tolerance below).

Phases: spec -> compute (per Stage 7A chunk; same membership; baseline recomputed and hash-checked) -> analyze.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_secondary --config ../configs/xai_secondary_10a_v1.json --phase spec --out-root ../runs/xai_secondary_10a
  ... --phase compute|analyze --run-dir <dir>
"""
import argparse
import io
import json
import math
import os
import time
import uuid

import numpy as np

R7A = "research_v1/runs/xai_prod_7a/20261003T173258Z_3b655f88_548b3a05/"
R7B_DIR = "research_v1/runs/rq4_stats_7b/"
VARIANTS = ("ig_abs", "ig_pos_smooth")
SIGMA, TRUNCATE = 2.0, 4.0
RECON_REL_TOL = 1e-5          # max|recon - stored| <= RECON_REL_TOL * max|stored|  (float32 GPU vs float64 CPU summation)
RANDOM_P_TOL = 1e-6           # new-curve p_full/p_base vs Stage 7A random-curve p_full/p_base (identity of x, b, model)
B = 2000
SEEDS = {"ig_abs_own": 20261100, "ig_pos_smooth_own": 20261101, "ig_abs_vs_orig": 20261102, "ig_pos_smooth_vs_orig": 20261103}
MODS = ("dermoscopy", "clinical")
CURVES = ("ins", "del")


# ------------------------------------------------------------------ pure helpers (tested)
def gaussian_kernel(sigma=SIGMA, truncate=TRUNCATE):
    r = int(truncate * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def smooth(m, sigma=SIGMA, truncate=TRUNCATE):
    """Separable Gaussian, symmetric (half-sample) reflection padding, float64. Input is not modified."""
    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("expected (H, W)")
    k = gaussian_kernel(sigma, truncate)
    r = (len(k) - 1) // 2
    p = np.pad(a, ((0, 0), (r, r)), mode="symmetric")
    a = sum(k[i] * p[:, i:i + a.shape[1]] for i in range(len(k)))
    p = np.pad(a, ((r, r), (0, 0)), mode="symmetric")
    return sum(k[i] * p[i:i + a.shape[0], :] for i in range(len(k)))


def pos_map(attr):
    """Primary: ReLU(sum over RGB of signed attribution); attr (3, H, W)."""
    return np.maximum(np.asarray(attr, dtype=np.float64).sum(0), 0.0)


def abs_map(attr):
    """Secondary A: sum over RGB of |signed attribution| (NOT |sum over RGB|)."""
    return np.abs(np.asarray(attr, dtype=np.float64)).sum(0)


def pos_smooth_map(attr):
    """Secondary B: smoothing applied AFTER the primary transformation."""
    return smooth(pos_map(attr))


def recon_ok(recon, stored, tol=RECON_REL_TOL):
    d = float(np.max(np.abs(np.asarray(recon, np.float64) - np.asarray(stored, np.float64))))
    scale = float(np.max(np.abs(stored)))
    return (d <= tol * scale if scale > 0 else d == 0.0), d


def eligible_secondary(completeness_failure, validity_flags):
    """Original IG completeness must pass (a map transformation does not repair it) and the variant map must be
    finite and non-constant."""
    reasons = ([] if not completeness_failure else ["ig_completeness_failure"]) + [f"map:{f}" for f in validity_flags]
    return not reasons, reasons


def pair_population(rows):
    """rows: per-image {pair_id, modality, ok, ins, del}. Pairs with both modalities ok and finite deltas."""
    by = {}
    for r in rows:
        if r["modality"] in by.setdefault(r["pair_id"], {}):
            raise ValueError("duplicate modality")
        by[r["pair_id"]][r["modality"]] = r
    out = {}
    for p, d in by.items():
        if set(d) != set(MODS):
            raise ValueError(f"incomplete pair {p}")
        if all(d[m]["ok"] and d[m]["ins"] is not None and d[m]["del"] is not None
               and math.isfinite(d[m]["ins"]) and math.isfinite(d[m]["del"]) for m in MODS):
            out[p] = d
    return out


def contrast_vectors(orig, var, ids):
    """Per-pair vectors on the SAME pair ids: orig/variant per modality and D = clinical - dermoscopy,
    and variant - original (direction: variant minus original)."""
    v = {}
    for c in CURVES:
        for m in MODS:
            v[f"orig|{c}|{m}"] = np.array([orig[p][m][c] for p in ids])
            v[f"var|{c}|{m}"] = np.array([var[p][m][c] for p in ids])
            v[f"var_minus_orig|{c}|{m}"] = v[f"var|{c}|{m}"] - v[f"orig|{c}|{m}"]
        v[f"orig|{c}|D"] = v[f"orig|{c}|clinical"] - v[f"orig|{c}|dermoscopy"]
        v[f"var|{c}|D"] = v[f"var|{c}|clinical"] - v[f"var|{c}|dermoscopy"]
        v[f"var_minus_orig|{c}|D"] = v[f"var|{c}|D"] - v[f"orig|{c}|D"]
    return v


# ------------------------------------------------------------------ phases
def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _abs(cfg, rel):
    from . import config as C
    return C.resolve(cfg, rel)


def _r7b(cfg):
    import glob
    d = sorted(glob.glob(os.path.join(_abs(cfg, R7B_DIR), "*")))
    d = [x for x in d if os.path.isdir(x)]
    if len(d) != 1:
        raise RuntimeError("expected exactly one Stage 7B run")
    return d[0]


def phase_spec(cfg, a):
    from . import provenance as P, xai_prod as XP
    from .desc_4b3 import publish_bytes, publish_json
    run = _abs(cfg, R7A)
    rec = XP._read_json(os.path.join(run, "receipt_audit.json"))
    for n, h in rec["outputs"].items():
        if P.sha256_file(os.path.join(run, n)) != h:
            raise RuntimeError(f"Stage 7A output changed {n}")
    s7, s7sha = XP._load(run, "spec.json")
    chunk_files = {}
    for ch in s7["chunks"]:
        d = f"chunk_{ch['chunk']:02d}"
        if not XP.verify_dir(os.path.join(run, d), s7sha):
            raise RuntimeError(f"{d} does not verify")
        chunk_files[d] = XP._read_json(os.path.join(run, d, "chunk_manifest.json"))["files"]
    r7b = _r7b(cfg)
    r7b_rec = XP._read_json(os.path.join(r7b, "receipt_analyze.json"))
    srcs = {n: P.sha256_file(os.path.join(os.path.dirname(__file__), n)) for n in
            ("xai_secondary.py", "xai_attr.py", "xai_curves.py", "xai_controls.py", "xai_prod.py", "rq4_stats.py", "rq4_analysis.py",
             "torch_data.py", "preprocess.py")}
    spec = {"spec_version": "xai_secondary_10a_v1", "created_utc": _now(),
            "model": s7["model"], "stage7a": {"run": R7A, "spec_sha256": s7sha, "audit_receipt_outputs": rec["outputs"],
                                              "chunk_files": chunk_files},
            "stage7b": {"run": os.path.relpath(r7b, cfg["project_root"]).replace(os.sep, "/"), "analyze_receipt_outputs": r7b_rec["outputs"]},
            "cohort": "frozen 498 pairs / 996 references (Stage 7A order and chunk membership)",
            "variants": {"ig_abs": "sum_RGB |signed attribution| (stored ig512_absolute; validated against reconstruction)",
                         "ig_pos_smooth": "Gaussian(sigma=2 px) applied after ReLU(sum_RGB signed attribution) (stored ig512_primary "
                                          "validated against reconstruction, then smoothed)"},
            "smoothing": {"sigma_px": SIGMA, "truncate": TRUNCATE, "radius": int(TRUNCATE * SIGMA + 0.5), "taps": 2 * int(TRUNCATE * SIGMA + 0.5) + 1,
                          "kernel": "sampled exp(-x^2/(2 sigma^2)), normalised to 1, separable rows->cols",
                          "padding": "symmetric half-sample reflection (numpy 'symmetric' = scipy.ndimage 'reflect')",
                          "dtype": "float64", "status": "retrospective implementation conventions frozen in Stage 10A before any curve"},
            "reconstruction_tolerance": {"relative_to_stored_max": RECON_REL_TOL},
            "curves": {"S": XP.S, "eval_batch": XP.EVAL_BATCH, "endpoint_tol": 1e-4, "target": "melanoma probability (sigmoid of logit)",
                       "baseline": "attr_v1 blurred baseline recomputed per Stage 7A chunk; SHA-256 must equal the Stage 7A record",
                       "precision": "xai_precision_v1", "map_dtype": "float64"},
            "eligibility": "secondary: original IG completeness pass (Stage 7A record; never repaired) AND variant map finite and "
                           "non-constant; pair requires both modalities. Primary eligibility unchanged. Endpoint failure -> unavailable.",
            "controls": {"random": "reused Stage 7A random curves (5 maps per image, same model/x/baseline/S/eval_batch/precision); "
                                   "identity check: new-curve p_full and p_base equal the 7A random-curve values within "
                                   f"{RANDOM_P_TOL}; mean of the same 5 AUCs per image (controls_v1 control_scores)",
                         "not_used": "centre, Sobel, mismatch (mismatch curves belong to primary maps and are not relabelled); "
                                     "deferred for secondary variants"},
            "analysis": {"per_image": "delta_ins = AUC_ins(variant) - mean random AUC_ins; delta_del = mean random AUC_del - AUC_del(variant)",
                         "own_population": "pairs eligible for the variant in both modalities",
                         "comparison_population": "intersection: variant AND original primary IG eligible in both modalities",
                         "statistics": "sample median, Hodges-Lehmann (rq4_stats), paired case bootstrap B=2000 with ONE shared draw "
                                       "per replicate across all vectors (pairs keep both modalities and both maps)",
                         "direction": "variant minus original; D = clinical minus dermoscopy",
                         "seeds": SEEDS, "tests": "none (descriptive); primary 12-test Holm family unchanged"},
            "sources": srcs}
    from . import provenance as Pv
    receipt = Pv.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "spec.json", spec)
    publish_bytes(out, "spec.json.sha256", f"{P.sha256_file(os.path.join(out, 'spec.json'))}  spec.json\n".encode())
    receipt["outputs"] = {"spec.json": P.sha256_file(os.path.join(out, "spec.json"))}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out)


def _load_spec(cfg, out):
    from . import provenance as P, xai_prod as XP
    spec, ssha = XP._load(out, "spec.json")
    for n, h in spec["sources"].items():
        if P.sha256_file(os.path.join(os.path.dirname(__file__), n)) != h:
            raise RuntimeError(f"source changed {n} (resume refused)")
    run = _abs(cfg, R7A)
    for n, h in spec["stage7a"]["audit_receipt_outputs"].items():
        if P.sha256_file(os.path.join(run, n)) != h:
            raise RuntimeError(f"Stage 7A output changed {n}")
    return spec, ssha


def phase_compute(cfg, a):
    import torch
    from . import provenance as P, xai_attr as XA, xai_curves as XC, xai_prod as XP
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load_spec(cfg, out)
    run = _abs(cfg, R7A)
    s7, s7sha = XP._load(run, "spec.json")
    XP.quarantine_tmp(out)
    model = XP._model(cfg)
    t_all = time.perf_counter()
    for ch in s7["chunks"]:
        final = os.path.join(out, f"var_{ch['chunk']:02d}")
        if XP.verify_dir(final, ssha):
            continue
        src = os.path.join(run, f"chunk_{ch['chunk']:02d}")
        if not XP.verify_dir(src, s7sha):
            raise RuntimeError(f"{src} does not verify")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        rr = s7["refs"][ch["start"]:ch["end"]]
        recs7 = XP._read_json(os.path.join(src, "records.json"))
        z = np.load(os.path.join(src, "maps.npz"))
        if [str(k) for k in z["ref_keys"]] != [r["ref_key"] for r in rr] or [r["ref_key"] for r in recs7] != [r["ref_key"] for r in rr]:
            raise RuntimeError("chunk identity")
        x = XP._inputs(cfg, rr)
        base = XA.gaussian_baseline(x)
        res, smooth_maps = [], []
        tc = 0.0
        for i, r in enumerate(rr):
            r7 = recs7[i]
            if XP._sha_bytes(base[i].detach().cpu().numpy().tobytes()) != r7["baseline_sha256"]:
                raise RuntimeError(f"baseline differs from Stage 7A for {r['ref_key']}")
            attr = z["ig512_attr_signed"][i]
            ok_p, d_p = recon_ok(pos_map(attr), z["ig512_primary"][i])
            ok_a, d_a = recon_ok(abs_map(attr), z["ig512_absolute"][i])
            if not (ok_p and ok_a):
                raise RuntimeError(f"stored map reconstruction failed {r['ref_key']} {d_p} {d_a}")
            maps = {"ig_abs": np.asarray(z["ig512_absolute"][i], np.float64),
                    "ig_pos_smooth": smooth(np.asarray(z["ig512_primary"][i], np.float64))}
            smooth_maps.append(maps["ig_pos_smooth"].astype(np.float32))
            rnd = r7["controls"]["random"]
            row = {"ref_key": r["ref_key"], "record_uid": r["record_uid"], "pair_id": r["pair_id"], "label_mel": r["label_mel"],
                   "modality": r["modality"], "stage3b_pilot_member": r7["stage3b_pilot_member"],
                   "baseline_sha256": r7["baseline_sha256"], "ig_completeness": r7["ig_completeness"],
                   "recon_max_abs_diff": {"primary": d_p, "absolute": d_a},
                   "random_7a": {"status": rnd["status"], "auc_ins_each": rnd["auc_ins_each"], "auc_del_each": rnd["auc_del_each"],
                                 "p_full": rnd["curves"][0]["p_full"], "p_base": rnd["curves"][0]["p_base"]},
                   "variants": {}}
            for v in VARIANTS:
                mp = torch.as_tensor(maps[v], dtype=torch.float64, device="cuda")
                val = XA.validity(mp)
                ok, reasons = eligible_secondary(r7["ig_completeness"]["failure"], val["flags"])
                d = {"validity": val, "eligible_secondary": ok, "reasons": reasons, "map_sha256_float64": XP._sha_bytes(maps[v].tobytes())}
                if ok:
                    ts = time.perf_counter()
                    c = XC.curves(model, x[i], base[i], mp, s=XP.S, eval_batch=XP.EVAL_BATCH)
                    tc += time.perf_counter() - ts
                    good = c["curve_status"] == "ok" and c["endpoint_status"] == "pass"
                    ident = abs(c["p_full"] - rnd["curves"][0]["p_full"]) <= RANDOM_P_TOL and abs(c["p_base"] - rnd["curves"][0]["p_base"]) <= RANDOM_P_TOL
                    if not ident:
                        raise RuntimeError(f"random-control identity check failed {r['ref_key']}")
                    d.update(status="available" if good else "unavailable", reason=None if good else "curve invalid or endpoint failure",
                             curve=XP._strip(c), random_identity_p_diff=[c["p_full"] - rnd["curves"][0]["p_full"], c["p_base"] - rnd["curves"][0]["p_base"]])
                else:
                    d.update(status="ineligible", reason=";".join(reasons))
                row["variants"][v] = d
            res.append(row)
        tmp = f"{final}.tmp-{uuid.uuid4().hex[:8]}"
        os.makedirs(tmp)
        buf = io.BytesIO()
        np.savez_compressed(buf, ref_keys=np.array([r["ref_key"] for r in rr]), ig_pos_smooth_float32=np.stack(smooth_maps))
        with open(os.path.join(tmp, "maps_smooth.npz"), "xb") as f:
            f.write(buf.getvalue())
        with open(os.path.join(tmp, "records.json"), "x", encoding="utf-8") as f:
            json.dump(res, f)
        meta = {"spec_sha256": ssha, "chunk": ch, "source_chunk_manifest_sha256": P.sha256_file(os.path.join(src, "chunk_manifest.json")),
                "seconds": time.perf_counter() - t0, "curve_seconds": tc,
                "peak_cuda_mib": [round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)]}
        XP.publish_dir(tmp, final, meta)
        print(f"var {ch['chunk']:02d} done {meta['seconds']:.1f}s peak={meta['peak_cuda_mib']}", flush=True)
        del x
    done = sum(XP.verify_dir(os.path.join(out, f"var_{c['chunk']:02d}"), ssha) for c in s7["chunks"])
    print(f"COMPUTE STATUS {done}/{len(s7['chunks'])} in {time.perf_counter() - t_all:.1f}s", flush=True)


def _variant_rows(recs, v):
    from . import xai_controls as XK
    rows = []
    for r in recs:
        d = r["variants"][v]
        rnd = dict(r["random_7a"], status=r["random_7a"]["status"])
        if d["status"] == "available":
            sc = XK.control_scores({"auc_ins": d["curve"]["auc_insertion_v1"], "auc_del": d["curve"]["auc_deletion_v1"]}, rnd)
        else:
            sc = {"status": "unavailable", "insertion_advantage": None, "deletion_advantage": None}
        rows.append({"ref_key": r["ref_key"], "pair_id": r["pair_id"], "modality": r["modality"], "label_mel": r["label_mel"],
                     "status": d["status"], "reasons": d.get("reason") or "", "ok": d["status"] == "available" and sc["status"] == "available",
                     "ins": sc["insertion_advantage"], "del": sc["deletion_advantage"],
                     "auc_ins": d.get("curve", {}).get("auc_insertion_v1"), "auc_del": d.get("curve", {}).get("auc_deletion_v1"),
                     "endpoint_status": d.get("curve", {}).get("endpoint_status")})
    return rows


def _summ(x):
    from . import rq4_stats as R4
    return {"n": int(len(x)), "median": R4.sample_median(x), "hl": R4.hodges_lehmann(x)}


def phase_analyze(cfg, a):
    from . import provenance as P, rq4_analysis as RA, rq4_stats as R4, xai_prod as XP
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    t0 = time.perf_counter()
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load_spec(cfg, out)
    run = _abs(cfg, R7A)
    s7, _ = XP._load(run, "spec.json")
    recs = []
    for ch in s7["chunks"]:
        d = os.path.join(out, f"var_{ch['chunk']:02d}")
        if not XP.verify_dir(d, ssha):
            raise RuntimeError(f"{d} incomplete")
        recs += XP._read_json(os.path.join(d, "records.json"))
    if [r["ref_key"] for r in recs] != [r["ref_key"] for r in s7["refs"]]:
        raise RuntimeError("coverage")
    # original primary IG per-image rows (Stage 7B definitions)
    orows, _, checks = RA._image_rows(run, None)
    orig_rows = [{"pair_id": r["pair_id"], "modality": r["modality"], "ok": r["status"] == "primary",
                  "ins": None if r["delta_ins"] == "" else float(r["delta_ins"]), "del": None if r["delta_del"] == "" else float(r["delta_del"])}
                 for r in orows["ig512"]]
    orig = pair_population(orig_rows)
    if len(orig) != 373:
        raise RuntimeError(f"original IG pairs {len(orig)} != 373")
    # reproduce Stage 7B primary IG HL / median on its own population
    fam = {r["test"]: r for r in XP._read_csv_rows(os.path.join(_abs(cfg, spec["stage7b"]["run"]), "rq4_family_table.csv"))}
    repro = 0.0
    ids_o = sorted(orig)
    for c in CURVES:
        for m in MODS:
            v = np.array([orig[p][m][c] for p in ids_o])
            f7 = fam[f"E4.1 ig512 {c} {m}"]
            repro = max(repro, abs(R4.hodges_lehmann(v) - float(f7["hl"])), abs(R4.sample_median(v) - float(f7["median"])))
        v = np.array([orig[p]["clinical"][c] - orig[p]["dermoscopy"][c] for p in ids_o])
        f7 = fam[f"E4.2 ig512 {c} D"]
        repro = max(repro, abs(R4.hodges_lehmann(v) - float(f7["hl"])))
    if repro > 1e-12:
        raise RuntimeError(f"Stage 7B IG reproduction failed {repro}")
    res = {"stage7b_ig_reproduction_max_abs_diff": repro, "variants": {}}
    elig_rows, img_rows, comp_rows, own_rows, pair_rows = [], [], [], [], []
    reps = {}
    for v in VARIANTS:
        vr = _variant_rows(recs, v)
        img_rows += [dict(r, variant=v) for r in vr]
        for m in MODS:
            for lab in ("1", "0"):
                rr = [r for r in vr if r["modality"] == m and r["label_mel"] == lab]
                reasons = {}
                for r in rr:
                    if not r["ok"]:
                        reasons[r["reasons"] or r["status"]] = reasons.get(r["reasons"] or r["status"], 0) + 1
                elig_rows.append({"variant": v, "modality": m, "label_mel": lab, "n": len(rr), "eligible": sum(r["ok"] for r in rr),
                                  "excluded": sum(not r["ok"] for r in rr), "reasons": json.dumps(reasons),
                                  "endpoint_failures": sum(r["endpoint_status"] == "fail" for r in rr)})
        own = pair_population(vr)
        inter = sorted(set(own) & set(orig))
        pop = {"own_pairs": len(own), "own_by_label": {lab: sum(own[p]["dermoscopy"]["label_mel"] == lab for p in own) for lab in ("1", "0")},
               "original_pairs": len(orig), "intersection_pairs": len(inter),
               "intersection_by_label": {lab: sum(own[p]["dermoscopy"]["label_mel"] == lab for p in inter) for lab in ("1", "0")},
               "variant_only_pairs": sorted(set(own) - set(orig)), "original_only_pairs": sorted(set(orig) - set(own)),
               "image_level_eligible": sum(r["ok"] for r in vr),
               "completeness_failed_images_excluded": sum("ig_completeness_failure" in r["reasons"] for r in vr),
               "map_invalid_images": sum("map:" in r["reasons"] for r in vr)}
        # own population (descriptive)
        ids = sorted(own)
        ov = {}
        for c in CURVES:
            for m in MODS:
                ov[f"{c}|{m}"] = np.array([own[p][m][c] for p in ids])
            ov[f"{c}|D"] = ov[f"{c}|clinical"] - ov[f"{c}|dermoscopy"]
        bs, meta = R4.bootstrap_stats(ov, B, SEEDS[f"{v}_own"])
        for k, x in ov.items():
            own_rows.append(dict(variant=v, population="own", quantity=k, **_summ(x), median_ci95=json.dumps(bs[k]["median"]["ci95"]),
                                 hl_ci95=json.dumps(bs[k]["hl"]["ci95"])))
        # intersection comparison (shared draws across orig, variant and their difference)
        cv = contrast_vectors({p: orig[p] for p in inter}, {p: own[p] for p in inter}, inter)
        bs2, meta2 = R4.bootstrap_stats(cv, B, SEEDS[f"{v}_vs_orig"])
        for k, x in cv.items():
            comp_rows.append(dict(variant=v, population="intersection", quantity=k, **_summ(x),
                                  median_ci95=json.dumps(bs2[k]["median"]["ci95"]), hl_ci95=json.dumps(bs2[k]["hl"]["ci95"])))
            reps[f"{v}__{k.replace('|', '__')}__hl"] = bs2[k]["hl"]["replicates"]
        for p in inter:
            pair_rows.append({"variant": v, "pair_id": p, "label_mel": own[p]["dermoscopy"]["label_mel"],
                              **{f"orig_{c}_{m}": orig[p][m][c] for c in CURVES for m in MODS},
                              **{f"var_{c}_{m}": own[p][m][c] for c in CURVES for m in MODS}})
        res["variants"][v] = {"population": pop, "bootstrap_own": meta, "bootstrap_intersection": meta2}
        print(v, {k: pop[k] for k in ("own_pairs", "intersection_pairs", "image_level_eligible", "completeness_failed_images_excluded",
                                       "map_invalid_images")}, flush=True)
    tim = [XP._read_json(os.path.join(out, f"var_{c['chunk']:02d}", "chunk_manifest.json")) for c in s7["chunks"]]
    res["measured"] = {"compute_seconds_sum": sum(t["seconds"] for t in tim), "curve_seconds_sum": sum(t["curve_seconds"] for t in tim),
                       "peak_cuda_mib_max": [max(t["peak_cuda_mib"][0] for t in tim), max(t["peak_cuda_mib"][1] for t in tim)],
                       "recon_max_abs_diff": {k: max(r["recon_max_abs_diff"][k] for r in recs) for k in ("primary", "absolute")},
                       "random_identity_max_abs_p_diff": max(abs(x) for r in recs for v in VARIANTS
                                                             for x in r["variants"][v].get("random_identity_p_diff", [0.0])),
                       "disk_bytes": sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs),
                       "analyze_seconds": time.perf_counter() - t0}
    publish_json(out, "results.json", res)
    publish_csv(out, "eligibility_table.csv", elig_rows, list(elig_rows[0]))
    publish_csv(out, "per_image_variant_scores.csv", img_rows, list(img_rows[0]))
    publish_csv(out, "own_population_stats.csv", own_rows, list(own_rows[0]))
    publish_csv(out, "intersection_comparison.csv", comp_rows, list(comp_rows[0]))
    publish_csv(out, "intersection_per_pair.csv", pair_rows, list(pair_rows[0]))
    buf = io.BytesIO()
    np.savez_compressed(buf, **reps)
    publish_bytes(out, "bootstrap_replicates.npz", buf.getvalue())
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))
                          if os.path.isfile(os.path.join(out, n)) and not n.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    print(json.dumps(res["measured"]), repro, "verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "compute", "analyze"))
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
    {"spec": phase_spec, "compute": phase_compute, "analyze": phase_analyze}[a.phase](cfg, a)


if __name__ == "__main__":
    main()
