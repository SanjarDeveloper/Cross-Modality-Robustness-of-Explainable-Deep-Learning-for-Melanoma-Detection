"""Stage 12A: secondary IG variants (Stage 10A: ig_abs, ig_pos_smooth) against the centre, Sobel and mismatch controls
(02_rev2 §6.1 controls (b)-(d); "Secondary attribution" row). Seed-0 standard ResNet-50, Derm7pt XAI cohort only.

Phases:
  spec      freeze the comparison; verify 7A/10A artefacts; reconstruct every variant map with the frozen 10A recipe
            (float64 from the stored float32 7A arrays; smoothing via xai_secondary.smooth) and verify its SHA-256 against
            the 10A record; verify centre/Sobel curve identity (same x, baseline, model, settings: p_full/p_base equal)
  mismatch  per 7A chunk (same membership; baseline hash re-checked): curves of the DONOR's variant map on the RECIPIENT
            image, donors from the frozen 7A ig512 assignment, donor eligibility = donor's variant record 'available'
            (variant validity + inherited IG completeness); never re-selected; hashed atomic dirs
  analyze   descriptive: per variant x control, E4.1 (per modality) and E4.2 (D = clinical - dermoscopy) medians and HL on
            complete modality pairs; centre/Sobel with the 7B paired case bootstrap (B=2000); mismatch point estimates
            only; direct variant-minus-original-IG on identical eligible pairs; 10A random summaries reproduced

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_secondary_ctrl --config ../configs/xai_secondary_ctrl_12a_v1.json --phase spec --out-root ../runs/xai_secondary_ctrl_12a
  ... --phase mismatch|analyze --run-dir <dir>
"""
import argparse
import io
import json
import os
import time
import uuid

import numpy as np

R7A = "research_v1/runs/xai_prod_7a/20261003T173258Z_3b655f88_548b3a05"
R10A = "research_v1/runs/xai_secondary_10a/20261004T171602Z_a97243d1_c22139d2"
VARIANTS = ("ig_abs", "ig_pos_smooth")
CONTROLS = ("random", "centre", "sobel", "mismatch")
MODS = ("dermoscopy", "clinical")
CURVES = ("ins", "del")
B = 2000
SEEDS = {f"{v}|{c}": 20261400 + 10 * i + j for i, v in enumerate(VARIANTS) for j, c in enumerate(CONTROLS)}
SEEDS.update({f"{v}|{c}|vs_orig": 20261500 + 10 * i + j for i, v in enumerate(VARIANTS) for j, c in enumerate(CONTROLS)})
SEEDS.update({"ig_abs|random": 20261100, "ig_pos_smooth|random": 20261101})   # = Stage 10A own-population seeds (reproduction)
IDENTITY_TOL = 1e-6


# ------------------------------------------------------------------ pure helpers (tested)
def variant_map(v, ig_primary_f32, ig_abs_f32):
    """Frozen Stage 10A recipe: float64 from the stored float32 arrays; smoothing applied to the primary map."""
    from .xai_secondary import smooth
    if v == "ig_abs":
        return np.asarray(ig_abs_f32, np.float64)
    if v == "ig_pos_smooth":
        return smooth(np.asarray(ig_primary_f32, np.float64))
    raise ValueError(v)


def sha64(m):
    import hashlib
    return hashlib.sha256(np.asarray(m, np.float64).tobytes()).hexdigest()


def donor_key(record_uid, modality):
    """Variant mismatch reuses the frozen 7A ig512 donor assignment (same recipient, same planned donor)."""
    from .xai_controls import assignment_key
    return assignment_key(record_uid, modality, "ig512", "none")


def complete_pairs(rows):
    """rows: per-image {pair_id, modality, ins, del} with None when unavailable -> {pair: {mod: row}} complete pairs."""
    by = {}
    for r in rows:
        by.setdefault(r["pair_id"], {})[r["modality"]] = r
    return {p: d for p, d in by.items() if set(d) == set(MODS) and all(d[m]["ins"] is not None and d[m]["del"] is not None for m in MODS)}


# ------------------------------------------------------------------ io
def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _abs(cfg, rel):
    from . import config as C
    return C.resolve(cfg, rel)


def _load_7a_10a(cfg):
    from . import provenance as P, xai_prod as XP
    run7, run10 = _abs(cfg, R7A), _abs(cfg, R10A)
    s7, s7sha = XP._load(run7, "spec.json")
    s10, s10sha = XP._load(run10, "spec.json")
    for n, h in XP._read_json(os.path.join(run7, "receipt_audit.json"))["outputs"].items():
        if P.sha256_file(os.path.join(run7, n)) != h:
            raise RuntimeError(f"7A changed {n}")
    for n, h in XP._read_json(os.path.join(run10, "receipt_analyze.json"))["outputs"].items():
        if P.sha256_file(os.path.join(run10, n)) != h:
            raise RuntimeError(f"10A changed {n}")
    r7, r10, maps = {}, {}, {}
    for ch in s7["chunks"]:
        d7, d10 = os.path.join(run7, f"chunk_{ch['chunk']:02d}"), os.path.join(run10, f"var_{ch['chunk']:02d}")
        if not (XP.verify_dir(d7, s7sha) and XP.verify_dir(d10, s10sha)):
            raise RuntimeError(f"chunk {ch['chunk']} does not verify")
        z = np.load(os.path.join(d7, "maps.npz"))
        for i, r in enumerate(XP._read_json(os.path.join(d7, "records.json"))):
            r7[r["ref_key"]] = dict(r, chunk=ch["chunk"])
            maps[r["ref_key"]] = (z["ig512_primary"][i], z["ig512_absolute"][i])
        for r in XP._read_json(os.path.join(d10, "records.json")):
            r10[r["ref_key"]] = r
    if set(r7) != set(r10) or len(r7) != 996:
        raise RuntimeError("7A/10A record identity")
    return s7, s7sha, r7, r10, maps


def phase_spec(cfg, a):
    from . import provenance as P, xai_prod as XP
    from .desc_4b3 import publish_bytes, publish_json
    s7, s7sha, r7, r10, maps = _load_7a_10a(cfg)
    fid = {v: {"checked": 0, "hash_match": 0} for v in VARIANTS}
    ident = {"centre": 0.0, "sobel": 0.0, "checked": 0}
    for k, r in r10.items():
        if r["record_uid"] != r7[k]["record_uid"] or r["baseline_sha256"] != r7[k]["baseline_sha256"]:
            raise RuntimeError(f"identity {k}")
        for v in VARIANTS:
            m = variant_map(v, *maps[k])
            fid[v]["checked"] += 1
            fid[v]["hash_match"] += sha64(m) == r["variants"][v]["map_sha256_float64"]
            c = r["variants"][v].get("curve")
            if c:
                for cn in ("centre", "sobel"):
                    cc = r7[k]["controls"][cn].get("curve")
                    if cc:
                        ident[cn] = max(ident[cn], abs(cc["p_full"] - c["p_full"]), abs(cc["p_base"] - c["p_base"]))
                ident["checked"] += 1
    if any(f["hash_match"] != f["checked"] for f in fid.values()) or max(ident["centre"], ident["sobel"]) > IDENTITY_TOL:
        raise RuntimeError(f"fidelity/identity failed {fid} {ident}")
    assign, ash = XP._load(_abs(cfg, R7A), "mismatch_assignment.json")
    spec = {"spec_version": "xai_secondary_ctrl_12a_v1", "created_utc": _now(),
            "inputs": {"stage7a": {"run": R7A, "spec_sha256": s7sha, "mismatch_assignment_sha256": ash},
                       "stage10a": {"run": R10A, "spec_sha256": P.sha256_file(os.path.join(_abs(cfg, R10A), "spec.json"))}},
            "variants": "Stage 10A unchanged: ig_abs = sum_RGB|attr| (stored 7A ig512_absolute -> float64); ig_pos_smooth = Gaussian "
                        "sigma=2 (frozen 10A kernel/padding) of ReLU(sum_RGB attr) (stored 7A ig512_primary -> float64)",
            "fidelity": {"variant_map_hash_check": fid, "note": "maps reconstructed with the frozen 10A recipe (float64 from the stored "
                         "float32 7A arrays) and matched to the 10A map_sha256_float64 of every record; the 10A float32 smoothed copy is "
                         "NOT used for ranking"},
            "centre_sobel_reuse": {"source": "7A controls.centre / controls.sobel curves", "identity_check_max_abs_p_diff": ident,
                                   "tolerance": IDENTITY_TOL, "rationale": "control curves depend only on (model, x, baseline, S, "
                                   "eval_batch, precision), identical to the 10A variant curves (p_full/p_base equal)"},
            "mismatch": {"assignment": "frozen 7A ig512 assignment (fixed derangement within label x modality; other case); the "
                                       "recipient keeps its planned donor; donor eligible iff its 10A variant record is 'available' "
                                       "(variant validity + inherited IG completeness); otherwise unavailable, never replaced",
                         "curve": "donor's variant map (float64, frozen recipe) on the recipient image and recipient baseline, "
                                  "S=32, eval_batch 11, xai_precision_v1; primary-IG mismatch curves are NOT relabelled",
                         "uncertainty": "point estimates only (donor dependence not captured by recipient resampling)"},
            "analysis": {"per_image": "insertion = AUC_ins(variant) - AUC_ins(control); deletion = AUC_del(control) - AUC_del(variant); "
                                      "random = mean of the same 5 AUCs (controls_v1.control_scores)",
                         "population": "complete modality pairs: variant curve available AND control available in both modalities",
                         "statistics": "median and HL (rq4_stats); centre/Sobel/random: 7B paired case bootstrap, one shared draw per "
                                       "replicate across all vectors of a comparison, B=2000; mismatch: point estimates",
                         "vs_original": "variant minus original primary IG with the SAME control on identical pairs (original IG "
                                        "primary + variant + control available in both modalities)",
                         "seeds": SEEDS, "tests": "none (descriptive); 12-test Holm family unchanged; no preferred variant"},
            "sources": {n: P.sha256_file(os.path.join(os.path.dirname(__file__), n)) for n in
                        ("xai_secondary_ctrl.py", "xai_secondary.py", "xai_controls.py", "xai_curves.py", "xai_prod.py", "rq4_stats.py",
                         "rq4_analysis.py")}}
    from . import provenance as Pv
    receipt = Pv.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "spec.json", spec)
    publish_bytes(out, "spec.json.sha256", f"{P.sha256_file(os.path.join(out, 'spec.json'))}  spec.json\n".encode())
    receipt["outputs"] = {"spec.json": P.sha256_file(os.path.join(out, "spec.json"))}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out, fid, ident)


def _spec(cfg, out):
    from . import provenance as P, xai_prod as XP
    spec, ssha = XP._load(out, "spec.json")
    for n, h in spec["sources"].items():
        if P.sha256_file(os.path.join(os.path.dirname(__file__), n)) != h:
            raise RuntimeError(f"source changed {n}")
    return spec, ssha


def phase_mismatch(cfg, a):
    import torch
    from . import xai_attr as XA, xai_curves as XC, xai_prod as XP
    out = os.path.abspath(a.run_dir)
    spec, ssha = _spec(cfg, out)
    s7, s7sha, r7, r10, maps = _load_7a_10a(cfg)
    assign = XP._load(_abs(cfg, R7A), "mismatch_assignment.json")[0]["assignment"]
    uid_ref = {r["record_uid"]: k for k, r in r7.items()}
    XP.quarantine_tmp(out)
    model = XP._model(cfg)
    t_all = time.perf_counter()
    for ch in s7["chunks"]:
        final = os.path.join(out, f"mm_{ch['chunk']:02d}")
        if XP.verify_dir(final, ssha):
            continue
        rr = s7["refs"][ch["start"]:ch["end"]]
        if a.max_chunks is not None and ch["chunk"] >= a.max_chunks:
            break
        t0 = time.perf_counter()
        x = XP._inputs(cfg, rr)
        base = XA.gaussian_baseline(x)
        res, tc = [], 0.0
        for i, r in enumerate(rr):
            k = r["ref_key"]
            if XP._sha_bytes(base[i].detach().cpu().numpy().tobytes()) != r7[k]["baseline_sha256"]:
                raise RuntimeError(f"baseline differs from 7A {k}")
            a_ = assign[donor_key(r["record_uid"], r["modality"])]
            row = {"ref_key": k, "donor_record_uid": a_["donor"], "assignment_status": a_["status"]}
            for v in VARIANTS:
                if a_["status"] != "assigned":
                    row[v] = {"status": "unavailable", "reason": a_["reason"]}
                    continue
                dk = uid_ref[a_["donor"]]
                if r10[dk]["variants"][v]["status"] != "available":
                    row[v] = {"status": "unavailable", "reason": f"planned donor variant map not eligible ({r10[dk]['variants'][v].get('reason')})",
                              "donor_ref_key": dk}
                    continue
                dm = variant_map(v, *maps[dk])
                if sha64(dm) != r10[dk]["variants"][v]["map_sha256_float64"]:
                    raise RuntimeError(f"donor map hash {dk} {v}")
                ts = time.perf_counter()
                c = XC.curves(model, x[i], base[i], torch.as_tensor(dm, dtype=torch.float64, device="cuda"), s=XP.S, eval_batch=XP.EVAL_BATCH)
                tc += time.perf_counter() - ts
                good = c["curve_status"] == "ok" and c["endpoint_status"] == "pass"
                row[v] = {"status": "available" if good else "unavailable", "reason": None if good else "curve invalid or endpoint failure",
                          "donor_ref_key": dk, "donor_map_sha256_float64": sha64(dm), "curve": XP._strip(c),
                          "auc_ins": c["auc_insertion_v1"], "auc_del": c["auc_deletion_v1"],
                          "recipient_p_identity": [c["p_full"] - r7[k]["controls"]["random"]["curves"][0]["p_full"],
                                                   c["p_base"] - r7[k]["controls"]["random"]["curves"][0]["p_base"]]}
            res.append(row)
        tmp = f"{final}.tmp-{uuid.uuid4().hex[:8]}"
        os.makedirs(tmp)
        with open(os.path.join(tmp, "records.json"), "x", encoding="utf-8") as f:
            json.dump(res, f)
        XP.publish_dir(tmp, final, {"spec_sha256": ssha, "chunk": ch["chunk"], "seconds": time.perf_counter() - t0, "curve_seconds": tc})
        print(f"mm {ch['chunk']:02d} {time.perf_counter() - t0:.1f}s", flush=True)
        del x
    print(f"MISMATCH STATUS in {time.perf_counter() - t_all:.1f}s", flush=True)


def phase_analyze(cfg, a):
    from . import provenance as P, rq4_analysis as RA, rq4_stats as R4, xai_controls as XK, xai_prod as XP
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    t0 = time.perf_counter()
    out = os.path.abspath(a.run_dir)
    spec, ssha = _spec(cfg, out)
    s7, s7sha, r7, r10, maps = _load_7a_10a(cfg)
    mm = {}
    for ch in s7["chunks"]:
        d = os.path.join(out, f"mm_{ch['chunk']:02d}")
        if not XP.verify_dir(d, ssha):
            raise RuntimeError(f"{d} incomplete")
        for r in XP._read_json(os.path.join(d, "records.json")):
            mm[r["ref_key"]] = r
    if set(mm) != set(r7):
        raise RuntimeError("mismatch coverage")
    ident = max(abs(x) for r in mm.values() for v in VARIANTS for x in r[v].get("recipient_p_identity", [0.0]))
    if ident > IDENTITY_TOL:
        raise RuntimeError(f"recipient identity {ident}")
    # original primary IG per-image deltas per control (Stage 7A scores_v2 via rq4_analysis rows + scores)
    run7 = _abs(cfg, R7A)
    orows, sc, _ = RA._image_rows(run7, None)
    ostat = {r["ref_key"]: r["status"] for r in orows["ig512"]}

    def orig_delta(k, cn):
        s = sc[(k, "ig512", cn)]
        if ostat[k] != "primary" or s["control_status"] != "available":
            return None, None
        return float(s["delta_ins"]), float(s["delta_del"])

    def var_delta(k, v, cn):
        d = r10[k]["variants"][v]
        if d["status"] != "available":
            return None, None, d.get("reason") or d["status"]
        expl = {"auc_ins": d["curve"]["auc_insertion_v1"], "auc_del": d["curve"]["auc_deletion_v1"]}
        ctrl = mm[k][v] if cn == "mismatch" else r7[k]["controls"][cn]
        s_ = XK.control_scores(expl, ctrl)
        if s_["status"] != "available":
            return None, None, f"control:{s_.get('reason')}"
        return s_["insertion_advantage"], s_["deletion_advantage"], None

    results, counts, cmp_rows, ex_rows = [], [], [], []
    reps = {}
    for v in VARIANTS:
        for cn in CONTROLS:
            rows, excl = [], {}
            for k, r in r7.items():
                i_, d_, why = var_delta(k, v, cn)
                rows.append({"pair_id": r["pair_id"], "modality": r["modality"], "label": r["label_mel"], "ins": i_, "del": d_})
                if why:
                    excl[why] = excl.get(why, 0) + 1
            cp = complete_pairs(rows)
            ids = sorted(cp)
            vec = {}
            for c_ in CURVES:
                for m in MODS:
                    vec[f"{c_}|{m}"] = np.array([cp[p][m][c_] for p in ids])
                vec[f"{c_}|D"] = vec[f"{c_}|clinical"] - vec[f"{c_}|dermoscopy"]
            boot = cn != "mismatch"
            bs = R4.bootstrap_stats(vec, B, SEEDS[f"{v}|{cn}"])[0] if boot else None
            counts.append({"variant": v, "control": cn, "pairs": len(ids), "mel_pairs": sum(cp[p]["dermoscopy"]["label"] == "1" for p in ids),
                           "nonmel_pairs": sum(cp[p]["dermoscopy"]["label"] == "0" for p in ids), "image_exclusions": json.dumps(excl)})
            for q, x in vec.items():
                results.append({"variant": v, "control": cn, "quantity": q, "pairs": len(ids), "median": R4.sample_median(x), "hl": R4.hodges_lehmann(x),
                                "median_ci95": json.dumps(bs[q]["median"]["ci95"]) if boot else "", "hl_ci95": json.dumps(bs[q]["hl"]["ci95"]) if boot else "",
                                "note": "" if boot else "point estimate only; donor dependence not captured"})
            # direct comparison with original IG on identical pairs (same control)
            orow = []
            for k, r in r7.items():
                o = orig_delta(k, cn)
                orow.append({"pair_id": r["pair_id"], "modality": r["modality"], "ins": o[0], "del": o[1]})
            op = complete_pairs(orow)
            inter = sorted(set(op) & set(cp))
            cv = {}
            for c_ in CURVES:
                for m in MODS:
                    cv[f"orig|{c_}|{m}"] = np.array([op[p][m][c_] for p in inter])
                    cv[f"var|{c_}|{m}"] = np.array([cp[p][m][c_] for p in inter])
                    cv[f"var_minus_orig|{c_}|{m}"] = cv[f"var|{c_}|{m}"] - cv[f"orig|{c_}|{m}"]
                for w in ("orig", "var", "var_minus_orig"):
                    cv[f"{w}|{c_}|D"] = cv[f"{w}|{c_}|clinical"] - cv[f"{w}|{c_}|dermoscopy"]
            bs2 = R4.bootstrap_stats(cv, B, SEEDS[f"{v}|{cn}|vs_orig"])[0] if boot else None
            for q, x in cv.items():
                cmp_rows.append({"variant": v, "control": cn, "quantity": q, "pairs": len(inter), "median": R4.sample_median(x), "hl": R4.hodges_lehmann(x),
                                 "hl_ci95": json.dumps(bs2[q]["hl"]["ci95"]) if boot else "", "note": "" if boot else "point estimate only"})
            ex_rows.append({"variant": v, "control": cn, "variant_pairs": len(ids), "original_pairs": len(op), "intersection_pairs": len(inter),
                            "variant_only": len(set(cp) - set(op)), "original_only": len(set(op) - set(cp))})
            print(v, cn, "pairs", len(ids), "inter", len(inter), flush=True)
    # reproduce 10A random own-population summaries
    o10 = {(r["variant"], r["quantity"]): r for r in XP._read_csv_rows(os.path.join(_abs(cfg, R10A), "own_population_stats.csv"))}
    rep = 0.0
    for r in results:
        if r["control"] == "random":
            o = o10[(r["variant"], r["quantity"])]
            rep = max(rep, abs(float(o["hl"]) - r["hl"]), abs(float(o["median"]) - r["median"]), abs(int(o["n"]) - r["pairs"]))
            rep = max(rep, max(abs(a_ - b_) for a_, b_ in zip(json.loads(o["hl_ci95"]), json.loads(r["hl_ci95"]))))
    if rep > 1e-12:
        raise RuntimeError(f"10A random reproduction failed {rep}")
    man = [XP._read_json(os.path.join(out, f"mm_{c['chunk']:02d}", "chunk_manifest.json")) for c in s7["chunks"]]
    res = {"stage10a_random_reproduction_max_abs_diff": rep, "recipient_identity_max_abs_p_diff": ident,
           "mismatch_runtime_s": sum(m_["seconds"] for m_ in man), "mismatch_curve_s": sum(m_["curve_seconds"] for m_ in man),
           "mismatch_unavailable": {v: {} for v in VARIANTS}, "analyze_seconds": time.perf_counter() - t0}
    for r in mm.values():
        for v in VARIANTS:
            if r[v]["status"] != "available":
                k_ = r[v]["reason"]
                res["mismatch_unavailable"][v][k_] = res["mismatch_unavailable"][v].get(k_, 0) + 1
    publish_json(out, "results.json", res)
    publish_csv(out, "variant_control_stats.csv", results, list(results[0]))
    publish_csv(out, "population_counts.csv", counts, list(counts[0]))
    publish_csv(out, "variant_vs_original.csv", cmp_rows, list(cmp_rows[0]))
    publish_csv(out, "intersection_counts.csv", ex_rows, list(ex_rows[0]))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))
                          if os.path.isfile(os.path.join(out, n)) and not n.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    print(json.dumps(res, indent=1), "verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "mismatch", "analyze"))
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
    {"spec": phase_spec, "mismatch": phase_mismatch, "analyze": phase_analyze}[a.phase](cfg, a)


if __name__ == "__main__":
    main()
