"""Stage 10D: descriptive statistical analysis of the Stage 10C simulated-HAM XAI arm (CPU only; saved records).

Estimands (02_rev2 §6.3 directions; §6.7 simulated arm; Stage 10B analysis plan "descriptive"):
  faithfulness  Δ_ins = AUC_ins(explainer) - mean_k AUC_ins(random_k);  Δ_del = mean_k AUC_del(random_k) - AUC_del(explainer)
                (controls_v1.control_scores; the five random AUCs averaged within each image first)
  per condition: median and Hodges-Lehmann (HL) of Δ over images with a primary map and an available random control
  change from severity 0: per image d = Δ(condition) - Δ(severity 0) on EXACTLY the images eligible (map + random) at both;
                both levels and the paired change summarised on that same subset
  method comparison: Δ(IG2048) - Δ(Grad-CAM) on the per-condition eligibility intersection
  localisation: matched populations (all maps in a comparison available); area-matched IoU (median) and pointing (rate);
                explainer - control paired differences; change from severity 0 on images available at both
  uncertainty: lesion-cluster bootstrap B=2000, ONE shared draw of lesions per replicate across all statistics
                (all images and all conditions of a drawn lesion together, multiplicity kept); percentile 95% intervals
                (numpy linear); undefined replicates counted, never redrawn. Pointwise, not simultaneous.
  no hypothesis tests (no simulated-arm testing family was prespecified); the Derm7pt 12-test Holm family is untouched.
Retrospective additions (frozen in spec before computation): the bootstrap seed, the paired-change and method-comparison
summaries, the localisation summaries (median IoU, pointing rate), and the as-computed sensitivity summaries.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_sim_stats --config ../configs/xai_sim_stats_10d_v1.json --phase spec --out-root ../runs/xai_sim_stats_10d
  ... --phase analyze --run-dir <dir>
"""
import argparse
import io
import json
import os
import time

import numpy as np

R10C = "research_v1/runs/xai_sim_prod_10c/20261009T073606Z_d9545ce7_386f4032"
B = 2000
SEED = 20261200
EXPL = ("gradcam", "ig2048")
CTRL_LOC = ("random_mean", "centre", "sobel")
CURVES = ("ins", "del")


# ------------------------------------------------------------------ pure helpers (tested)
def lesion_weights(clusters, B_, seed):
    """clusters: list (per image) of cluster ids. Returns (B, n_images) int multiplicity matrix: per replicate, draw
    K lesions with replacement (K = #lesions); every image of a drawn lesion gets +1 per draw."""
    ids = sorted(set(clusters))
    idx = {c: i for i, c in enumerate(ids)}
    lab = np.array([idx[c] for c in clusters])
    rng = np.random.Generator(np.random.PCG64(seed))
    W = np.zeros((B_, len(clusters)), dtype=np.int32)
    for b in range(B_):
        cnt = np.bincount(rng.integers(0, len(ids), size=len(ids)), minlength=len(ids))
        W[b] = cnt[lab]
    return W


def hl(x):
    from .rq4_stats import hodges_lehmann
    return hodges_lehmann(x)


def summarise(values, mask, W, stats=("median", "hl")):
    """values: (n_images,) float with NaN where unavailable; mask: bool population. Point + bootstrap percentile CI per stat.
    Replicates with zero weight in the population are undefined (counted)."""
    v = values[mask]
    if np.isnan(v).any():
        raise ValueError("unavailable value inside the analysis population")
    fns = {"median": np.median, "hl": hl, "mean": np.mean}
    out = {"n": int(mask.sum())}
    reps = {s: [] for s in stats}
    und = 0
    for b in range(W.shape[0]):
        w = W[b][mask]
        if w.sum() == 0:
            und += 1
            continue
        s_ = np.repeat(v, w)
        for s in stats:
            reps[s].append(fns[s](s_))
    for s in stats:
        a = np.asarray(reps[s])
        out[s] = float(fns[s](v)) if len(v) else None
        out[f"{s}_ci95"] = np.quantile(a, [0.025, 0.975], method="linear").tolist() if a.size else None
    out["undefined_replicates"] = und
    return out


def population(mask, labels, clusters):
    m = np.asarray(mask, bool)
    return {"images": int(m.sum()), "lesions": len({c for c, k in zip(clusters, m) if k}),
            "mel": int(sum(1 for l_, k in zip(labels, m) if k and l_ == "1")),
            "nonmel": int(sum(1 for l_, k in zip(labels, m) if k and l_ == "0"))}


# ------------------------------------------------------------------ io
def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _abs(cfg, rel):
    from . import config as C
    return C.resolve(cfg, rel)


def phase_spec(cfg, a):
    from . import provenance as P, xai_prod as XP
    from .desc_4b3 import publish_bytes, publish_json
    run = _abs(cfg, R10C)
    s10, ssha = XP._load(run, "spec.json")
    rec = XP._read_json(os.path.join(run, "receipt_audit.json"))
    if P.verify_receipt(os.path.join(run, "receipt_audit.json"), cfg["project_root"]):
        raise RuntimeError("Stage 10C audit receipt does not verify")
    spec = {"spec_version": "xai_sim_stats_10d_v1", "created_utc": _now(),
            "input": {"run": R10C, "spec_sha256": ssha, "audit_receipt_outputs": rec["outputs"]},
            "scope": "descriptive; no hypothesis tests (no simulated-arm testing family prespecified); Derm7pt Holm family untouched",
            "estimands": __doc__.split("Estimands")[1].split("Retrospective")[0].strip(),
            "retrospective_additions": ["bootstrap seed 20261200 (one shared lesion draw per replicate)",
                                        "paired change-from-severity-0 summaries on exactly the images eligible at both conditions",
                                        "IG-minus-Grad-CAM comparison on the per-condition eligibility intersection",
                                        "localisation summaries: median area-matched IoU, pointing rate, explainer-minus-control paired "
                                        "differences, change from severity 0 on images available at both",
                                        "as-computed sensitivity summaries (finite non-primary maps; constant maps ranked by flat pixel index)",
                                        "mismatch-control summaries are point estimates only (donor dependence not captured by "
                                        "recipient-lesion resampling)"],
            "bootstrap": {"B": B, "seed": SEED, "unit": "lesion (cluster_id); all images and all 11 conditions of a drawn lesion "
                                                         "together; multiplicity kept", "ci": "percentile 2.5/97.5, numpy linear",
                          "undefined": "replicates with no weight in a population are counted, never redrawn",
                          "pointwise": "intervals are pointwise, not simultaneous across conditions"},
            "stats": {"faithfulness": ["median", "hl"], "localisation_iou": ["median"], "pointing": ["mean"]},
            "sources": {n: P.sha256_file(os.path.join(os.path.dirname(__file__), n)) for n in
                        ("xai_sim_stats.py", "xai_controls.py", "rq4_stats.py", "xai_sim.py", "xai_sim_prod.py")}}
    from . import provenance as Pv
    receipt = Pv.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "analysis_spec.json", spec)
    publish_bytes(out, "analysis_spec.json.sha256", f"{P.sha256_file(os.path.join(out, 'analysis_spec.json'))}  analysis_spec.json\n".encode())
    receipt["outputs"] = {"analysis_spec.json": P.sha256_file(os.path.join(out, "analysis_spec.json"))}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out)


def _load_inputs(cfg, spec):
    from . import provenance as P, xai_controls as XK, xai_prod as XP, xai_sim as XS
    run = _abs(cfg, R10C)
    s10, ssha = XP._load(run, "spec.json")
    if ssha != spec["input"]["spec_sha256"]:
        raise RuntimeError("10C spec changed")
    for n, h in spec["input"]["audit_receipt_outputs"].items():
        if P.sha256_file(os.path.join(run, n)) != h:
            raise RuntimeError(f"10C output changed {n}")
    recs, mm = {}, {}
    for ch in s10["batching"]["chunks"]:
        for pre, store in (("chunk", recs), ("mismatch", mm)):
            d = os.path.join(run, f"{pre}_{ch['chunk']:03d}")
            if not XP.verify_dir(d, ssha):
                raise RuntimeError(f"{d} does not verify")
            for r in XP._read_json(os.path.join(d, "records.json")):
                if r["ref_key"] in store:
                    raise RuntimeError("duplicate")
                store[r["ref_key"]] = r
    imgs = [r["record_uid"] for r in s10["records"]]
    labels = [r["label_mel"] for r in s10["records"]]
    clusters = [r["cluster_id"] for r in s10["records"]]
    conds = [c["key"] for c in s10["conditions"]]
    n, C = len(imgs), len(conds)
    nan = lambda: np.full((C, n), np.nan)
    D = {}
    for e in EXPL:
        for c_ in CURVES:
            for ctl in ("random", "centre", "sobel", "mismatch"):
                D[f"{e}|{c_}|{ctl}"] = nan()
            D[f"{e}|{c_}|random|as_computed"] = nan()
        D[f"{e}|status"] = np.full((C, n), "", dtype=object)
        for k in ("iou_area", "pointing_hit"):
            D[f"loc|{e}|{k}"] = nan()
            D[f"loc|mismatch_{e}|{k}"] = nan()
    for k in ("iou_area", "pointing_hit"):
        for ctl in CTRL_LOC:
            D[f"loc|{ctl}|{k}"] = nan()
    D["mask_fraction"] = nan()
    for ci, c in enumerate(conds):
        f, s = c.split("|")
        for ii, u in enumerate(imgs):
            r = recs[XS.ref_key(u, f, float(s))]
            m = mm[r["ref_key"]]
            D["mask_fraction"][ci, ii] = r["mask"]["fraction"]
            for e in EXPL:
                ex = r["explainers"][e]
                D[f"{e}|status"][ci, ii] = ex["status"]
                prim = ex["status"] == "primary"
                pc = ex.get("primary_curve") or {}
                expl = {"auc_ins": pc.get("auc_insertion_v1") if prim else None, "auc_del": pc.get("auc_deletion_v1") if prim else None}
                for ctl in ("random", "centre", "sobel"):
                    sc = XK.control_scores(expl, r["controls"][ctl])
                    if sc["status"] == "available":
                        D[f"{e}|ins|{ctl}"][ci, ii] = sc["insertion_advantage"]
                        D[f"{e}|del|{ctl}"][ci, ii] = sc["deletion_advantage"]
                sc = XK.control_scores(expl, m[e]) if m[e].get("status") == "available" else {"status": "unavailable"}
                if sc["status"] == "available":
                    D[f"{e}|ins|mismatch"][ci, ii] = sc["insertion_advantage"]
                    D[f"{e}|del|mismatch"][ci, ii] = sc["deletion_advantage"]
                cv = pc if prim else (ex.get("as_computed_curve") or {})
                if cv.get("curve_status") == "ok":
                    sc = XK.control_scores({"auc_ins": cv["auc_insertion_v1"], "auc_del": cv["auc_deletion_v1"]}, r["controls"]["random"])
                    if sc["status"] == "available":
                        D[f"{e}|ins|random|as_computed"][ci, ii] = sc["insertion_advantage"]
                        D[f"{e}|del|random|as_computed"][ci, ii] = sc["deletion_advantage"]
                L = r["localisation"][e]
                if L["status"] == "available":
                    D[f"loc|{e}|iou_area"][ci, ii] = L["iou_area"]
                    D[f"loc|{e}|pointing_hit"][ci, ii] = float(L["pointing_hit"])
                ml = m[e].get("localisation") if m[e].get("status") == "available" else None
                if ml and ml["status"] == "available":
                    D[f"loc|mismatch_{e}|iou_area"][ci, ii] = ml["iou_area"]
                    D[f"loc|mismatch_{e}|pointing_hit"][ci, ii] = float(ml["pointing_hit"])
            for ctl in CTRL_LOC:
                L = r["localisation"][ctl]
                if L["status"] == "available":
                    D[f"loc|{ctl}|iou_area"][ci, ii] = L["iou_area"]
                    D[f"loc|{ctl}|pointing_hit"][ci, ii] = float(L["pointing_hit"])
    return s10, recs, imgs, labels, clusters, conds, D


def phase_analyze(cfg, a):
    from . import provenance as P, xai_prod as XP
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    t0 = time.perf_counter()
    out = os.path.abspath(a.run_dir)
    spec, ssha = XP._load(out, "analysis_spec.json")
    for n, h in spec["sources"].items():
        if P.sha256_file(os.path.join(os.path.dirname(__file__), n)) != h:
            raise RuntimeError(f"source changed {n}")
    s10, recs, imgs, labels, clusters, conds, D = _load_inputs(cfg, spec)
    # reproduce Stage 10C audit counts
    audit_ig = {r["group"]: int(r["ig_completeness_failures"]) for r in XP._read_csv_rows(os.path.join(_abs(cfg, R10C), "ig_failure_table.csv"))}
    ig_fail = sum(r["ig_completeness"]["failure"] for r in recs.values())
    gc_excl = int((D["gradcam|status"] != "primary").sum())
    audit_joint = {(r["condition"], r["explainer"]): int(r["jointly_eligible_with_random_control"])
                   for r in XP._read_csv_rows(os.path.join(_abs(cfg, R10C), "joint_eligibility_table.csv"))}
    W = lesion_weights(clusters, B, SEED)
    est, pops, chg, cmp_, loc, sens, elig = [], [], [], [], [], [], []
    for e in EXPL:
        for ci, c in enumerate(conds):
            st = D[f"{e}|status"][ci]
            for lab in ("1", "0", "all"):
                sel = np.array([lab == "all" or l_ == lab for l_ in labels])
                reasons = {}
                for k, ok in zip(st[sel], sel[sel]):
                    if k != "primary":
                        reasons[k] = reasons.get(k, 0) + 1
                elig.append({"explainer": e, "condition": c, "class": lab, "n": int(sel.sum()), "primary": int((st[sel] == "primary").sum()),
                             "excluded": int((st[sel] != "primary").sum()), "rate_excluded": float((st[sel] != "primary").mean()),
                             "status_counts": json.dumps(reasons)})
            for cur in CURVES:
                v = D[f"{e}|{cur}|random"][ci]
                m = ~np.isnan(v)
                s = summarise(v, m, W)
                pops.append({"analysis": "level", "explainer": e, "condition": c, "curve": cur, **population(m, labels, clusters)})
                est.append({"explainer": e, "condition": c, "curve": cur, "control": "random", **population(m, labels, clusters),
                            "median": s["median"], "median_ci95": json.dumps(s["median_ci95"]), "hl": s["hl"], "hl_ci95": json.dumps(s["hl_ci95"]),
                            "undefined_replicates": s["undefined_replicates"]})
                for ctl in ("centre", "sobel", "mismatch"):
                    vv = D[f"{e}|{cur}|{ctl}"][ci]
                    mm_ = ~np.isnan(vv)
                    est.append({"explainer": e, "condition": c, "curve": cur, "control": ctl, **population(mm_, labels, clusters),
                                "median": float(np.median(vv[mm_])) if mm_.any() else None, "median_ci95": "", "hl": hl(vv[mm_]) if mm_.any() else None,
                                "hl_ci95": "", "undefined_replicates": "",
                                "note": "point estimate only" + ("; donor dependence not captured" if ctl == "mismatch" else "")})
                va = D[f"{e}|{cur}|random|as_computed"][ci]
                ma = ~np.isnan(va)
                sens.append({"explainer": e, "condition": c, "curve": cur, **population(ma, labels, clusters),
                             "median_as_computed": float(np.median(va[ma])), "hl_as_computed": hl(va[ma]),
                             "median_primary": s["median"], "hl_primary": s["hl"], "n_primary": s["n"]})
                if ci > 0:
                    v0 = D[f"{e}|{cur}|random"][0]
                    both = ~np.isnan(v) & ~np.isnan(v0)
                    if cur == "ins" and int(both.sum()) != audit_joint[(c, e)]:
                        raise RuntimeError(f"joint population differs from 10C audit {c} {e}")
                    sb, sp, sd = summarise(v0, both, W), summarise(v, both, W), summarise(v - v0, both, W)
                    chg.append({"explainer": e, "condition": c, "curve": cur, **population(both, labels, clusters),
                                "baseline_hl": sb["hl"], "baseline_hl_ci95": json.dumps(sb["hl_ci95"]), "baseline_median": sb["median"],
                                "condition_hl": sp["hl"], "condition_hl_ci95": json.dumps(sp["hl_ci95"]), "condition_median": sp["median"],
                                "paired_change_hl": sd["hl"], "paired_change_hl_ci95": json.dumps(sd["hl_ci95"]),
                                "paired_change_median": sd["median"], "paired_change_median_ci95": json.dumps(sd["median_ci95"]),
                                "undefined_replicates": sd["undefined_replicates"]})
    for ci, c in enumerate(conds):
        for cur in CURVES:
            vi, vg = D[f"ig2048|{cur}|random"][ci], D[f"gradcam|{cur}|random"][ci]
            both = ~np.isnan(vi) & ~np.isnan(vg)
            s = summarise(vi - vg, both, W)
            cmp_.append({"condition": c, "curve": cur, "contrast": "IG2048 minus Grad-CAM", **population(both, labels, clusters),
                         "hl": s["hl"], "hl_ci95": json.dumps(s["hl_ci95"]), "median": s["median"], "median_ci95": json.dumps(s["median_ci95"]),
                         "undefined_replicates": s["undefined_replicates"]})
    # localisation (matched populations)
    for e in EXPL:
        for ci, c in enumerate(conds):
            keys = [e, *CTRL_LOC]
            m = np.all([~np.isnan(D[f"loc|{k}|iou_area"][ci]) for k in keys], axis=0)
            row = {"explainer": e, "condition": c, **population(m, labels, clusters)}
            for k in keys:
                si = summarise(D[f"loc|{k}|iou_area"][ci], m, W, ("median",))
                sp = summarise(D[f"loc|{k}|pointing_hit"][ci], m, W, ("mean",))
                row[f"{k}_iou_median"], row[f"{k}_iou_ci95"] = si["median"], json.dumps(si["median_ci95"])
                row[f"{k}_pointing"], row[f"{k}_pointing_ci95"] = sp["mean"], json.dumps(sp["mean_ci95"])
            for k in CTRL_LOC:
                sd = summarise(D[f"loc|{e}|iou_area"][ci] - D[f"loc|{k}|iou_area"][ci], m, W, ("median",))
                pd_ = summarise(D[f"loc|{e}|pointing_hit"][ci] - D[f"loc|{k}|pointing_hit"][ci], m, W, ("mean",))
                row[f"iou_diff_vs_{k}_median"], row[f"iou_diff_vs_{k}_ci95"] = sd["median"], json.dumps(sd["median_ci95"])
                row[f"pointing_diff_vs_{k}"], row[f"pointing_diff_vs_{k}_ci95"] = pd_["mean"], json.dumps(pd_["mean_ci95"])
            mmv = D[f"loc|mismatch_{e}|iou_area"][ci]
            mmm = m & ~np.isnan(mmv)
            row["mismatch_n"], row["mismatch_iou_median_point"] = int(mmm.sum()), float(np.median(mmv[mmm])) if mmm.any() else None
            row["mismatch_pointing_point"] = float(np.nanmean(D[f"loc|mismatch_{e}|pointing_hit"][ci][mmm])) if mmm.any() else None
            row["mask_fraction_median"] = float(np.median(D["mask_fraction"][ci][m])) if m.any() else None
            if ci > 0:
                b2 = ~np.isnan(D[f"loc|{e}|iou_area"][ci]) & ~np.isnan(D[f"loc|{e}|iou_area"][0])
                sc = summarise(D[f"loc|{e}|iou_area"][ci] - D[f"loc|{e}|iou_area"][0], b2, W, ("median",))
                row.update(change_n=int(b2.sum()), iou_change_from_s0_median=sc["median"], iou_change_from_s0_ci95=json.dumps(sc["median_ci95"]))
            loc.append(row)
    checks = {"ig_completeness_failures": ig_fail, "audit_ig_pooled": audit_ig["pooled"], "gradcam_excluded": gc_excl,
              "joint_populations_match_10c_audit": True, "references": len(recs)}
    if ig_fail != audit_ig["pooled"] or len(recs) != 3740:
        raise RuntimeError(f"audit reproduction failed {checks}")
    publish_json(out, "results.json", {"checks": checks, "bootstrap": spec["bootstrap"], "runtime_s": time.perf_counter() - t0,
                                       "units_note": "3,740 references = 340 images x 11 conditions; not independent observations"})
    for name, rows in (("faithfulness_by_condition.csv", est), ("change_from_s0.csv", chg), ("ig_vs_gradcam.csv", cmp_),
                       ("localisation_matched.csv", loc), ("as_computed_sensitivity.csv", sens), ("eligibility_by_condition_class.csv", elig),
                       ("populations.csv", pops)):
        publish_csv(out, name, rows, sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 999))
    for e in EXPL:
        publish_bytes(out, f"plot_faithfulness_{e}.png", _plot_faith([r for r in est if r["explainer"] == e and r["control"] == "random"], e))
    publish_bytes(out, "plot_localisation.png", _plot_loc(loc))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze"})
    receipt["outputs"] = {n_: P.sha256_file(os.path.join(out, n_)) for n_ in sorted(os.listdir(out))
                          if os.path.isfile(os.path.join(out, n_)) and not n_.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    print(checks, "runtime", round(time.perf_counter() - t0, 1), "verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def _plot_faith(rows, e):
    from PIL import Image, ImageDraw
    W_, H_, L, T = 1100, 420, 70, 50
    img = Image.new("RGB", (W_, H_ * 2), "white")
    d = ImageDraw.Draw(img)
    for p, cur in enumerate(CURVES):
        rr = [r for r in rows if r["curve"] == cur]
        vals = [x for r in rr for x in json.loads(r["hl_ci95"])] + [0.0]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.08 or 0.01
        lo, hi = lo - pad, hi + pad
        y0 = p * H_
        Y = lambda v: y0 + T + (hi - v) / (hi - lo) * (H_ - T - 70)
        d.text((L, y0 + 10), f"{e}: Delta_{cur} vs random (HL, 95% lesion-bootstrap CI); n images per condition below; >0 = better than random", fill="black")
        d.line([L, Y(0), W_ - 20, Y(0)], fill=(170, 170, 170))
        for k in range(5):
            v = lo + k * (hi - lo) / 4
            d.text((5, Y(v) - 6), f"{v:.3f}", fill="black")
        for i, r in enumerate(rr):
            x = L + 30 + i * (W_ - L - 60) / max(len(rr) - 1, 1)
            a, b = json.loads(r["hl_ci95"])
            d.line([x, Y(a), x, Y(b)], fill=(31, 119, 180), width=2)
            d.ellipse([x - 4, Y(r["hl"]) - 4, x + 4, Y(r["hl"]) + 4], fill=(31, 119, 180))
            d.text((x - 30, y0 + H_ - 60), r["condition"].replace("|", " ").replace("illumination", "illum").replace("compression", "jpeg").replace("nuisance", "nuis"), fill="black")
            d.text((x - 12, y0 + H_ - 45), f"n={r['images']}", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _plot_loc(loc):
    from PIL import Image, ImageDraw
    W_, H_, L, T = 1100, 420, 70, 50
    img = Image.new("RGB", (W_, H_ * 2), "white")
    d = ImageDraw.Draw(img)
    cols = {"ig2048": (31, 119, 180), "gradcam": (214, 39, 40), "random_mean": (127, 127, 127), "centre": (44, 160, 44), "sobel": (255, 127, 14)}
    for p, e in enumerate(EXPL):
        rr = [r for r in loc if r["explainer"] == e]
        y0 = p * H_
        Y = lambda v: y0 + T + (1 - v) * (H_ - T - 70)
        d.text((L, y0 + 10), f"Matched population for {e}: median area-matched IoU (uses true mask area); n images per condition below", fill="black")
        for k in range(6):
            d.text((5, Y(k / 5) - 6), f"{k / 5:.1f}", fill="black")
        for j, k in enumerate((e, "random_mean", "centre", "sobel")):
            d.text((W_ - 200, y0 + 30 + 14 * j), k, fill=cols[k])
            pts = []
            for i, r in enumerate(rr):
                x = L + 30 + i * (W_ - L - 260) / max(len(rr) - 1, 1)
                a, b = json.loads(r[f"{k}_iou_ci95"])
                d.line([x + 3 * j, Y(a), x + 3 * j, Y(b)], fill=cols[k])
                pts.append((x + 3 * j, Y(r[f"{k}_iou_median"])))
                if j == 0:
                    d.text((x - 30, y0 + H_ - 60), r["condition"].replace("|", " ").replace("illumination", "illum").replace("compression", "jpeg").replace("nuisance", "nuis"), fill="black")
                    d.text((x - 12, y0 + H_ - 45), f"n={r['images']}", fill="black")
            d.line(pts, fill=cols[k], width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "analyze"))
    ap.add_argument("--out-root")
    ap.add_argument("--run-dir")
    a = ap.parse_args()
    from . import config as C
    cfg = C.load(a.config)
    if cfg["purpose"] != "xai_diag":
        raise C.ConfigError("requires purpose 'xai_diag'")
    {"spec": phase_spec, "analyze": phase_analyze}[a.phase](cfg, a)


if __name__ == "__main__":
    main()
