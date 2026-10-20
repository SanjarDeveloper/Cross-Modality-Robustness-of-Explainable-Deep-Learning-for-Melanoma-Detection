"""Stage 9G2: RQ2 perturbation sweeps for the nine aug_heavy_v1 models and matched standard-vs-heavy
robustness contrasts. Phases (separate processes): spec -> sweep -> analyze -> compare.

spec    : new. Heavy registry from Stage 9G1 (checkpoint, strategy, frozen threshold, own unperturbed HAM TEST
          reference), written in the Stage 9C2 spec format.
sweep   : the UNCHANGED, tested rq3_rq2.phase_sweep (unit-major; each unit transformed once from the original
          cached images; audit digests vs Stage 8B; severity-0 baseline check; hashed atomic unit dirs; resumable).
analyze : the UNCHANGED rq3_rq2.phase_analyze (Stage 8C definitions; bootstrap seed 20261050 = the Stage 8C/9C2
          draws, shared across all models). Its ResNet-s0/Stage 8C reproduction block is not applicable here.
compare : new. All 18 cells re-bootstrapped in ONE loop with the same draws; standard results must reproduce
          Stage 9C2 and heavy results this run's analyze output; per replicate: both worst losses (incl. s=0),
          benefit = worst_std - worst_heavy, both grid minima (absolute) and their difference; censored crossings
          compared ordinally only.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq3h_rq2 --config ../configs/rq3h_rq2_9g2_v1.json --phase spec --out-root ../runs/rq3h_rq2_9g2
  ... --phase sweep|analyze|compare --run-dir <dir>
"""
import argparse
import io
import json
import os
import time

import numpy as np

from . import rq3_rq2 as R

R9G1 = "research_v1/runs/rq3_heavy_eval_9g1/20261004T152428Z_2cea4004_64df5fc0/"
R9C2 = "research_v1/runs/rq3_rq2_9c2/20261004T103829Z_73f9ae39_a3207e31/"
OVERLAP = {"blur": "overlaps heavy recipe (GaussianBlur / MotionBlur / Defocus)",
           "illumination": "overlaps heavy recipe (ColorJitter / RandomBrightnessContrast)",
           "compression": "overlaps heavy recipe (ImageCompression JPEG 30-90)",
           "scale": "partial overlap (RandomResizedCrop zoom-in, ShiftScaleRotate 0.75-1.25 with black fill; RQ2 zoom-out to 2.4x with skin-tone surround is outside that range)",
           "nuisance": "not in the recipe (glare / hair occlusion); GaussNoise is in the recipe but is not an RQ2 nuisance component"}
TOL_REPRO = 1e-9


# ------------------------------------------------------------------ pure helpers (tested)
def match(heavy, standard):
    out = {}
    for h in heavy:
        m = [s for s in standard if s["arch"] == h["arch"] and s["seed"] == h["seed"]]
        if len(m) != 1:
            raise RuntimeError(f"{h['id']}: no unique standard match")
        out[h["id"]] = m[0]
    return out


def worst_reps(rep_by_level):
    """rep_by_level: list over GRID of arrays (B,) with NaN = undefined. Returns per-replicate worst loss
    (Stage 8C rule, incl. s=0) and per-replicate grid minimum (absolute); NaN if any level undefined."""
    from . import rq2_stats as R2
    M = np.vstack(rep_by_level)
    w = np.array([np.nan if np.isnan(M[:, b]).any() else R2.worst_case([float(v) for v in M[:, b]])[0] for b in range(M.shape[1])])
    mn = np.where(np.isnan(M).any(axis=0), np.nan, M.min(axis=0))
    return w, mn


def benefit(w_std, w_heavy):
    """Robustness benefit per replicate: worst_loss_standard - worst_loss_heavy (positive = smaller loss under heavy)."""
    return np.asarray(w_std) - np.asarray(w_heavy)


def crossing_order(cs, ch):
    """Ordinal comparison of two crossing markers (float, '>1.0' censored, or None). Never subtracts.
    Returns one of: undefined, both_censored, heavy_later_censored, heavy_earlier_censored, heavy_later, heavy_earlier, equal."""
    from .rq2_stats import CENSORED
    if cs is None or ch is None:
        return "undefined"
    if cs == CENSORED and ch == CENSORED:
        return "both_censored"
    if ch == CENSORED:
        return "heavy_later_censored"
    if cs == CENSORED:
        return "heavy_earlier_censored"
    return "heavy_later" if ch > cs else ("heavy_earlier" if ch < cs else "equal")


ORDER_CATS = ("heavy_later", "heavy_later_censored", "equal", "both_censored", "heavy_earlier", "heavy_earlier_censored", "undefined")


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return np.quantile(v, [0.025, 0.975], method="linear").tolist() if len(v) else [None, None]


# ------------------------------------------------------------------ phases
def phase_spec(cfg, a):
    from . import provenance as P
    import torch
    from .desc_4b3 import publish_bytes, publish_json
    from .rq3_heavy_eval import check_strategy
    g = R._json(R._abs(cfg, R9G1 + "spec.json"))
    reg = R._json(R._abs(cfg, R9G1 + "threshold_registry.json"))
    for rn in ("receipt_spec.json", "receipt_threshold.json", "receipt_infer.json", "receipt_analyze.json", "receipt_compare.json"):
        rp = R._abs(cfg, R9G1 + rn)
        if P.verify_receipt(rp, cfg["project_root"]):
            raise RuntimeError(f"9G1 {rn} does not verify")
    s9c2 = R._json(R._abs(cfg, R9C2 + "spec.json"))
    rec = R._json(R._abs(cfg, R9C2 + "receipt_analyze.json"))
    for n, h in rec["outputs"].items():
        if P.sha256_file(R._abs(cfg, R9C2 + n)) != h:
            raise RuntimeError(f"9C2 changed {n}")
    models = []
    for e in g["models"]:
        t = R.bind(e, reg)
        ck = torch.load(os.path.join(R._abs(cfg, e["run"]), e["checkpoint"]), map_location="cpu", weights_only=False)
        check_strategy(ck, "aug_heavy_v1")
        if P.sha256_file(os.path.join(R._abs(cfg, e["run"]), e["checkpoint"])) != e["sha256"] or ck["epoch_completed"] != e["epoch"]:
            raise RuntimeError(f"{e['id']}: checkpoint identity")
        ref = R9G1 + f"pred_{e['id']}_ham_test.csv"
        models.append({k: e[k] for k in ("id", "arch", "seed", "run", "checkpoint", "epoch", "sha256")} |
                      {"strategy": "aug_heavy_v1", "threshold": t, "reference_ham_test": {"path": ref, "sha256": P.sha256_file(R._abs(cfg, ref))},
                       "sweep": "new (Stage 9G2)"})
    pairs = match(models, s9c2["models"])
    srcs = dict(s9c2["sources"])
    for n, h in srcs.items():
        if P.sha256_file(os.path.join(os.path.dirname(__file__), n)) != h:
            raise RuntimeError(f"source changed since 9C2: {n}")
    spec = {"spec_version": "rq3h_rq2_9g2_v1", "created_utc": R._now(), "models": models,
            "threshold_registry": {"path": R9G1 + "threshold_registry.json", "sha256": P.sha256_file(R._abs(cfg, R9G1 + "threshold_registry.json"))},
            "stage9g1_spec_sha256": P.sha256_file(R._abs(cfg, R9G1 + "spec.json")),
            "stage8b": s9c2["stage8b"], "simulator": s9c2["simulator"], "grid": s9c2["grid"], "pipeline": s9c2["pipeline"], "sources": srcs, "orchestrator": {"rq3h_rq2.py_sha256_at_spec": P.sha256_file(__file__), "rq3_heavy_eval.py": P.sha256_file(os.path.join(os.path.dirname(__file__), "rq3_heavy_eval.py"))},
            "manifest": {"path": cfg["manifests"]["ham10000"], "sha256": P.sha256_file(R._abs(cfg, cfg["manifests"]["ham10000"]))},
            "baseline_tolerance": R.BASELINE_TOL,
            "perturbation_identity": s9c2["perturbation_identity"] + "; identical to the Stage 9C2 standard sweeps",
            "coverage": {"per_model_rows": 69660, "new_models": 9, "new_rows": 9 * 69660, "all_18_cells_rows": 18 * 69660},
            "bootstrap": s9c2["bootstrap"],
            "statistics": s9c2["statistics"],
            "standard_reference": {"run": R9C2, "spec_sha256": P.sha256_file(R._abs(cfg, R9C2 + "spec.json")),
                                   "analyze_receipt_outputs": rec["outputs"],
                                   "matches": {h: {"id": v["id"], "sha256": v["sha256"], "threshold": v["threshold"]} for h, v in pairs.items()}},
            "contrasts": {"frozen_before_analysis": True,
                          "benefit": "worst_loss_standard - worst_loss_heavy per replicate (both maxima recomputed per replicate incl. s=0); positive = smaller worst-case loss under heavy",
                          "absolute": "grid minimum (incl. s=0) per replicate for both; d_min = min_heavy - min_standard; baselines and every level's absolute sens/spec/AUROC reported side by side",
                          "crossing": "both markers and censoring shares side by side; per-replicate ordinal comparison only (censored > any finite; never subtracted)",
                          "metrics": ["sensitivity (pp)", "AUROC"], "B": R.B, "draws": "seed 20261050, one loop over all 18 cells",
                          "reproduction_tolerance": TOL_REPRO,
                          "interpretation": "descriptive, conditional on fitted models and thresholds; no training variability; no significance tests"},
            "factor_overlap_with_heavy_recipe": OVERLAP}
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "spec.json", spec)
    publish_bytes(out, "spec.json.sha256", f"{P.sha256_file(os.path.join(out, 'spec.json'))}  spec.json\n".encode())
    receipt["outputs"] = {"spec.json": P.sha256_file(os.path.join(out, "spec.json"))}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out)
    for m in models:
        print(m["id"], m["threshold"], "->", pairs[m["id"]]["id"], pairs[m["id"]]["threshold"])


def _arrays(cfg, out, spec, ssha):
    data, ident0 = {}, None
    for m in spec["models"]:
        units = R._model_units(cfg, out, m, ssha)
        arr = {}
        for k, rr in units.items():
            ident = [(r["record_uid"], r["label_mel"], r["cluster_id"]) for r in rr]
            ident0 = ident0 or ident
            if ident != ident0 or any(r["checkpoint_sha256"] != m["sha256"] for r in rr):
                raise RuntimeError(f"{m['id']} {k}: identity")
            arr[k] = np.array([float(r["prob"]) for r in rr])
        data[m["id"]] = arr
    return data, ident0


def phase_compare(cfg, a):
    from . import provenance as P, rq2_stats as R2
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    from .rq2_predict import FACTORS, SEVERITIES
    t0 = time.perf_counter()
    out = os.path.abspath(a.run_dir)
    spec, ssha = R._load_spec(cfg, out)
    rec = R._json(os.path.join(out, "receipt_analyze.json"))
    for n, h in rec["outputs"].items():
        if P.sha256_file(os.path.join(out, n)) != h:
            raise RuntimeError(f"changed {n}")
    sdir = R._abs(cfg, R9C2)
    sspec, sssha = R._load_spec(cfg, sdir)
    if P.sha256_file(os.path.join(sdir, "spec.json")) != spec["standard_reference"]["spec_sha256"]:
        raise RuntimeError("standard reference changed")
    for n, h in spec["standard_reference"]["analyze_receipt_outputs"].items():
        if P.sha256_file(os.path.join(sdir, n)) != h:
            raise RuntimeError(f"9C2 output changed {n}")
    dh, ih = _arrays(cfg, out, spec, ssha)
    ds, is_ = _arrays(cfg, sdir, sspec, sssha)
    if ih != is_:
        raise RuntimeError("standard and heavy record identity differ")
    models = {m["id"]: dict(m, strategy="aug_heavy_v1") for m in spec["models"]} | {m["id"]: dict(m, strategy="aug_std_v1") for m in sspec["models"]}
    data = dh | ds
    if len(models) != 18 or sum(len(v) for v in data.values()) != 18 * 45 or any(len(p) != 1548 for v in data.values() for p in v.values()):
        raise RuntimeError("18-cell coverage incomplete")
    coverage = {"cells": 18, "units_per_cell": 45, "rows_per_cell": 45 * 1548, "logical_rows": 18 * 45 * 1548}
    y = np.array([r[1] == "1" for r in ih])
    _, groups = R2.lesion_index([r[2] for r in ih])
    rng = np.random.Generator(np.random.PCG64(spec["bootstrap"]["seed"]))
    rep = {mid: {k: {"sens": np.full(R.B, np.nan), "auc": np.full(R.B, np.nan)} for k in arr} for mid, arr in data.items()}
    for b in range(R.B):
        idx = R2.draw_rows(groups, rng)
        yb = y[idx]
        for mid, arr in data.items():
            th = models[mid]["threshold"]
            for k, p in arr.items():
                if k[1] == 0.0 and k[0] != "blur":
                    continue
                pb = p[idx]
                rep[mid][k]["sens"][b] = (pb[yb] >= th).mean() if yb.any() else np.nan
                au = R2.auc_ranks(pb, yb)
                rep[mid][k]["auc"][b] = np.nan if au is None else au
    for mid in rep:
        for f in FACTORS[1:]:
            rep[mid][(f, 0.0)] = rep[mid][("blur", 0.0)]
    # points
    pts = {}
    for mid, arr in data.items():
        th = models[mid]["threshold"]
        for k, p in arr.items():
            pts[(mid, k)] = {"sens": float((p[y] >= th).mean()), "spec": float((p[~y] < th).mean()), "auc": R2.auc_ranks(p, y)}
    # reproduction against published worst-case tables
    W = {}
    for mid in data:
        for f in FACTORS:
            for met in ("sens", "auc"):
                W[(mid, f, met)] = worst_reps([rep[mid][(f, s)][met] for s in SEVERITIES])
    repro = {"standard_vs_9C2": 0.0, "heavy_vs_9G2_analyze": 0.0}
    for src, path in (("standard_vs_9C2", os.path.join(sdir, "worst_case_table.csv")), ("heavy_vs_9G2_analyze", os.path.join(out, "worst_case_table.csv"))):
        for r in R._read(path):
            sc = 100 if r["metric"] == "sens" else 1
            w = W[(r["model"], r["factor"], r["metric"])][0]
            obs = R2.worst_case([pts[(r["model"], (r["factor"], s))][r["metric"]] for s in SEVERITIES])[0]
            c = ci(w)
            repro[src] = max(repro[src], abs(obs * sc - float(r["worst_loss"])), abs(c[0] * sc - float(r["ci_lo"])), abs(c[1] * sc - float(r["ci_hi"])))
    if max(repro.values()) > TOL_REPRO:
        raise RuntimeError(f"reproduction failed {repro}")
    pairs = spec["standard_reference"]["matches"]
    wc_rows, lvl_rows, cr_rows, results = [], [], [], {}
    for hid, sref in pairs.items():
        sid, hm = sref["id"], models[hid]
        for f in FACTORS:
            for met in ("sens", "auc"):
                sc = 100 if met == "sens" else 1
                ws, ms = W[(sid, f, met)]
                wh, mh = W[(hid, f, met)]
                bn = benefit(ws, wh)
                dmin = mh - ms
                b0s, b0h = rep[sid][(f, 0.0)][met], rep[hid][(f, 0.0)][met]
                vs = [pts[(sid, (f, s))][met] for s in SEVERITIES]
                vh = [pts[(hid, (f, s))][met] for s in SEVERITIES]
                os_, ls = R2.worst_case(vs)
                oh, lh = R2.worst_case(vh)
                und = int((~np.isfinite(bn)).sum())
                wc_rows.append({"arch": hm["arch"], "seed": hm["seed"], "factor": f, "overlap": OVERLAP[f].split(" (")[0], "metric": met,
                                "threshold_std": models[sid]["threshold"], "threshold_heavy": hm["threshold"],
                                "baseline_std": vs[0] * sc, "baseline_heavy": vh[0] * sc, "d_baseline": (vh[0] - vs[0]) * sc,
                                "d_baseline_ci": json.dumps([x * sc for x in ci(b0h - b0s)]),
                                "worst_loss_std": os_ * sc, "worst_loss_std_ci": json.dumps([x * sc for x in ci(ws)]), "worst_level_std": ls,
                                "worst_loss_heavy": oh * sc, "worst_loss_heavy_ci": json.dumps([x * sc for x in ci(wh)]), "worst_level_heavy": lh,
                                "benefit": (os_ - oh) * sc, "benefit_ci": json.dumps([x * sc for x in ci(bn)]),
                                "share_reps_benefit_pos": float((bn[np.isfinite(bn)] > 0).mean()),
                                "min_abs_std": min(vs) * sc, "min_abs_heavy": min(vh) * sc, "d_min_abs": (min(vh) - min(vs)) * sc,
                                "d_min_abs_ci": json.dumps([x * sc for x in ci(dmin)]), "n_undefined": und})
            for s in SEVERITIES:
                a_, b_ = pts[(sid, (f, s))], pts[(hid, (f, s))]
                lvl_rows.append({"arch": hm["arch"], "seed": hm["seed"], "factor": f, "severity": s,
                                 "sens_std": a_["sens"], "sens_heavy": b_["sens"], "spec_std": a_["spec"], "spec_heavy": b_["spec"],
                                 "auroc_std": a_["auc"], "auroc_heavy": b_["auc"],
                                 "d_sens_pp_std": (a_["sens"] - pts[(sid, (f, 0.0))]["sens"]) * 100,
                                 "d_sens_pp_heavy": (b_["sens"] - pts[(hid, (f, 0.0))]["sens"]) * 100,
                                 "d_auroc_std": a_["auc"] - pts[(sid, (f, 0.0))]["auc"], "d_auroc_heavy": b_["auc"] - pts[(hid, (f, 0.0))]["auc"]})
            cnt = {c: 0 for c in ORDER_CATS}
            mk = {}
            for mid in (sid, hid):
                cv = [R2.crossing([None if np.isnan(rep[mid][(f, s)]["sens"][b]) else float(rep[mid][(f, s)]["sens"][b]) for s in SEVERITIES])
                      for b in range(R.B)]
                mk[mid] = cv
            for b in range(R.B):
                cnt[crossing_order(mk[sid][b], mk[hid][b])] += 1
            obs_s = R2.crossing([pts[(sid, (f, s))]["sens"] for s in SEVERITIES])
            obs_h = R2.crossing([pts[(hid, (f, s))]["sens"] for s in SEVERITIES])
            row = {"arch": hm["arch"], "seed": hm["seed"], "factor": f, "crossing_std": obs_s, "crossing_heavy": obs_h,
                   "observed_order": crossing_order(obs_s, obs_h)}
            for mid, lab in ((sid, "std"), (hid, "heavy")):
                v = [x for x in mk[mid] if x is not None]
                cq = R2.censored_quantiles(v)
                row |= {f"no_crossing_share_{lab}": sum(x == R2.CENSORED for x in v) / len(v),
                        f"q025_{lab}": cq[0.025], f"q50_{lab}": cq[0.5], f"q975_{lab}": cq[0.975]}
            row |= {f"share_{c}": cnt[c] / R.B for c in ORDER_CATS}
            cr_rows.append(row)
    ranges = []
    for arch in ("resnet50", "efficientnet_b3", "densenet121"):
        for f in FACTORS:
            for met in ("sens", "auc"):
                rr = sorted((r for r in wc_rows if r["arch"] == arch and r["factor"] == f and r["metric"] == met), key=lambda r: r["seed"])
                for q in ("baseline_std", "baseline_heavy", "worst_loss_std", "worst_loss_heavy", "benefit", "min_abs_std", "min_abs_heavy", "d_min_abs"):
                    v = [r[q] for r in rr]
                    ranges.append({"arch": arch, "factor": f, "metric": met, "quantity": q, "s0": v[0], "s1": v[1], "s2": v[2],
                                   "min": min(v), "max": max(v), "n_positive": sum(x > 0 for x in v), "note": "seed range, not a CI"})
    publish_json(out, "contrast_results.json", {"coverage": coverage, "reproduction_max_abs_diff": repro, "runtime_s": time.perf_counter() - t0,
                                                 "factor_overlap": OVERLAP})
    publish_csv(out, "matched_worst_case.csv", wc_rows, list(wc_rows[0]))
    publish_csv(out, "matched_per_level.csv", lvl_rows, list(lvl_rows[0]))
    publish_csv(out, "matched_crossing.csv", cr_rows, list(cr_rows[0]))
    publish_csv(out, "matched_seed_ranges.csv", ranges, list(ranges[0]))
    for met, lab in (("sens", "worst-case sensitivity loss (pp)"), ("auc", "worst-case AUROC loss")):
        publish_bytes(out, f"plot_matched_worst_{met}.png", _plot_matched(wc_rows, met, lab))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "compare"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("contrast_results.json", "matched_worst_case.csv", "matched_per_level.csv",
                                                                           "matched_crossing.csv", "matched_seed_ranges.csv", "spec.json",
                                                                           "plot_matched_worst_sens.png", "plot_matched_worst_auc.png")}
    rp = P.write_receipt(receipt, out, name="receipt_compare.json")
    print(coverage, repro, "runtime", round(time.perf_counter() - t0, 1), "verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def _plot_matched(rows, met, lab):
    """One panel per factor; x = 9 matched cells; standard (circle) and heavy (square) worst loss with 95% intervals."""
    from PIL import Image, ImageDraw
    from .rq2_predict import FACTORS
    rr = [r for r in rows if r["metric"] == met]
    vals = [v for r in rr for v in json.loads(r["worst_loss_std_ci"]) + json.loads(r["worst_loss_heavy_ci"])] + [0.0]
    ymin, ymax = min(vals), max(vals)
    pad = (ymax - ymin) * 0.06 or 1
    ymin, ymax = ymin - pad, ymax + pad
    PW, PH, L, T = 300, 360, 50, 50
    img = Image.new("RGB", (L + PW * 5 + 20, T + PH + 60), "white")
    d = ImageDraw.Draw(img)
    d.text((L, 10), f"{lab}: standard (blue circle) vs heavy (orange square), 95% lesion-bootstrap intervals; cells = arch x seed", fill="black")
    Y = lambda v: T + (ymax - v) / (ymax - ymin) * PH
    for k in range(6):
        v = ymin + k * (ymax - ymin) / 5
        d.text((4, Y(v) - 6), f"{v:.3g}", fill="black")
    for j, f in enumerate(FACTORS):
        x0 = L + j * PW
        d.rectangle([x0, T, x0 + PW - 10, T + PH], outline="black")
        d.line([x0, Y(0), x0 + PW - 10, Y(0)], fill=(170, 170, 170))
        d.text((x0 + 4, T + 4), f, fill="black")
        fr = sorted((r for r in rr if r["factor"] == f), key=lambda r: (["resnet50", "efficientnet_b3", "densenet121"].index(r["arch"]), r["seed"]))
        for i, r in enumerate(fr):
            xc = x0 + 18 + i * (PW - 40) / 8
            for dx, key, col, shape in ((-5, "worst_loss_std", (31, 119, 180), "o"), (5, "worst_loss_heavy", (255, 127, 14), "s")):
                lo, hi = json.loads(r[key + "_ci"])
                d.line([xc + dx, Y(lo), xc + dx, Y(hi)], fill=col, width=2)
                yv = Y(r[key])
                (d.ellipse if shape == "o" else d.rectangle)([xc + dx - 3, yv - 3, xc + dx + 3, yv + 3], fill=col)
            d.text((xc - 8, T + PH + 4), {"resnet50": "R", "efficientnet_b3": "E", "densenet121": "D"}[r["arch"]] + str(r["seed"]), fill="black")
    d.text((L, T + PH + 30), "R = ResNet-50, E = EfficientNet-B3, D = DenseNet-121; digit = seed. Loss read with baselines (see tables).", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "sweep", "analyze", "compare"))
    ap.add_argument("--out-root")
    ap.add_argument("--run-dir")
    a = ap.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from . import config as C
    from .seeds import seed_everything
    cfg = C.load(a.config)
    if cfg["purpose"] != "predict_v1":
        raise C.ConfigError("requires purpose 'predict_v1'")
    seed_everything(0, cfg["determinism"])
    {"spec": phase_spec, "sweep": R.phase_sweep, "analyze": R.phase_analyze, "compare": phase_compare}[a.phase](cfg, a)


if __name__ == "__main__":
    main()
