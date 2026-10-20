"""Stage 8C: descriptive RQ2 statistics from verified Stage 8B predictions (CPU only).

Pure helpers (tested): fast AUROC via ranks, worst-case loss with frozen tie rule, 10-pp sensitivity
crossing with right-censoring, censor-aware empirical quantiles, ranking shares with split ties.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq2_stats --config ../configs/rq2_stats_8c_v1.json --phase spec --out-root ../runs/rq2_stats_8c
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq2_stats --config ../configs/rq2_stats_8c_v1.json --phase analyze --run-dir <dir>
"""
import argparse
import csv
import datetime
import io
import json
import math
import os

import numpy as np

FACTORS = ("blur", "illumination", "compression", "scale", "nuisance")
GRID = tuple(i / 8 for i in range(9))
B = 2000
DROP = 0.10
CENSORED = ">1.0"
EPS = 1e-12          # exact-boundary tolerance for float sensitivities (k/170 - 0.10)


# ------------------------------------------------------------------ pure helpers (tested)
def auc_ranks(p, y):
    """Mann-Whitney AUROC with average ranks (ties = 1/2). None if a class is absent."""
    from scipy.stats import rankdata
    p, y = np.asarray(p, float), np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def worst_case(values):
    """values: metric at GRID levels (index 0 = baseline); None allowed -> None result.
    loss(s) = m(0) - m(s); worst = max over the grid incl. s=0 (so >= 0). Tie rule: smallest severity."""
    if any(v is None for v in values):
        return None, None
    losses = [values[0] - v for v in values]
    w = max(losses)
    level = next(GRID[i] for i, l in enumerate(losses) if l == w)
    return w, level


def crossing(sens, grid=GRID, drop=DROP):
    """First severity with sens(s) <= sens(0) - drop, linear interpolation between the first bracketing
    adjacent levels (no monotonicity forced). Returns float, CENSORED (no crossing in grid) or None
    (undefined: any sensitivity missing)."""
    if any(v is None for v in sens):
        return None
    target = sens[0] - drop
    for i in range(1, len(grid)):
        if sens[i] <= target + EPS:
            a, b = sens[i - 1], sens[i]          # a > target >= b (first bracketing pair)
            if abs(b - target) <= EPS:
                return grid[i]
            return grid[i - 1] + (a - target) / (a - b) * (grid[i] - grid[i - 1])
    return CENSORED


def censored_quantiles(vals, qs=(0.025, 0.5, 0.975)):
    """vals: floats and CENSORED (None excluded beforehand). Ordered ranks with censored after all finite
    values; quantile at rank ceil(q*n) (1-based, inverse-CDF type 1). Landing in the censored group -> '>1.0'."""
    fin = sorted(v for v in vals if v != CENSORED)
    n = len(vals)
    out = {}
    for q in qs:
        if n == 0:
            out[q] = None
            continue
        k = max(1, math.ceil(q * n))
        out[q] = fin[k - 1] if k <= len(fin) else CENSORED
    return out


def ranking_shares(rep_losses):
    """rep_losses: list of {factor: worst loss or None}. Valid replicates need every factor defined.
    One unit of credit split equally among tied maxima."""
    shares = {f: 0.0 for f in FACTORS}
    valid = ties = 0
    for d in rep_losses:
        if any(d.get(f) is None for f in FACTORS):
            continue
        valid += 1
        m = max(d[f] for f in FACTORS)
        win = [f for f in FACTORS if d[f] == m]
        ties += len(win) > 1
        for f in win:
            shares[f] += 1.0 / len(win)
    return {"shares": {f: (v / valid if valid else None) for f, v in shares.items()}, "valid": valid,
            "excluded": len(rep_losses) - valid, "tie_replicates": ties}


def lesion_index(clusters):
    """Map image rows to lesion groups; returns (lesion ids sorted, list of row-index arrays)."""
    groups = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    ids = sorted(groups)
    return ids, [np.asarray(groups[c], dtype=np.int64) for c in ids]


def draw_rows(groups, rng):
    """One lesion-cluster bootstrap draw: all images of each sampled lesion, multiplicity kept."""
    pick = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[k] for k in pick])


# ------------------------------------------------------------------ phases
def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _run8b(cfg):
    from . import config as C
    return C.resolve(cfg, cfg["supporting_inputs"]["stage8b_verify_receipt"]).rsplit(os.sep, 1)[0]


def phase_spec(cfg, a):
    from . import provenance as P
    from .desc_4b3 import publish_bytes, publish_json
    run = _run8b(cfg)
    rec = json.load(open(os.path.join(run, "receipt_verify.json"), encoding="utf-8"))
    if rec["status"] != "completed":
        raise RuntimeError("Stage 8B verification not completed")
    files = {}
    for n, h in rec["outputs"].items():
        if P.sha256_file(os.path.join(run, n)) != h:
            raise RuntimeError(f"changed {n}")
    for d in sorted(os.listdir(run)):
        if d.startswith("u_") and os.path.isdir(os.path.join(run, d)) and ".tmp" not in d:
            files[f"{d}/predictions.csv"] = P.sha256_file(os.path.join(run, d, "predictions.csv"))
    spec = {"spec_version": "rq2_stats_8c_v1", "created_utc": _now(), "stage8b_run": os.path.basename(run),
            "inputs": {"receipt_outputs": rec["outputs"], "prediction_files": files},
            "cohort": "HAM TEST 1548 images / 170 melanoma / 1155 lesions; identical across units",
            "baseline": "severity-0 copies verified identical; one baseline (not independent data)",
            "metrics": "image-level sensitivity, specificity (frozen threshold p>=t), AUROC (average-rank Mann-Whitney)",
            "changes": "sensitivity change in pp (level - baseline); AUROC change on 0-1 scale",
            "bootstrap": {"type": "lesion-cluster percentile bootstrap", "B": B, "seed": cfg["seeds"]["bootstrap_seed"],
                          "draws": "the SAME lesion draws for baseline, all severities and all factors",
                          "rng": "numpy Generator(PCG64(seed)); integers(0,K,K) over sorted lesion IDs",
                          "quantiles": "numpy linear 0.025/0.975 for continuous quantities",
                          "undefined": "per metric (no melanoma / no non-melanoma in a replicate); excluded; never "
                                       "redrawn; >5% -> unreliable"},
            "worst_case": "loss(s) = m(0) - m(s) over the 9-level grid incl. s=0 (>=0); tie -> smallest severity; "
                          "recomputed within every replicate; 'worst loss over the sampled grid'",
            "crossing": {"definition": "first s with sens(s) <= sens(0) - 0.10 (+1e-12 float tolerance; exact grid hits count); linear interpolation between the first "
                                       "bracketing adjacent levels; exact grid hits count; no monotonicity forced",
                         "per_replicate": "relative to that replicate's own baseline",
                         "categories": "finite crossing / no crossing (right-censored, '>1.0') / undefined",
                         "quantiles": "censor-aware empirical type-1: sorted finite values then censored; rank ceil(q*n); "
                                      "landing in censored group -> '>1.0'"},
            "ranking": "per metric, share of valid replicates where a factor has the largest worst-case loss; ties split "
                       "equally; replicates with any undefined factor excluded and counted",
            "no_tests": "no ANOVA, no hypothesis tests, no multiplicity family (02_rev2 §4)"}
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "analysis_spec.json", spec)
    publish_bytes(out, "analysis_spec.json.sha256", f"{P.sha256_file(os.path.join(out, 'analysis_spec.json'))}  analysis_spec.json\n".encode())
    receipt["outputs"] = {"analysis_spec.json": P.sha256_file(os.path.join(out, "analysis_spec.json"))}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out)


def load_units(run, spec_files, model_sha, th_sha, t):
    from . import provenance as P
    units, ref = {}, None
    for f in FACTORS:
        for s in GRID:
            name = f"u_{f}__{s:.3f}/predictions.csv"
            path = os.path.join(run, *name.split("/"))
            if P.sha256_file(path) != spec_files[name]:
                raise RuntimeError(f"changed {name}")
            rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
            ident = [(r["record_uid"], r["label_mel"], r["cluster_id"]) for r in rows]
            if ref is None:
                ref = ident
            if ident != ref or len({r["key"] for r in rows}) != len(rows):
                raise RuntimeError(f"identity/order mismatch in {name}")
            p = np.array([float(r["prob"]) for r in rows])
            if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
                raise RuntimeError(f"invalid probabilities {name}")
            if any(r["checkpoint_sha256"] != model_sha or r["threshold_sha256"] != th_sha for r in rows):
                raise RuntimeError(f"model/threshold identity {name}")
            if any(int(r["pred"]) != int(float(r["prob"]) >= t) for r in rows):
                raise RuntimeError(f"pred mismatch {name}")
            units[(f, s)] = p
    base = [units[(f, 0.0)] for f in FACTORS]
    if not all(np.array_equal(base[0], b) for b in base[1:]):
        raise RuntimeError("severity-0 copies differ")
    y = np.array([r[1] == "1" for r in ref])
    clusters = [r[2] for r in ref]
    if (len(ref), int(y.sum()), len(set(clusters))) != (1548, 170, 1155):
        raise RuntimeError("cohort counts")
    return units, y, clusters


def _metrics(p, y, t):
    pos, neg = y.sum(), (~y).sum()
    sens = float((p[y] >= t).mean()) if pos else None
    spec = float((p[~y] < t).mean()) if neg else None
    return sens, spec, auc_ranks(p, y)


def phase_analyze(cfg, a):
    from . import config as C, provenance as P
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    out = os.path.abspath(a.run_dir)
    sp = os.path.join(out, "analysis_spec.json")
    if open(sp + ".sha256", encoding="utf-8").read().split()[0] != P.sha256_file(sp):
        raise RuntimeError("spec hash")
    spec = json.load(open(sp, encoding="utf-8"))
    run = _run8b(cfg)
    s8 = json.load(open(os.path.join(run, "spec.json"), encoding="utf-8"))
    t = s8["threshold"]["value"]
    units, y, clusters = load_units(run, spec["inputs"]["prediction_files"], s8["model_sha256"],
                                    s8["inputs"]["threshold"]["sha256"], t)
    # point estimates
    pt = {}
    for f in FACTORS:
        for s in GRID:
            p = units[(f, s)]
            sens, spe, auc = _metrics(p, y, t)
            pt[(f, s)] = {"TP": int((p[y] >= t).sum()), "FN": int((p[y] < t).sum()), "TN": int((p[~y] < t).sum()),
                          "FP": int((p[~y] >= t).sum()), "sens": sens, "spec": spe, "auc": auc}
    if (pt[("blur", 0.0)]["TP"], pt[("blur", 0.0)]["FN"], pt[("blur", 0.0)]["TN"], pt[("blur", 0.0)]["FP"]) != (157, 13, 911, 467):
        raise RuntimeError("baseline counts")
    # bootstrap (shared draws)
    ids, groups = lesion_index(clusters)
    rng = np.random.Generator(np.random.PCG64(spec["bootstrap"]["seed"]))
    rep = {k: {"sens": np.full(B, np.nan), "auc": np.full(B, np.nan)} for k in pt}
    for b in range(B):
        idx = draw_rows(groups, rng)
        yb = y[idx]
        for k, p in units.items():
            if k[1] == 0.0 and k[0] != "blur":
                continue                         # baseline computed once (copies identical)
            sens, _, auc = _metrics(p[idx], yb, t)
            rep[k]["sens"][b] = np.nan if sens is None else sens
            rep[k]["auc"][b] = np.nan if auc is None else auc
    for f in FACTORS[1:]:
        rep[(f, 0.0)] = rep[("blur", 0.0)]
    q = lambda a_: np.quantile(a_[np.isfinite(a_)], [0.025, 0.975], method="linear").tolist() if np.isfinite(a_).any() else None
    level_rows, res = [], {"spec_sha256": P.sha256_file(sp)}
    for f in FACTORS:
        for s in GRID:
            k = (f, s)
            ds = (rep[k]["sens"] - rep[(f, 0.0)]["sens"]) * 100
            da = rep[k]["auc"] - rep[(f, 0.0)]["auc"]
            und_s, und_a = int(np.isnan(ds).sum()), int(np.isnan(da).sum())
            level_rows.append({"factor": f, "severity": s, **{kk: pt[k][kk] for kk in ("TP", "FN", "TN", "FP")},
                               "sensitivity": pt[k]["sens"], "specificity": pt[k]["spec"], "auroc": pt[k]["auc"],
                               "d_sens_pp": (pt[k]["sens"] - pt[(f, 0.0)]["sens"]) * 100,
                               "d_sens_ci_lo": q(ds)[0], "d_sens_ci_hi": q(ds)[1],
                               "d_auroc": pt[k]["auc"] - pt[(f, 0.0)]["auc"], "d_auroc_ci_lo": q(da)[0], "d_auroc_ci_hi": q(da)[1],
                               "undefined_sens": und_s, "undefined_auc": und_a,
                               "unreliable": max(und_s, und_a) / B > 0.05})
    # worst case + crossing + ranking
    wc_rows, cr_rows, rank_in = [], [], {"sens": [], "auc": []}
    wc_reps = {m: {f: np.full(B, np.nan) for f in FACTORS} for m in ("sens", "auc")}
    for f in FACTORS:
        for m in ("sens", "auc"):
            obs, lvl = worst_case([pt[(f, s)][m] for s in GRID])
            levels = []
            for b in range(B):
                vals = [None if np.isnan(rep[(f, s)][m][b]) else float(rep[(f, s)][m][b]) for s in GRID]
                w, l = worst_case(vals)
                if w is not None:
                    wc_reps[m][f][b] = w
                    levels.append(l)
            arr = wc_reps[m][f]
            scale_ = 100 if m == "sens" else 1
            lv = {str(s): levels.count(s) for s in GRID if levels.count(s)}
            wc_rows.append({"factor": f, "metric": "sensitivity_pp" if m == "sens" else "auroc",
                            "worst_loss": obs * scale_, "worst_level": lvl,
                            "ci_lo": q(arr)[0] * scale_, "ci_hi": q(arr)[1] * scale_,
                            "undefined_reps": int(np.isnan(arr).sum()), "worst_level_distribution": json.dumps(lv)})
        obs_c = crossing([pt[(f, s)]["sens"] for s in GRID])
        cvals, und = [], 0
        for b in range(B):
            v = crossing([None if np.isnan(rep[(f, s)]["sens"][b]) else float(rep[(f, s)]["sens"][b]) for s in GRID])
            if v is None:
                und += 1
            else:
                cvals.append(v)
        cq = censored_quantiles(cvals)
        cr_rows.append({"factor": f, "observed_crossing": obs_c, "valid_reps": len(cvals), "undefined_reps": und,
                        "no_crossing_reps": sum(v == CENSORED for v in cvals),
                        "no_crossing_share": sum(v == CENSORED for v in cvals) / len(cvals) if cvals else None,
                        "q025": cq[0.025], "q50": cq[0.5], "q975": cq[0.975]})
    for m in ("sens", "auc"):
        rank_in[m] = [{f: (None if np.isnan(wc_reps[m][f][b]) else float(wc_reps[m][f][b])) for f in FACTORS} for b in range(B)]
    ranking = {m: ranking_shares(rank_in[m]) for m in ("sens", "auc")}
    res.update(levels=level_rows, worst_case=wc_rows, crossing=cr_rows, ranking=ranking,
               baseline={k: pt[("blur", 0.0)][k] for k in ("TP", "FN", "TN", "FP", "sens", "spec", "auc")})
    publish_json(out, "results.json", res)
    publish_csv(out, "per_level_table.csv", level_rows, list(level_rows[0]))
    publish_csv(out, "worst_case_table.csv", wc_rows, list(wc_rows[0]))
    publish_csv(out, "crossing_table.csv", cr_rows, list(cr_rows[0]))
    rk = [{"metric": m, "factor": f, "share": ranking[m]["shares"][f], "valid": ranking[m]["valid"],
           "excluded": ranking[m]["excluded"], "tie_replicates": ranking[m]["tie_replicates"]} for m in ranking for f in FACTORS]
    publish_csv(out, "ranking_table.csv", rk, list(rk[0]))
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"{k[0]}__{k[1]:.3f}__{m}": rep[k][m] for k in rep for m in ("sens", "auc")},
                        **{f"worst__{m}__{f}": wc_reps[m][f] for m in wc_reps for f in FACTORS})
    publish_bytes(out, "bootstrap_replicates.npz", buf.getvalue())
    for m, lab in (("d_sens_pp", "Sensitivity change from baseline (pp)"), ("d_auroc", "AUROC change from baseline")):
        publish_bytes(out, f"plot_{m}.png", _plot(level_rows, m, lab))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out)) if not n.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def _plot(rows, key, label):
    from PIL import Image, ImageDraw
    W, H, L, R, T, Bm = 900, 520, 70, 170, 40, 60
    lo_k, hi_k = ("d_sens_ci_lo", "d_sens_ci_hi") if key == "d_sens_pp" else ("d_auroc_ci_lo", "d_auroc_ci_hi")
    ymin = min(min(r[lo_k] for r in rows), 0)
    ymax = max(max(r[hi_k] for r in rows), 0)
    pad = (ymax - ymin) * 0.05 or 1
    ymin, ymax = ymin - pad, ymax + pad
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    X = lambda s: L + s * (W - L - R)
    Y = lambda v: T + (ymax - v) / (ymax - ymin) * (H - T - Bm)
    d.rectangle([L, T, W - R, H - Bm], outline="black")
    d.line([L, Y(0), W - R, Y(0)], fill=(150, 150, 150))
    for s in GRID:
        d.text((X(s) - 10, H - Bm + 5), f"{s:g}", fill="black")
    for k in range(6):
        v = ymin + k * (ymax - ymin) / 5
        d.text((5, Y(v) - 6), f"{v:.3g}", fill="black")
    d.text((L, 10), f"{label} - points with pointwise 95% lesion-bootstrap intervals (not simultaneous bands)", fill="black")
    d.text((L + 250, H - 25), "severity", fill="black")
    cols = {"blur": (31, 119, 180), "illumination": (255, 127, 14), "compression": (44, 160, 44), "scale": (214, 39, 40),
            "nuisance": (148, 103, 189)}
    for j, f in enumerate(FACTORS):
        rr = [r for r in rows if r["factor"] == f]
        off = (j - 2) * 0.006
        pts = [(X(r["severity"] + off), Y(r[key])) for r in rr]
        d.line(pts, fill=cols[f], width=2)
        for r, (x, yv) in zip(rr, pts):
            d.line([x, Y(r[lo_k]), x, Y(r[hi_k])], fill=cols[f])
            d.ellipse([x - 3, yv - 3, x + 3, yv + 3], fill=cols[f])
        d.rectangle([W - R + 15, T + 20 * j, W - R + 30, T + 20 * j + 12], fill=cols[f])
        d.text((W - R + 35, T + 20 * j), f, fill="black")
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
    if cfg["purpose"] != "analysis_v1":
        raise C.ConfigError("requires purpose 'analysis_v1'")
    (phase_spec if a.phase == "spec" else phase_analyze)(cfg, a)


if __name__ == "__main__":
    main()
