"""Turn result rows into the study's answers: thresholds, factor ranking,
faithfulness-under-shift, and the CMRS.

Usage:
  python code/analyse.py --tag resnet50_standard_s0
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_paths import output_root, paired_path
ROOT = output_root()
RES = f"{ROOT}/results"


def load(tag, suite):
    p = f"{RES}/{suite}_{tag}.json"
    return json.load(open(p))


# ------------------------------------------------------------------ RQ1/RQ2
def thresholds(sweep, drop=0.10):
    """RQ1: severity at which sensitivity falls >drop absolute proportion.

    Linear interpolation between bracketing severities. Reported as NaN when
    the factor never crosses the threshold in the sampled range -- which is
    itself a finding ("this factor does not break the model"), not a failure.
    """
    out = {}
    by = {}
    for r in sweep:
        by.setdefault(r["factor"], []).append(r)
    for f, rows in by.items():
        rows = sorted(rows, key=lambda r: r["severity"])
        base = rows[0]["sensitivity"]
        target = base - drop
        cross = None
        for a, b in zip(rows, rows[1:]):
            if a["sensitivity"] >= target > b["sensitivity"]:
                span = a["sensitivity"] - b["sensitivity"]
                frac = (a["sensitivity"] - target) / span if span else 0
                cross = a["severity"] + frac * (b["severity"] - a["severity"])
                break
        sens = [r["sensitivity"] for r in rows]
        aur = [r["auroc"] for r in rows]
        out[f] = {
            "baseline_sens": base,
            "threshold_severity": cross,
            "sens_at_max_severity": sens[-1],
            # Endpoint drop alone is misleading: a factor whose sensitivity
            # dips and then returns to baseline scores 0 while genuinely
            # degrading the model. Report worst-case and AUROC loss too, and
            # rank on worst-case.
            "endpoint_drop": base - sens[-1],
            "worst_drop": base - min(sens),
            "severity_at_worst": rows[int(np.argmin(sens))]["severity"],
            "auroc_drop": aur[0] - min(aur),
            "curve": [(r["severity"], r["sensitivity"], r["auroc"]) for r in rows],
        }
    return out


def factor_ranking(thr):
    """RQ2: which acquisition factor costs the most melanoma sensitivity.

    Ranked by WORST-CASE drop across the sampled severities, not by the
    endpoint, so a non-monotone factor is not scored as harmless.
    """
    rank = sorted(thr.items(), key=lambda kv: -kv[1]["worst_drop"])
    return [{"factor": f, "worst_drop": v["worst_drop"],
             "endpoint_drop": v["endpoint_drop"],
             "auroc_drop": v["auroc_drop"],
             "threshold_severity": v["threshold_severity"]} for f, v in rank]


# --------------------------------------------------------------------- RQ4
def faithfulness_shift(xai):
    """RQ4: does explanation quality degrade under modality shift, and does
    it degrade differently for melanoma than for non-melanoma?"""
    d7 = [r for r in xai if r["dataset"] == "derm7pt"]
    out = {}
    if len(d7) >= 2:
        derm = next(r for r in d7 if r["modality"] == "dermoscopy")
        clin = next(r for r in d7 if r["modality"] == "clinical")
        out["real_shift"] = {
            "deletion": {"dermoscopy": derm["deletion_auc"],
                         "clinical": clin["deletion_auc"],
                         "change": clin["deletion_auc"] - derm["deletion_auc"]},
            "insertion": {"dermoscopy": derm["insertion_auc"],
                          "clinical": clin["insertion_auc"],
                          "change": clin["insertion_auc"] - derm["insertion_auc"]},
        }
        for cls in ("mel", "ben"):
            k = f"insertion_auc_{cls}"
            if k in derm and k in clin:
                out["real_shift"][f"insertion_{cls}"] = {
                    "dermoscopy": derm[k], "clinical": clin[k],
                    "change": clin[k] - derm[k]}
    sim = [r for r in xai if r["dataset"] == "ham10000"]
    if sim:
        b = next((r for r in sim if r["factor"] == "none"), None)
        if b:
            out["simulated"] = {
                r["factor"] + f"@{r['severity']}": {
                    "insertion_change": r["insertion_auc"] - b["insertion_auc"],
                    "deletion_change": r["deletion_auc"] - b["deletion_auc"],
                    "iou_change": r.get("iou", float("nan")) - b.get("iou", float("nan")),
                    "pointing_change": (r.get("pointing", float("nan"))
                                        - b.get("pointing", float("nan"))),
                }
                for r in sim if r["factor"] != "none"
            }
    return out


# -------------------------------------------------------------------- CMRS
def cmrs(baseline, shifted):
    """Cross-Modality Reliability Score, 0-100.

    Four retention ratios, EQUALLY WEIGHTED. Equal weights are used as the
    headline rather than weights fitted on this study's own data: fitting
    weights on the same runs that are then scored is circular, and with 252
    melanomas it would overfit. Data-derived weights belong in a sensitivity
    analysis, not the primary metric.

    A missing axis (e.g. localisation on a dataset with no masks) is dropped
    and the score renormalised over the axes actually measured -- reported
    alongside the score so it cannot be mistaken for a full four-axis value.
    """
    ax, used = {}, []

    def ratio(a, b, higher_better=True):
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            return None
        if higher_better:
            return float(np.clip(b / a, 0, 1)) if a > 0 else None
        return float(np.clip(a / b, 0, 1)) if b > 0 else None

    # 1. discriminative retention (sensitivity ratio)
    v = ratio(baseline.get("sensitivity"), shifted.get("sensitivity"))
    if v is not None:
        ax["discriminative"] = v; used.append("discriminative")
    # 2. calibration stability (ECE is lower-better)
    be, se = baseline.get("ece"), shifted.get("ece")
    if be is not None and se is not None:
        ax["calibration"] = float(np.clip(1 - abs(se - be), 0, 1))
        used.append("calibration")
    # 3. explanation fidelity (insertion AUC is higher-better)
    v = ratio(baseline.get("insertion_auc"), shifted.get("insertion_auc"))
    if v is not None:
        ax["fidelity"] = v; used.append("fidelity")
    # 4. localisation stability (IoU)
    v = ratio(baseline.get("iou"), shifted.get("iou"))
    if v is not None:
        ax["localisation"] = v; used.append("localisation")

    score = 100.0 * float(np.mean(list(ax.values()))) if ax else float("nan")
    return {"cmrs": score, "axes": ax, "axes_measured": used,
            "complete": len(used) == 4,
            "zone": ("green" if score > 80 else "amber" if score >= 60 else "red")
                    if np.isfinite(score) else "n/a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="resnet50_standard_s0")
    ap.add_argument("--drop", type=float, default=0.10)
    a = ap.parse_args()

    real, sweep, xai = (load(a.tag, s) for s in ("real", "sweep", "xai"))
    report = {"tag": a.tag, "drop_definition": "absolute sensitivity difference",
              "drop": a.drop, "negative_class": "non-melanoma (legacy ben keys)"}

    print("=" * 74)
    print(f"ANALYSIS  {a.tag}")
    print("=" * 74)

    if real:
        paired = json.load(open(paired_path(RES, a.tag)))
        if paired["tag"] != a.tag:
            raise ValueError("Paired result tag does not match analysis tag")
        report["rq1_real"] = paired
        print("\n[RQ1] REAL modality shift, Derm7pt paired same-lesion")
        print(f"  n pairs={paired['n_pairs']}  melanomas={paired['n_melanoma']}")
        print(f"  melanoma sensitivity  dermoscopy {paired['sens_dermoscopy']:.3f}"
              f"  ->  clinical {paired['sens_clinical']:.3f}")
        ci = paired["sens_drop_ci95"]
        print(f"  drop = {paired['sens_drop']:+.3f}  95% CI [{ci[0]:+.3f},{ci[1]:+.3f}]")
        print(f"  AUROC {paired['auroc_dermoscopy']:.3f} -> "
              f"{paired['auroc_clinical']:.3f}   DeLong p={paired['delong_p']:.2e}")
        print(f"  McNemar b={paired['mcnemar_b']} c={paired['mcnemar_c']} "
              f"p={paired['mcnemar_p']:.2e}")
        sig = paired["delong_p"] < 0.05
        print(f"  => modality shift {'IS' if sig else 'is NOT'} statistically significant")

    if sweep:
        thr = thresholds(sweep, a.drop)
        rank = factor_ranking(thr)
        report["rq1_thresholds"] = thr
        report["rq2_ranking"] = rank
        print(f"\n[RQ1] Degradation thresholds (>{a.drop*100:.0f} percentage-point loss)")
        for f, v in thr.items():
            t = v["threshold_severity"]
            print(f"  {f:13s} threshold={'%.2f' % t if t else '  none'}   "
                  f"worst drop={v['worst_drop']:+.3f} (at s={v['severity_at_worst']:.2f})"
                  f"   AUROC loss={v['auroc_drop']:+.3f}")
        print("\n[RQ2] Factor ranking by melanoma sensitivity lost")
        for i, r in enumerate(rank, 1):
            print(f"  {i}. {r['factor']:13s} worst={r['worst_drop']:+.3f}  "
                  f"endpoint={r['endpoint_drop']:+.3f}  AUROC={r['auroc_drop']:+.3f}")

    if xai:
        fs = faithfulness_shift(xai)
        report["rq4_faithfulness"] = fs
        print("\n[RQ4] Explanation faithfulness under shift")
        if "real_shift" in fs:
            rs = fs["real_shift"]
            print(f"  insertion AUC (higher=more faithful): "
                  f"{rs['insertion']['dermoscopy']:.3f} -> "
                  f"{rs['insertion']['clinical']:.3f} "
                  f"({rs['insertion']['change']:+.3f})")
            print(f"  deletion  AUC (lower=more faithful): "
                  f"{rs['deletion']['dermoscopy']:.3f} -> "
                  f"{rs['deletion']['clinical']:.3f} "
                  f"({rs['deletion']['change']:+.3f})")
            for cls in ("mel", "ben"):
                k = f"insertion_{cls}"
                if k in rs:
                    label = "non-melanoma" if cls == "ben" else "melanoma"
                    print(f"    {label}: {rs[k]['dermoscopy']:.3f} -> "
                          f"{rs[k]['clinical']:.3f} ({rs[k]['change']:+.3f})")

        # CMRS from the real paired shift
        d = next((r for r in xai if r["dataset"] == "derm7pt"
                  and r["modality"] == "dermoscopy"), None)
        c = next((r for r in xai if r["dataset"] == "derm7pt"
                  and r["modality"] == "clinical"), None)
        if d and c:
            s = cmrs(d, c)
            report["cmrs_real"] = s
            print(f"\n[CMRS] real dermoscopy -> clinical: {s['cmrs']:.1f}/100 "
                  f"({s['zone']})")
            for k, v in s["axes"].items():
                print(f"    {k:15s} {v:.3f}")
            if not s["complete"]:
                print(f"    NOTE: only {len(s['axes_measured'])}/4 axes measurable "
                      f"here (no lesion masks; XAI rows do not include ECE).")

    out = f"{RES}/report_{a.tag}.json"
    json.dump(report, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
