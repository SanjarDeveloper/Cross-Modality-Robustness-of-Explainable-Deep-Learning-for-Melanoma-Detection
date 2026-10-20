"""Publication figures from the result files.

  python code/figures.py --tag resnet50_standard_s0
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_paths import output_root, paired_path
ROOT = output_root()
RES, FIG = f"{ROOT}/results", f"{ROOT}/results/figures"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def save_figure(fig, name):
    if os.environ.get("MELANOMA_RUN_MODE") == "smoke":
        fig.suptitle("SMOKE - software rehearsal, not scientific evidence",
                     fontsize=10, color="crimson")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def fig_sweep(tag):
    """RQ1/RQ2: severity-sensitivity curves with the 10% threshold marked."""
    sweep = json.load(open(f"{RES}/sweep_{tag}.json"))
    by = {}
    for r in sweep:
        by.setdefault(r["factor"], []).append(r)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for f, rows in by.items():
        rows = sorted(rows, key=lambda r: r["severity"])
        s = [r["severity"] for r in rows]
        y = [r["sensitivity"] for r in rows]
        lo = [r.get("sens_ci_lo", np.nan) for r in rows]
        hi = [r.get("sens_ci_hi", np.nan) for r in rows]
        ln, = a1.plot(s, y, marker="o", ms=3, label=f)
        a1.fill_between(s, lo, hi, alpha=0.13, color=ln.get_color())
        a2.plot(s, [r["auroc"] for r in rows], marker="o", ms=3, label=f)
    base = by[list(by)[0]][0]["sensitivity"]
    report = json.load(open(f"{RES}/report_{tag}.json"))
    drop = report["drop"]
    a1.axhline(base - drop, ls="--", c="crimson", lw=1)
    a1.text(0.02, base - drop + .012, f"-{100*drop:g} percentage points",
            color="crimson", fontsize=7)
    a1.set_xlabel("severity"); a1.set_ylabel("melanoma sensitivity")
    a1.set_title("Sensitivity vs degradation severity (95% CI)")
    a2.set_xlabel("severity"); a2.set_ylabel("AUROC")
    a2.set_title("AUROC vs degradation severity")
    a1.legend(fontsize=7, frameon=False)
    save_figure(fig, "fig1_sweep.png")


def fig_real(tag):
    """RQ1 headline: paired same-lesion modality shift."""
    p = json.load(open(paired_path(RES, tag)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4))
    x = [0, 1]
    a1.bar(x, [p["sens_dermoscopy"], p["sens_clinical"]],
           color=["#3b7dd8", "#d8703b"], width=.55)
    lo, hi = p["sens_drop_ci95"]
    a1.set_xticks(x); a1.set_xticklabels(["dermoscopy", "clinical\n(phone-like)"])
    a1.set_ylabel("melanoma sensitivity"); a1.set_ylim(0, 1.05)
    a1.set_title(f"Same {p['n_melanoma']} melanomas, two cameras")
    a1.text(.5, .25, f"drop {p['sens_drop']:.3f}\n95% CI [{lo:.3f},{hi:.3f}]"
                     f"\nMcNemar (all cases) p={p['mcnemar_p']:.1e}",
            ha="center", fontsize=8)
    a2.bar(x, [p["auroc_dermoscopy"], p["auroc_clinical"]],
           color=["#3b7dd8", "#d8703b"], width=.55)
    a2.set_xticks(x); a2.set_xticklabels(["dermoscopy", "clinical"])
    a2.set_ylabel("AUROC"); a2.set_ylim(0, 1.0)
    a2.set_title(f"DeLong p = {p['delong_p']:.1e}")
    save_figure(fig, "fig2_real_shift.png")


def fig_faithfulness(tag):
    """RQ4 core: faithfulness falls further than accuracy, and worst for melanoma."""
    x = json.load(open(f"{RES}/xai_{tag}.json"))
    d = next(r for r in x if r["dataset"] == "derm7pt" and r["modality"] == "dermoscopy")
    c = next(r for r in x if r["dataset"] == "derm7pt" and r["modality"] == "clinical")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))

    groups = ["all", "melanoma", "non-melanoma"]
    derm = [d["insertion_auc"], d["insertion_auc_mel"], d["insertion_auc_ben"]]
    clin = [c["insertion_auc"], c["insertion_auc_mel"], c["insertion_auc_ben"]]
    w, i = 0.36, np.arange(3)
    a1.bar(i - w / 2, derm, w, label="dermoscopy", color="#3b7dd8")
    a1.bar(i + w / 2, clin, w, label="clinical", color="#d8703b")
    for k in range(3):
        pct = 100 * (clin[k] - derm[k]) / derm[k]
        a1.text(i[k], max(derm[k], clin[k]) + .012, f"{pct:+.0f}%",
                ha="center", fontsize=8,
                color="crimson" if pct < -30 else "k")
    a1.set_xticks(i); a1.set_xticklabels(groups)
    a1.set_ylabel("insertion AUC  (higher = more faithful)")
    a1.set_title("Explanation faithfulness under real modality shift")
    a1.legend(fontsize=8, frameon=False)

    # accuracy vs faithfulness retention, side by side
    ret = {
        "sensitivity": c["sensitivity"] / d["sensitivity"],
        "AUROC": c["auroc"] / d["auroc"],
        "faithfulness\n(all)": c["insertion_auc"] / d["insertion_auc"],
        "faithfulness\n(melanoma)": c["insertion_auc_mel"] / d["insertion_auc_mel"],
    }
    cols = ["#3b7dd8", "#3b7dd8", "#d8703b", "#c0392b"]
    a2.bar(range(len(ret)), list(ret.values()), color=cols, width=.6)
    a2.axhline(1.0, ls="--", c="k", lw=.8)
    a2.set_xticks(range(len(ret)))
    a2.set_xticklabels(list(ret), fontsize=7.5)
    a2.set_ylabel("retention  (shifted / baseline)")
    a2.set_ylim(0, max(1.15, max(ret.values()) * 1.15))
    a2.set_title("Accuracy and explanation retention")
    for k, v in enumerate(ret.values()):
        a2.text(k, v + .02, f"{v:.2f}", ha="center", fontsize=8)
    save_figure(fig, "fig3_faithfulness.png")


def fig_cmrs(tag):
    """CMRS radar for the real shift."""
    rep = json.load(open(f"{RES}/report_{tag}.json"))
    s = rep.get("cmrs_real")
    if not s:
        return
    ax_names = list(s["axes"])
    vals = [s["axes"][k] for k in ax_names]
    n = len(vals)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals_c, ang_c = vals + vals[:1], ang + ang[:1]
    fig = plt.figure(figsize=(4.6, 4.6))
    a = plt.subplot(111, polar=True)
    a.plot(ang_c, vals_c, lw=2, color="#c0392b")
    a.fill(ang_c, vals_c, alpha=0.22, color="#c0392b")
    a.set_xticks(ang); a.set_xticklabels(ax_names, fontsize=8)
    a.set_ylim(0, 1)
    a.set_title(f"CMRS {s['cmrs']:.1f}/100 ({s['zone']})\n"
                f"{len(ax_names)}/4 axes measurable", fontsize=10)
    save_figure(fig, "fig4_cmrs.png")


def main():
    global FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="resnet50_standard_s0")
    a = ap.parse_args()
    FIG = os.path.join(RES, "figures", a.tag)
    os.makedirs(FIG, exist_ok=True)
    for fn in (fig_sweep, fig_real, fig_faithfulness, fig_cmrs):
        fn(a.tag)
        print("ok:", fn.__name__)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
