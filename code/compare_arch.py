"""RQ3 -- do architectures and training strategies differ in cross-modality
robustness?

The comparison that matters is NOT which model scores highest on dermoscopy;
it is which model RETAINS the most when the modality changes. A model can have
the best baseline and the worst retention. Both are reported so the
distinction is visible.

  python code/compare_arch.py
"""
import glob, json, os, sys
import numpy as np

from run_paths import output_root
ROOT = output_root()
RES = f"{ROOT}/results"


def collect():
    rows = []
    for f in sorted(glob.glob(f"{RES}/real_*.json")):
        tag = os.path.basename(f)[len("real_"):-len(".json")]
        real = json.load(open(f))
        d7d = next((r for r in real if r["dataset"] == "derm7pt"
                    and r["modality"] == "dermoscopy"), None)
        d7c = next((r for r in real if r["dataset"] == "derm7pt"
                    and r["modality"] == "clinical"), None)
        ham = next((r for r in real if r["dataset"] == "ham10000"), None)
        if not (d7d and d7c):
            continue
        e = {"tag": tag, "arch": d7d["arch"], "strategy": d7d["strategy"],
             "baseline_auroc_ham": ham["auroc"] if ham else None,
             "sens_dermoscopy": d7d["sensitivity"],
             "sens_clinical": d7c["sensitivity"],
             "sens_drop": d7d["sensitivity"] - d7c["sensitivity"],
             "sens_retention": (d7c["sensitivity"] / d7d["sensitivity"]
                                if d7d["sensitivity"] else None),
             "auroc_dermoscopy": d7d["auroc"], "auroc_clinical": d7c["auroc"],
             "auroc_retention": (d7c["auroc"] / d7d["auroc"]
                                 if d7d["auroc"] else None)}
        # faithfulness retention, if the xai suite ran for this tag
        xf = f"{RES}/xai_{tag}.json"
        if os.path.exists(xf):
            x = json.load(open(xf))
            xd = next((r for r in x if r["dataset"] == "derm7pt"
                       and r["modality"] == "dermoscopy"), None)
            xc = next((r for r in x if r["dataset"] == "derm7pt"
                       and r["modality"] == "clinical"), None)
            if xd and xc and xd["insertion_auc"]:
                e["faith_retention"] = xc["insertion_auc"] / xd["insertion_auc"]
                if xd.get("insertion_auc_mel"):
                    e["faith_retention_mel"] = (xc["insertion_auc_mel"]
                                                / xd["insertion_auc_mel"])
        # worst simulated factor, if the sweep ran
        sf = f"{RES}/sweep_{tag}.json"
        if os.path.exists(sf):
            sw = json.load(open(sf))
            by = {}
            for r in sw:
                by.setdefault(r["factor"], []).append(r)
            worst, wname = 0, None
            for fac, rs in by.items():
                rs = sorted(rs, key=lambda r: r["severity"])
                dr = rs[0]["sensitivity"] - min(r["sensitivity"] for r in rs)
                if dr > worst:
                    worst, wname = dr, fac
            e["worst_factor"] = wname
            e["worst_factor_drop"] = worst
        rows.append(e)
    return rows


def main():
    rows = collect()
    if not rows:
        raise FileNotFoundError("No real_*.json results. Run the campaign first.")
    rows.sort(key=lambda r: -(r.get("sens_retention") or 0))

    print("=" * 100)
    print("RQ3 -- cross-modality robustness by architecture x strategy")
    print("=" * 100)
    print(f"{'model':32s} {'HAM':>6s} {'sensD':>6s} {'sensC':>6s} "
          f"{'retain':>7s} {'faith':>7s} {'faithM':>7s}  worst factor")
    print("-" * 100)
    for r in rows:
        f_ = r.get("faith_retention")
        fm = r.get("faith_retention_mel")
        print(f"{r['tag']:32s} "
              f"{(r['baseline_auroc_ham'] or 0):6.3f} "
              f"{r['sens_dermoscopy']:6.3f} {r['sens_clinical']:6.3f} "
              f"{(r['sens_retention'] or 0):7.3f} "
              f"{f_ if f_ is None else format(f_, '7.3f')} "
              f"{fm if fm is None else format(fm, '7.3f')}  "
              f"{r.get('worst_factor','-')} "
              f"({r.get('worst_factor_drop', float('nan')):.2f})")

    print()
    best_base = max(rows, key=lambda r: r["baseline_auroc_ham"] or 0)
    best_ret = max(rows, key=lambda r: r["sens_retention"] or 0)
    print(f"Best dermoscopic baseline : {best_base['tag']} "
          f"(AUROC {best_base['baseline_auroc_ham']:.3f})")
    print(f"Best cross-modality retain: {best_ret['tag']} "
          f"(retains {best_ret['sens_retention']:.3f} of sensitivity)")
    if best_base["tag"] != best_ret["tag"]:
        print("  => The most accurate model in-distribution is NOT the most "
              "robust one. This dissociation is the RQ3 finding.")

    # does heavy augmentation help?
    for arch in sorted(set(r["arch"] for r in rows)):
        s = {r["strategy"]: r for r in rows if r["arch"] == arch}
        if "standard" in s and "heavy_aug" in s:
            d = s["heavy_aug"]["sens_retention"] - s["standard"]["sens_retention"]
            print(f"  {arch:18s} heavy_aug changes retention by {d:+.3f}")

    json.dump(rows, open(f"{RES}/rq3_comparison.json", "w"), indent=2)
    print(f"\nwrote {RES}/rq3_comparison.json")
    print("\nCAVEAT: one seed per cell. Differences smaller than run-to-run "
          "variance are not interpretable; re-run with more seeds to claim "
          "an architecture effect.")


if __name__ == "__main__":
    main()
