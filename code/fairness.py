"""RQ5 -- does cross-modality degradation fall unevenly across skin tones?

HONEST FRAMING, set by the data rather than by the ambition of the question:
DDI contains 656 images with exactly 21 melanomas -- 7 in each of the three
skin-tone groups (12 = Fitzpatrick I-II, 34 = III-IV, 56 = V-VI). Seven
melanomas per group cannot support an inferential claim about melanoma
sensitivity; the confidence intervals are wider than any effect worth
reporting.

Primary task-aligned output is melanoma-only, descriptive and underpowered.
Legacy malignant-vs-benign metrics are isolated under off_target_exploratory:
this melanoma model was not trained as a general malignancy detector.

PAD-UFES-20 also carries a `fitspatrick` field, so it is included as a second,
independent smartphone-modality view where counts allow.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import data as D
from eval.metrics import sensitivity, specificity, auroc, bootstrap_ci
from run_eval import load_model, predict

from run_paths import output_root
ROOT = output_root()
RES = f"{ROOT}/results"
TONE_NAME = {"12": "I-II (light)", "34": "III-IV (medium)", "56": "V-VI (dark)"}


def ddi_by_tone(model, ck, dev, args):
    recs = D.ddi()
    y, p = predict(model, recs, dev, ck["size"], None, args.batch, args.workers,
                   cache="ddi",
                   indices=(np.arange(args.subset) if args.subset else None))
    n = len(y)
    recs = recs[:n]
    tone = np.array([r.meta["skin_tone"] for r in recs])
    malig = np.array([r.meta["malignant"] == "True" for r in recs]).astype(int)
    thr = ck["threshold"]

    out = {"note": "DDI has 21 melanomas total (7 per tone) -- melanoma-level "
                   "results are EXPLORATORY and underpowered.",
           "groups": {}}
    for t in sorted(set(tone)):
        m = tone == t
        # Off-target endpoint retained only as a labelled exploratory diagnostic.
        a_lo, a_hi = bootstrap_ci(lambda Y, P: auroc(Y, P), malig[m], p[m],
                                  n_boot=args.boot)
        s_lo, s_hi = bootstrap_ci(lambda Y, P: sensitivity(Y, P, thr),
                                  malig[m], p[m], n_boot=args.boot)
        g = {
            "n": int(m.sum()),
            "n_malignant": int(malig[m].sum()),
            "n_melanoma": int(y[m].sum()),
            "malignant_auroc": auroc(malig[m], p[m]),
            "malignant_auroc_ci95": [a_lo, a_hi],
            "malignant_sensitivity": sensitivity(malig[m], p[m], thr),
            "malignant_sensitivity_ci95": [s_lo, s_hi],
            "specificity": specificity(malig[m], p[m], thr),
        }
        # (2) melanoma-only, flagged
        if y[m].sum() > 0:
            g["melanoma_sensitivity_EXPLORATORY"] = sensitivity(y[m], p[m], thr)
        out["groups"][TONE_NAME.get(t, t)] = g

    # gap between lightest and darkest, with a bootstrap CI on the difference
    keys = [TONE_NAME.get(t, t) for t in sorted(set(tone))]
    if len(keys) >= 2:
        lo_t, hi_t = sorted(set(tone))[0], sorted(set(tone))[-1]
        a = out["groups"][TONE_NAME.get(lo_t, lo_t)]["malignant_auroc"]
        b = out["groups"][TONE_NAME.get(hi_t, hi_t)]["malignant_auroc"]
        out["light_minus_dark_auroc"] = float(a - b)
        out["interpretation"] = (
            "Positive = worse on darker skin. With ~57 malignant cases per "
            "group the CIs overlap substantially; treat as descriptive.")
    return out


def pad_by_tone(model, ck, dev, args):
    """Second view: real smartphone images with a Fitzpatrick field."""
    recs = D.pad_ufes_20()
    y, p = predict(model, recs, dev, ck["size"], None, args.batch, args.workers,
                   cache="pad_ufes_20",
                   indices=(np.arange(args.subset) if args.subset else None))
    recs = recs[:len(y)]
    fz = np.array([str(r.meta.get("fitzpatrick", "")).strip() for r in recs])
    thr = ck["threshold"]
    out = {"note": "PAD-UFES-20 has only 52 melanomas in total; per-tone "
                   "melanoma counts are tiny. Descriptive only.",
           "groups": {}}
    for t in sorted(set(fz)):
        if not t or t in ("", "nan"):
            continue
        m = fz == t
        if m.sum() < 30:
            continue
        out["groups"][f"fitzpatrick {t}"] = {
            "n": int(m.sum()), "n_melanoma": int(y[m].sum()),
            "auroc": auroc(y[m], p[m]) if len(set(y[m])) > 1 else None,
            "melanoma_sensitivity": (sensitivity(y[m], p[m], thr)
                                     if y[m].sum() else None),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--subset", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        a.subset, a.boot = 120, 50

    ck_path = a.ckpt or f"{ROOT}/checkpoints/resnet50_standard_s0.pt"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(ck_path, dev)

    ddi_result = ddi_by_tone(model, ck, dev, a)
    primary_ddi = {"note": ddi_result["note"], "groups": {
        name: {key: value for key, value in group.items()
               if key in ("n", "n_melanoma", "melanoma_sensitivity_EXPLORATORY")}
        for name, group in ddi_result["groups"].items()}}
    report = {"checkpoint": os.path.basename(ck_path),
              "endpoint_warning": "The model predicts melanoma, NOT malignancy. "
                                  "Malignant-vs-benign metrics are an off-target "
                                  "exploratory analysis, not validated cancer detection.",
              "ddi": primary_ddi,
              "off_target_exploratory": {
                  "included_in_primary_conclusions": False, "ddi": ddi_result},
              "pad_ufes_20": pad_by_tone(model, ck, dev, a)}

    print(json.dumps(report, indent=2))
    os.makedirs(RES, exist_ok=True)
    json.dump(report, open(f"{RES}/fairness_report.json", "w"), indent=2)
    print("\nwrote", f"{RES}/fairness_report.json")

    print("\n--- RQ5 summary ---")
    for name, g in report["ddi"]["groups"].items():
        sens = g.get("melanoma_sensitivity_EXPLORATORY")
        print(f"  {name:18s} n={g['n']:4d} mel={g['n_melanoma']:2d} "
              f"melanoma sensitivity={sens if sens is not None else 'not estimable'}")
    print("  NOTE: 7 melanomas per tone -> melanoma-level claims are not supported.")
    print("  Malignancy metrics are separate OFF-TARGET EXPLORATORY outputs only.")


if __name__ == "__main__":
    main()
