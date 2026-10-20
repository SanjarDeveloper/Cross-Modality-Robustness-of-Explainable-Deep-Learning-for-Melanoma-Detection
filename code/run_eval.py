"""Evaluation campaign: one frozen model, many imaging conditions.

Every row logged is (arch, strategy, seed, dataset, modality, factor,
severity, metrics...). The model NEVER changes inside a run -- only the
input condition does -- which is what makes a difference attributable to
modality rather than to the model.

Usage:
  # Headline RQ1/RQ2 result: real paired cross-modality
  python code/run_eval.py --ckpt checkpoints/resnet50_standard_s0.pt --suite real

  # RQ1 thresholds: simulated single-factor sweep
  python code/run_eval.py --ckpt ... --suite sweep

  # RQ4 core: faithfulness under shift
  python code/run_eval.py --ckpt ... --suite xai
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import data as D
from msim import ModalityShift
from eval.metrics import (all_metrics, mcnemar, delong, bootstrap_ci,
                          sensitivity, auroc)
from eval.xai import GradCAM, target_layer, deletion_insertion, localisation

from run_paths import output_root
ROOT = output_root()


def load_model(ckpt, dev):
    import timm
    c = torch.load(ckpt, map_location=dev, weights_only=False)
    m = timm.create_model(c["arch"], pretrained=False, num_classes=1)
    m.load_state_dict(c["state_dict"])
    return m.to(dev).eval(), c


@torch.no_grad()
def predict(model, recs, dev, size=224, shift=None, batch=64, workers=4,
            cache=None, indices=None):
    """cache: name of a precached array to read instead of decoding JPEGs."""
    if cache and D.cache_exists(cache):
        D.validate_cache(cache, recs)
        ds = D.CachedDataset(cache, size, shift=shift, indices=indices)
    else:
        if indices is not None:
            recs = [recs[i] for i in indices]
        ds = D.LesionDataset(recs, size, shift=shift)
    dl = DataLoader(ds, batch_size=batch, num_workers=workers,
                    pin_memory=(dev == "cuda"))
    P, Y = [], []
    for b in dl:
        x = b["x"].to(dev, non_blocking=True)
        with torch.autocast("cuda", torch.float16, enabled=(dev == "cuda")):
            lg = model(x).squeeze(1)
        P.append(torch.sigmoid(lg.float()).cpu().numpy())
        Y.append(b["y"].numpy())
    return np.concatenate(Y), np.concatenate(P)


def row(base, **kw):
    r = dict(base)
    r.update(kw)
    return r


# ------------------------------------------------------------------- suites
def suite_real(model, ck, dev, args, out):
    """RQ1/RQ2 headline: Derm7pt paired same-lesion + PAD-UFES-20."""
    thr = ck["threshold"]
    base = {"arch": ck["arch"], "strategy": ck["strategy"], "seed": ck["seed"]}
    rows = []

    clin, derm = D.derm7pt()
    sub = D.smoke_indices(derm, args.subset)
    yd, pd_ = predict(model, derm, dev, ck["size"], None, args.batch, args.workers,
                      cache="derm7pt_dermoscopy", indices=sub)
    yc, pc = predict(model, clin, dev, ck["size"], None, args.batch, args.workers,
                     cache="derm7pt_clinical", indices=sub)
    assert (yd == yc).all(), "pair alignment broken"

    rows.append(row(base, dataset="derm7pt", modality="dermoscopy",
                    factor="none", severity=0.0, **all_metrics(yd, pd_, thr)))
    rows.append(row(base, dataset="derm7pt", modality="clinical",
                    factor="none", severity=0.0, **all_metrics(yc, pc, thr)))

    # --- paired tests on the SAME lesions: the cleanest modality evidence
    b, c, p_mc = mcnemar(yd, pd_, pc, thr)
    a_d, a_c, p_dl = delong(yd, pd_, pc)
    mel = yd == 1
    sd = sensitivity(yd, pd_, thr)
    sc = sensitivity(yc, pc, thr)
    lo, hi = bootstrap_ci(
        lambda Y, A, B: sensitivity(Y, A, thr) - sensitivity(Y, B, thr),
        yd, pd_, pc, n_boot=args.boot)
    paired = {
        "n_pairs": int(len(yd)), "n_melanoma": int(mel.sum()),
        "sens_dermoscopy": sd, "sens_clinical": sc,
        "sens_drop": sd - sc, "sens_drop_ci95": [lo, hi],
        "auroc_dermoscopy": a_d, "auroc_clinical": a_c,
        "delong_p": p_dl, "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p_mc,
    }
    tag = f"{ck['arch']}_{ck['strategy']}_s{ck['seed']}"
    paired.update(tag=tag, mcnemar_population="all paired cases, correctness")
    json.dump(paired, open(f"{out}/derm7pt_paired_{tag}.json", "w"), indent=2)
    print("\n--- Derm7pt paired (same lesion, two modalities) ---")
    print(json.dumps(paired, indent=2))

    pad = D.pad_ufes_20()
    yp, pp = predict(model, pad, dev, ck["size"], None, args.batch, args.workers,
                     cache="pad_ufes_20", indices=D.smoke_indices(pad, args.subset))
    rows.append(row(base, dataset="pad_ufes_20", modality="smartphone",
                    factor="none", severity=0.0, **all_metrics(yp, pp, thr)))

    ham = D.ham10000("test", seed=ck["seed"])
    yh, ph = predict(model, ham, dev, ck["size"], None, args.batch, args.workers,
                     cache="ham_test", indices=D.smoke_indices(ham, args.subset))
    rows.append(row(base, dataset="ham10000", modality="dermoscopy",
                    factor="none", severity=0.0, **all_metrics(yh, ph, thr)))
    return rows


def suite_sweep(model, ck, dev, args, out):
    """RQ1/RQ2: single-factor severity sweep on held-out dermoscopy."""
    from msim import FACTORS
    thr = ck["threshold"]
    base = {"arch": ck["arch"], "strategy": ck["strategy"], "seed": ck["seed"]}
    recs = D.ham10000("test", seed=ck["seed"])
    rows = []
    sevs = [round(s, 3) for s in np.linspace(0, 1, args.levels)]
    for f in FACTORS:
        for s in sevs:
            sh = None if s == 0 else ModalityShift(f, s)
            y, p = predict(model, recs, dev, ck["size"], sh, args.batch,
                           args.workers, cache="ham_test",
                           indices=D.smoke_indices(recs, args.subset))
            m = all_metrics(y, p, thr)
            lo, hi = bootstrap_ci(lambda Y, P: sensitivity(Y, P, thr), y, p,
                                  n_boot=args.boot)
            rows.append(row(base, dataset="ham10000", modality="simulated",
                            factor=f, severity=s, sens_ci_lo=lo, sens_ci_hi=hi, **m))
            print(f"  {f:13s} s={s:.2f}  sens={m['sensitivity']:.3f} "
                  f"[{lo:.3f},{hi:.3f}]  auroc={m['auroc']:.3f}", flush=True)
    return rows


def suite_xai(model, ck, dev, args, out):
    """RQ4 core: faithfulness + localisation under modality shift.

    Runs on a fixed subset (cost control) that is stratified to keep
    melanomas and non-melanomas balanced enough for a per-class breakdown --
    the comparison motivated by reference [17].
    """
    thr = ck["threshold"]
    base = {"arch": ck["arch"], "strategy": ck["strategy"], "seed": ck["seed"]}
    layer = target_layer(model, ck["arch"])
    rows = []

    def run(recs, tag, modality, factor, severity, shift, want_mask,
            cache=None, indices=None):
        if cache and D.cache_exists(cache):
            full = (D.ham10000("test", seed=ck["seed"]) if cache == "ham_test"
                    else D.derm7pt()[0 if modality == "clinical" else 1])
            D.validate_cache(cache, full)
            ds = D.CachedDataset(cache, ck["size"], shift=shift,
                                 want_mask=want_mask, indices=indices)
        else:
            ds = D.LesionDataset(recs, ck["size"], shift=shift, want_mask=want_mask)
        dl = DataLoader(ds, batch_size=args.xai_batch, num_workers=args.workers)
        DA, IA, IOU, HIT, Y, P = [], [], [], [], [], []
        with GradCAM(model, layer) as cam:
            for b in dl:
                x = b["x"].to(dev)
                s = cam(x)
                d, i = deletion_insertion(model, x, s, steps=args.steps)
                DA.append(d); IA.append(i)
                with torch.no_grad():
                    P.append(torch.sigmoid(model(x).squeeze(1).float()).cpu().numpy())
                Y.append(b["y"].numpy())
                if want_mask:
                    iou, hit = localisation(s, b["mask"].to(dev))
                    IOU.append(iou); HIT.append(hit)
        Y = np.concatenate(Y); P = np.concatenate(P)
        d = np.concatenate(DA); i = np.concatenate(IA)
        out_row = dict(base, dataset=tag, modality=modality, factor=factor,
                       severity=severity, n=len(Y), n_mel=int(Y.sum()),
                       sensitivity=sensitivity(Y, P, thr), auroc=auroc(Y, P),
                       deletion_auc=float(np.mean(d)), insertion_auc=float(np.mean(i)))
        # Legacy "ben" keys mean non-melanoma, which includes other cancers.
        for cls, name in ((1, "mel"), (0, "ben")):
            m = Y == cls
            if m.sum():
                out_row[f"deletion_auc_{name}"] = float(d[m].mean())
                out_row[f"insertion_auc_{name}"] = float(i[m].mean())
        if IOU:
            iou = np.concatenate(IOU); hit = np.concatenate(HIT)
            out_row["iou"] = float(np.nanmean(iou))
            out_row["pointing"] = float(np.nanmean(hit))
            for cls, name in ((1, "mel"), (0, "ben")):
                m = Y == cls
                if m.sum():
                    out_row[f"iou_{name}"] = float(np.nanmean(iou[m]))
                    out_row[f"pointing_{name}"] = float(np.nanmean(hit[m]))
        print(f"  {tag:12s} {factor:12s} s={severity:.2f} "
              f"del={out_row['deletion_auc']:.3f} ins={out_row['insertion_auc']:.3f} "
              f"iou={out_row.get('iou', float('nan')):.3f}", flush=True)
        return out_row

    # --- real modality shift, same lesions (masks unavailable here)
    clin, derm = D.derm7pt()
    idx = stratified_subset([r.label for r in derm], args.xai_n, seed=0)
    rows.append(run([derm[i] for i in idx], "derm7pt", "dermoscopy", "none", 0.0,
                    None, False, cache="derm7pt_dermoscopy", indices=idx))
    rows.append(run([clin[i] for i in idx], "derm7pt", "clinical", "none", 0.0,
                    None, False, cache="derm7pt_clinical", indices=idx))

    # --- simulated shift on HAM10000, where lesion masks DO exist, so the
    #     localisation axis is measurable
    ham = D.ham10000("test", seed=ck["seed"])
    hidx = stratified_subset([r.label for r in ham], args.xai_n, seed=0)
    hsub = [ham[i] for i in hidx]
    rows.append(run(hsub, "ham10000", "dermoscopy", "none", 0.0, None, True,
                    cache="ham_test", indices=hidx))
    from msim import FACTORS
    for f in FACTORS:
        for s in (0.5, 1.0):
            rows.append(run(hsub, "ham10000", "simulated", f, s,
                            ModalityShift(f, s), True,
                            cache="ham_test", indices=hidx))
    return rows


def stratified_subset(labels, n, seed=0):
    """Keep the melanoma fraction usable for a per-class comparison."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    k = min(n // 2, len(pos))
    take_p = rng.choice(pos, k, replace=False)
    take_n = rng.choice(neg, min(n - k, len(neg)), replace=False)
    return np.sort(np.concatenate([take_p, take_n]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default="real", choices=["real", "sweep", "xai"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--xai_batch", type=int, default=16)
    # See train.py: on Windows + cached arrays, workers=0 is ~3.7x FASTER.
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--levels", type=int, default=6, help="severity levels 0..1")
    ap.add_argument("--steps", type=int, default=32, help="deletion/insertion steps")
    ap.add_argument("--xai_n", type=int, default=400)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--subset", type=int, default=0,
                    help="evaluate only the first N cases (smoke test)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(a.ckpt, dev)
    out = f"{ROOT}/results"
    os.makedirs(out, exist_ok=True)
    print(f"model={ck['arch']} strategy={ck['strategy']} thr={ck['threshold']:.4f} "
          f"baseline_test_auroc={ck['test_metrics']['auroc']:.4f}")

    t0 = time.time()
    rows = {"real": suite_real, "sweep": suite_sweep, "xai": suite_xai}[a.suite](
        model, ck, dev, a, out)
    tag = f"{ck['arch']}_{ck['strategy']}_s{ck['seed']}"
    path = f"{out}/{a.suite}_{tag}.json"
    json.dump(rows, open(path, "w"), indent=2)
    print(f"\nwrote {path}  ({len(rows)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
