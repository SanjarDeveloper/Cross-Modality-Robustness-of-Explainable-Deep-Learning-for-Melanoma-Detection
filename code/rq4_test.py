"""Statistical test for the RQ4 claim, and the drift->error model.

The analyse.py summary reports faithfulness CHANGES but no test. The claims
that need one are:
  (a) faithfulness drops under modality shift (paired, same lesions);
  (b) it drops MORE for melanoma than for non-melanoma (interaction);
  (c) explanation drift predicts misclassification.

This re-runs Grad-CAM per lesion and keeps PER-IMAGE values (analyse.py only
stores means), which is what paired testing needs.

  python code/rq4_test.py --ckpt checkpoints/resnet50_standard_s0.pt --n 400
"""
import argparse, json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import data as D
from eval.xai import GradCAM, target_layer, deletion_insertion
from run_eval import load_model, stratified_subset

from run_paths import output_root
ROOT = output_root()


def per_image(model, ck, dev, recs, steps, batch, workers, cache=None, idx=None):
    if cache and D.cache_exists(cache):
        D.validate_cache(cache, D.derm7pt()[0 if cache.endswith("clinical") else 1])
        ds = D.CachedDataset(cache, ck["size"], indices=idx)
    else:
        ds = D.LesionDataset(recs, ck["size"])
    dl = DataLoader(ds, batch_size=batch, num_workers=workers)
    INS, DEL, P, Y = [], [], [], []
    with GradCAM(model, target_layer(model, ck["arch"])) as cam:
        for b in dl:
            x = b["x"].to(dev)
            s = cam(x)
            d, i = deletion_insertion(model, x, s, steps=steps)
            DEL.append(d); INS.append(i)
            with torch.no_grad():
                P.append(torch.sigmoid(model(x).squeeze(1).float()).cpu().numpy())
            Y.append(b["y"].numpy())
    return (np.concatenate(Y), np.concatenate(P),
            np.concatenate(INS), np.concatenate(DEL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    from scipy import stats

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(a.ckpt, dev)
    thr = ck["threshold"]

    clin, derm = D.derm7pt()
    idx = stratified_subset([r.label for r in derm], a.n, seed=0)
    yd, pd_, ind, deld = per_image(model, ck, dev, [derm[i] for i in idx],
                                   a.steps, a.batch, a.workers,
                                   cache="derm7pt_dermoscopy", idx=idx)
    yc, pc, inc, delc = per_image(model, ck, dev, [clin[i] for i in idx],
                                  a.steps, a.batch, a.workers,
                                  cache="derm7pt_clinical", idx=idx)
    assert (yd == yc).all()
    mel = yd == 1

    out = {"n": int(len(yd)), "n_melanoma": int(mel.sum()),
           "negative_class": "non-melanoma; legacy 'benign' keys do not imply benignity"}

    # (a) paired drop in faithfulness, same lesions
    d_ins = inc - ind
    t, p = stats.wilcoxon(ind, inc)
    out["faithfulness_drop_all"] = {
        "mean_change": float(d_ins.mean()),
        "wilcoxon_p": float(p),
        "cohens_d": float(d_ins.mean() / d_ins.std(ddof=1)),
    }

    # Legacy "benign" output keys refer to non-melanoma, including other cancers.
    dm, db = d_ins[mel], d_ins[~mel]
    u, pu = stats.mannwhitneyu(dm, db, alternative="less")   # more negative
    out["melanoma_vs_benign_drop"] = {
        "mean_drop_melanoma": float(dm.mean()),
        "mean_drop_benign": float(db.mean()),
        "difference": float(dm.mean() - db.mean()),
        "mannwhitney_p": float(pu),
        "cohens_d": float((dm.mean() - db.mean())
                          / np.sqrt((dm.var(ddof=1) + db.var(ddof=1)) / 2)),
    }

    # (c) does explanation drift predict misclassification on the shifted image?
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    wrong = ((pc >= thr).astype(int) != yc).astype(int)
    drift = np.column_stack([ind - inc, delc - deld])        # positive = worse
    keep = np.isfinite(drift).all(1)
    if wrong[keep].sum() > 5 and (1 - wrong[keep]).sum() > 5:
        lr = LogisticRegression(max_iter=2000).fit(drift[keep], wrong[keep])
        auc = roc_auc_score(wrong[keep], lr.predict_proba(drift[keep])[:, 1])
        out["drift_predicts_error"] = {
            "roc_auc": float(auc),
            "n_wrong": int(wrong[keep].sum()),
            "coef_insertion_drift": float(lr.coef_[0][0]),
            "coef_deletion_drift": float(lr.coef_[0][1]),
            "note": "in-sample fit; report as association, not a validated predictor",
        }

    print(json.dumps(out, indent=2))
    p = f"{ROOT}/results/rq4_tests_{ck['arch']}_{ck['strategy']}_s{ck['seed']}.json"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main()
