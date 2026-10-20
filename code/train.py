"""Train the frozen dermoscopic baseline (Week 1 deliverable).

Trains on HAM10000 ONLY. Derm7pt is never seen during training, which is
what keeps all 252 of its melanomas available as paired test data.

Usage:
  python code/train.py --arch resnet50 --epochs 8
  python code/train.py --arch efficientnet_b3 --epochs 8 --batch 16
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import data as D
from eval.metrics import all_metrics, threshold_at_sensitivity, auroc

from run_paths import output_root
ROOT = output_root()
TIMM_NAMES = {
    "resnet50": "resnet50",
    "efficientnet_b3": "efficientnet_b3",
    "densenet121": "densenet121",
}


def _heavy_tf(size):
    """RQ3 'heavy_aug' strategy.

    Deliberately mirrors the acquisition factors the simulator varies (blur,
    colour/illumination, scale, compression-like noise) WITHOUT importing the
    simulator itself -- training on the exact test-time transform would leak
    the evaluation into training and inflate robustness.
    """
    import albumentations as A
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(0.1, 0.25, 45, p=0.8),
        A.ColorJitter(0.4, 0.4, 0.4, 0.15, p=0.8),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 9)),
                 A.MotionBlur(blur_limit=9),
                 A.Defocus(radius=(3, 8))], p=0.5),
        A.OneOf([A.GaussNoise(), A.ImageCompression(quality_range=(30, 90))], p=0.4),
        A.RandomBrightnessContrast(0.3, 0.3, p=0.6),
    ])


def build(arch, device):
    import timm
    m = timm.create_model(TIMM_NAMES[arch], pretrained=True, num_classes=1)
    return m.to(device)


@torch.no_grad()
def infer(model, loader, device, amp=True):
    model.eval()
    P, Y = [], []
    for b in loader:
        x = b["x"].to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16, enabled=amp and device == "cuda"):
            logit = model(x).squeeze(1)
        P.append(torch.sigmoid(logit.float()).cpu().numpy())
        Y.append(b["y"].numpy())
    return np.concatenate(Y), np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50", choices=list(TIMM_NAMES))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    # Windows spawns a fresh process per worker; with the 256px cache the
    # per-item cost is ~0.6 ms, so worker overhead DOMINATES. Measured on this
    # machine: workers=0 -> 612 img/s, workers=4 -> 157 img/s. Default to 0.
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--strategy", default="standard",
                    choices=["standard", "heavy_aug"])
    ap.add_argument("--subset", type=int, default=0,
                    help="train on only N images (smoke test)")
    ap.add_argument("--no_cache", action="store_true",
                    help="decode JPEGs instead of using the 256px cache")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("WARNING: CUDA unavailable -- training on CPU will be very slow.")

    tr = D.ham10000("train", seed=a.seed)
    va = D.ham10000("val", seed=a.seed)
    te = D.ham10000("test", seed=a.seed)
    print(f"HAM10000  train={len(tr)} ({sum(r.label for r in tr)} mel)  "
          f"val={len(va)} ({sum(r.label for r in va)})  "
          f"test={len(te)} ({sum(r.label for r in te)})")

    # heavy_aug is the RQ3 robustness strategy: stronger colour/geometry
    # augmentation at training time, which is the cheap defence a practitioner
    # would try first against modality shift.
    heavy = a.strategy == "heavy_aug"
    dl = lambda ds, sh: DataLoader(ds, batch_size=a.batch, shuffle=sh,
                                   num_workers=a.workers, pin_memory=(dev == "cuda"),
                                   persistent_workers=a.workers > 0, drop_last=sh)

    use_cache = (not a.no_cache) and D.cache_exists("ham_train")
    print(f"image source: {'256px cache' if use_cache else 'raw JPEG decode'}")
    if use_cache:
        def mk(nm, aug_, recs):
            D.validate_cache(nm, recs)
            return D.CachedDataset(nm, a.size, train_aug=aug_,
                                   indices=D.smoke_indices(recs, a.subset, a.seed))
        tr_ds = mk("ham_train", True, tr)
        va_ds, te_ds = mk("ham_val", False, va), mk("ham_test", False, te)
        tr_labels = tr_ds.labels_of()
    else:
        if a.subset:
            tr, va, te = ([rs[i] for i in D.smoke_indices(rs, a.subset, a.seed)]
                          for rs in (tr, va, te))
        tr_ds = D.LesionDataset(tr, a.size, train_aug=True)
        va_ds, te_ds = D.LesionDataset(va, a.size), D.LesionDataset(te, a.size)
        tr_labels = tr
    tr_dl, va_dl, te_dl = dl(tr_ds, True), dl(va_ds, False), dl(te_ds, False)
    if heavy:
        tr_ds.tf = _heavy_tf(a.size)
    if not len(tr_dl):
        raise ValueError("Training subset is smaller than one full batch.")

    model = build(a.arch, dev)
    # Class imbalance: HAM10000 is 11% melanoma. Without pos_weight the model
    # collapses toward the majority class and sensitivity is meaningless.
    crit = nn.BCEWithLogitsLoss(pos_weight=D.pos_weight(tr_labels).to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * max(1, len(tr_dl)))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    best, best_state = -1, None
    hist = []
    for ep in range(a.epochs):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for i, b in enumerate(tr_dl):
            x = b["x"].to(dev, non_blocking=True)
            y = b["y"].float().to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16, enabled=(dev == "cuda")):
                loss = crit(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            seen += len(y)
            if i % 50 == 0:
                print(f"  ep{ep} {i}/{len(tr_dl)} loss {tot/max(seen,1):.4f}", flush=True)
        yv, pv = infer(model, va_dl, dev)
        av = auroc(yv, pv)
        hist.append({"epoch": ep, "train_loss": tot / max(seen, 1), "val_auroc": av,
                     "secs": time.time() - t0})
        print(f"epoch {ep}: loss {tot/max(seen,1):.4f}  val_auroc {av:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if av > best:
            best = av
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # Operating point is chosen on VALIDATION and then frozen for every
    # downstream condition, so degradation cannot be masked by re-tuning.
    yv, pv = infer(model, va_dl, dev)
    thr = threshold_at_sensitivity(yv, pv, 0.95)
    yt, pt = infer(model, te_dl, dev)
    m = all_metrics(yt, pt, thr)
    print("\n=== FROZEN DERMOSCOPIC BASELINE (HAM10000 test) ===")
    print(json.dumps(m, indent=2))
    print(f"threshold (95% sens on val) = {thr:.4f}")

    os.makedirs(f"{ROOT}/checkpoints", exist_ok=True)
    tag = f"{a.arch}_{a.strategy}_s{a.seed}"
    torch.save({"state_dict": model.state_dict(), "arch": a.arch,
                "strategy": a.strategy, "seed": a.seed, "size": a.size,
                "threshold": thr, "val_auroc": best, "test_metrics": m,
                "history": hist, "config": vars(a)},
               f"{ROOT}/checkpoints/{tag}.pt")
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    with open(f"{ROOT}/results/baseline_{tag}.json", "w") as f:
        json.dump({"arch": a.arch, "strategy": a.strategy, "seed": a.seed,
                   "threshold": thr, "val_auroc": best, "test": m,
                   "history": hist, "config": vars(a)}, f, indent=2)
    print(f"saved checkpoints/{tag}.pt")


if __name__ == "__main__":
    main()
