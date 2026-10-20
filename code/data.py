"""Datasets and leakage-safe splits.

Design decisions that matter scientifically:

1. TRAIN ONLY ON HAM10000 (dermoscopy). Derm7pt is never trained on, so all
   1011 paired cases stay available as test data -> 252 melanomas -> power
   0.78 for a 10% sensitivity drop. Training on Derm7pt would cut this to
   101 melanomas (power 0.36) AND contaminate the cross-modality question.

2. SPLIT HAM10000 BY lesion_id, NOT image_id. 1956 lesions have >1 image;
   an image-level split leaks the same lesion into train and test and
   inflates the baseline, which would make every degradation number wrong.

3. Derm7pt gives two images of the SAME lesion (clinic = phone-like,
   derm = dermoscopic). Paired by construction, so McNemar/DeLong on the
   same lesions isolates modality with no confound.
"""
import csv, os, sys, hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

sys.path.insert(0, os.path.dirname(__file__))
import labels as L

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATASETS"))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _read(path, **kw):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, **kw))


def _stable_hash(s):
    """Deterministic across runs and platforms (unlike hash())."""
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------- records
class Rec:
    """One evaluation unit: an image path, a binary label, and provenance."""
    __slots__ = ("img", "label", "dataset", "modality", "group", "mask", "meta")

    def __init__(self, img, label, dataset, modality, group, mask=None, meta=None):
        self.img, self.label = img, label
        self.dataset, self.modality, self.group = dataset, modality, group
        self.mask, self.meta = mask, meta or {}


def ham10000(split=None, val_frac=0.15, test_frac=0.15, seed=0):
    """Dermoscopy training set. Split by lesion_id (leakage-safe)."""
    rows = _read(f"{ROOT}/ham10000/HAM10000_metadata.tab", delimiter="\t")
    recs = []
    for r in rows:
        iid = r["image_id"]
        seg = f"{ROOT}/ham10000/segmentations/{iid}_segmentation.png"
        recs.append(Rec(
            img=f"{ROOT}/ham10000/images/{iid}.jpg",
            label=L.ham10000(r["dx"]),
            dataset="ham10000", modality="dermoscopy",
            group=r["lesion_id"],
            mask=seg if os.path.exists(seg) else None,
            meta={"image_id": iid, "dx": r["dx"]},
        ))
    if split is None:
        return recs
    # deterministic lesion-level assignment
    def bucket(rec):
        h = (_stable_hash(rec.group) ^ _stable_hash(seed)) % 10000 / 10000.0
        if h < test_frac:
            return "test"
        if h < test_frac + val_frac:
            return "val"
        return "train"
    return [r for r in recs if bucket(r) == split]


def derm7pt():
    """PRIMARY cross-modality test set: 1011 lesions x 2 modalities.

    Returns (clinical, dermoscopic) lists aligned index-for-index, so
    clinical[i] and dermoscopic[i] are the SAME lesion. Never used for
    training -- all 252 melanomas remain test data.
    """
    rows = _read(f"{ROOT}/derm7pt/meta/meta.csv")
    clin, derm = [], []
    for i, r in enumerate(rows):
        y = L.derm7pt(r["diagnosis"])
        gid = r.get("case_id") or f"case{r['case_num']}"
        common = dict(dataset="derm7pt", group=gid,
                      meta={"case_num": r["case_num"], "dx": r["diagnosis"], "pair": i})
        clin.append(Rec(img=f"{ROOT}/derm7pt/images/{r['clinic']}", label=y,
                        modality="clinical", **common))
        derm.append(Rec(img=f"{ROOT}/derm7pt/images/{r['derm']}", label=y,
                        modality="dermoscopy", **common))
    assert len(clin) == len(derm) == 1011
    assert all(c.label == d.label for c, d in zip(clin, derm)), "pair label mismatch"
    return clin, derm


def pad_ufes_20():
    """Real smartphone images. Only 52 melanomas -> supporting evidence."""
    rows = _read(f"{ROOT}/pad_ufes_20/metadata.csv")
    return [Rec(img=f"{ROOT}/pad_ufes_20/images/{r['img_id']}",
                label=L.pad_ufes_20(r["diagnostic"]),
                dataset="pad_ufes_20", modality="smartphone",
                group=r["patient_id"],
                meta={"dx": r["diagnostic"], "fitzpatrick": r.get("fitspatrick", ""),
                      "biopsed": r.get("biopsed", "")})
            for r in rows]


def ddi():
    """Skin-tone fairness (RQ5). 21 melanomas, 7 per tone -> EXPLORATORY."""
    rows = _read(f"{ROOT}/ddidiversedermatologyimages/ddi_metadata.csv")
    return [Rec(img=f"{ROOT}/ddidiversedermatologyimages/{r['DDI_file']}",
                label=L.ddi(r["disease"]),
                dataset="ddi", modality="clinical", group=r["DDI_ID"],
                meta={"skin_tone": r["skin_tone"], "disease": r["disease"],
                      "malignant": r["malignant"]})
            for r in rows]


# ----------------------------------------------------------------- torch ds
def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"unreadable image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class LesionDataset(Dataset):
    """Records -> tensors. `shift` is an optional Modality-Shift transform
    applied BEFORE resize/normalise, so degradation acts on the raw image
    exactly as a real camera would produce it."""

    def __init__(self, recs, size=224, train_aug=False, shift=None, want_mask=False):
        self.recs, self.size, self.shift = recs, size, shift
        self.want_mask = want_mask
        self.tf = self._build(train_aug, size)

    @staticmethod
    def _build(train_aug, size):
        import albumentations as A
        if train_aug:
            return A.Compose([
                A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(0.05, 0.1, 20, p=0.7),
                A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
            ])
        return A.Compose([A.Resize(size, size)])

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = load_rgb(r.img)
        if self.shift is not None:
            img = self.shift(img)
        mask = None
        if self.want_mask and r.mask:
            mask = cv2.imread(r.mask, cv2.IMREAD_GRAYSCALE)
        out = self.tf(image=img, mask=mask) if mask is not None else self.tf(image=img)
        x = out["image"].astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1))
        item = {"x": x, "y": r.label, "idx": i}
        if self.want_mask:
            m = out.get("mask")
            item["mask"] = (torch.from_numpy((m > 127).astype(np.float32))
                            if m is not None else torch.zeros(self.size, self.size))
            item["has_mask"] = m is not None
        return item


class CachedDataset(Dataset):
    """Same interface as LesionDataset but reads pre-decoded 256px arrays.

    Removes the JPEG-decode bottleneck measured at ~30 img/s (vs ~650 img/s
    of raw GPU throughput). Degradation is applied to the cached 256px image,
    which is what the simulator severities were calibrated against.
    """

    def __init__(self, name, size=224, train_aug=False, shift=None,
                 want_mask=False, indices=None, cache_dir=None):
        import json
        cd = cache_dir or os.path.join(ROOT, "..", "cache")
        cd = os.path.abspath(cd)
        self.img = np.load(f"{cd}/{name}_img.npy", mmap_mode="r")
        self.meta = json.load(open(f"{cd}/{name}_meta.json"))
        self.mask = (np.load(f"{cd}/{name}_mask.npy", mmap_mode="r")
                     if self.meta["has_mask"] and want_mask else None)
        self.labels = np.asarray(self.meta["labels"])
        self.idx = np.arange(len(self.img)) if indices is None else np.asarray(indices)
        self.size, self.shift, self.want_mask = size, shift, want_mask
        self.tf = LesionDataset._build(train_aug, size)

    def __len__(self):
        return len(self.idx)

    def labels_of(self):
        return self.labels[self.idx]

    def __getitem__(self, i):
        j = int(self.idx[i])
        img = np.asarray(self.img[j])
        if self.shift is not None:
            img = self.shift(img)
        m = np.asarray(self.mask[j]) if self.mask is not None else None
        out = self.tf(image=img, mask=m) if m is not None else self.tf(image=img)
        x = out["image"].astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        item = {"x": torch.from_numpy(x.transpose(2, 0, 1)),
                "y": int(self.labels[j]), "idx": j}
        if self.want_mask:
            mm = out.get("mask")
            item["mask"] = (torch.from_numpy((mm > 127).astype(np.float32))
                            if mm is not None else torch.zeros(self.size, self.size))
        return item


def cache_exists(name, cache_dir=None):
    cd = os.path.abspath(cache_dir or os.path.join(ROOT, "..", "cache"))
    return os.path.exists(f"{cd}/{name}_img.npy")


def validate_cache(name, recs, cache_dir=None):
    """Reject cached split/order mismatches instead of changing input resolution."""
    import json
    cd = os.path.abspath(cache_dir or os.path.join(ROOT, "..", "cache"))
    with open(os.path.join(cd, f"{name}_meta.json")) as f:
        meta = json.load(f)
    expected = ([str(r.group) for r in recs], [int(r.label) for r in recs],
                [r.meta for r in recs])
    actual = (meta["groups"], meta["labels"], meta["meta"])
    arr = np.load(os.path.join(cd, f"{name}_img.npy"), mmap_mode="r")
    if actual != expected or len(arr) != len(recs) or meta["n"] != len(recs):
        raise ValueError(f"Cache {name} does not match requested split/order/labels. "
                         "Do not reuse a seed-0 cache for another split seed.")


def smoke_indices(recs, n, seed=0):
    """Small deterministic, two-class rehearsal subset, not a study sample."""
    if not n:
        return None
    if n < 2:
        raise ValueError("Smoke subset must contain at least two cases.")
    labels = np.asarray([r.label for r in recs])
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if not len(pos) or not len(neg):
        raise ValueError("Smoke subset requires both melanoma and non-melanoma.")
    n = min(n, len(recs))
    k = min(len(pos), max(1, round(n * len(pos) / len(recs))))
    k = max(k, n - len(neg))
    k = min(k, n - 1)
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate((rng.choice(pos, k, replace=False),
                                   rng.choice(neg, n - k, replace=False))))


def pos_weight(recs_or_labels):
    """Class imbalance weight for BCEWithLogitsLoss (HAM10000 is 11% melanoma).

    Accepts either a list of Rec or a sequence of 0/1 labels, so the cached
    and uncached paths can share it.
    """
    seq = recs_or_labels
    labels = [r.label if isinstance(r, Rec) else int(r) for r in seq]
    p = sum(labels)
    return torch.tensor([(len(labels) - p) / max(p, 1)], dtype=torch.float32)
