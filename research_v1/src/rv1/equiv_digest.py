"""Stage 2C1: fixed-input output digests, for cross-environment comparison.

Runs with numpy/cv2/albumentations/Pillow only (no torch), so the same code can be
executed in .venv_prep and .venv_model:
  <python> -B -m rv1.equiv_digest --config ../configs/loader_check_v1.json --out <file.json>

Inputs (fixed): the 29 Stage 2B2C prep_check sample records (evaluation input and, for
HAM, the 224 mask); the first 20 HAM-train records by stable_seed("equiv_v1", uid)
under aug_std_v1 at (train_seed, epoch) in (0,0), (0,1), (2,5), with masks.
Digest = SHA-256 of the exact array bytes (float32 x / bool mask).
"""
import argparse
import csv
import hashlib
import json
import os
import sys

from . import augment as AU, config as C, decode as DC, preprocess as PP, prep_check as PC
from .seeds import stable_seed


def _h(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def run(config_path):
    import numpy as np
    cfg = C.load(config_path)
    root = cfg["project_root"]
    ds_root, cache_dir = os.path.join(root, "DATASETS"), os.path.join(root, "cache")
    rows = {ds: DC.read_manifest(C.resolve(cfg, cfg["manifests"][ds]), ds)[0] for ds in PC.CACHES}

    def rd(path):
        with open(C.resolve(cfg, path), newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    sel = PC.select(rows, rd(cfg["supporting_inputs"]["mask_audit_csv"]),
                    rd(cfg["supporting_inputs"]["decode_audit_csv"]))
    out, readers = {}, {}
    for ds, cache, r, _ in sel:
        key = (cache, ds == "ham10000")
        if key not in readers:
            readers[key] = PP.EvalReader(rows[ds], cache, cache_dir, ds_root, want_mask=key[1])
        R = readers[key]
        it = R[next(i for i, q in enumerate(R.rows) if q["record_uid"] == r["record_uid"])]
        out[f"eval|{r['record_uid']}|x"] = _h(it["x"])
        if it["mask"] is not None:
            out[f"eval|{r['record_uid']}|mask"] = _h(it["mask"])
    R = readers.get(("ham_train", True)) or PP.EvalReader(rows["ham10000"], "ham_train", cache_dir,
                                                          ds_root, want_mask=True)
    train = sorted(R.rows, key=lambda q: (stable_seed("equiv_v1", q["record_uid"]), q["record_uid"]))[:20]
    for q in train:
        img = np.asarray(R.img[int(q["cache_index"])])
        les, mm = DC.interpret_mask(os.path.join(ds_root, *q["mask_path"].split("/")))
        m256 = AU.mask_to_cache_grid(les)
        for ts, ep in ((0, 0), (0, 1), (2, 5)):
            o = AU.augment(img, "aug_std_v1", ts, ep, q["image_uid"], m256, mm["flags"])
            out[f"aug|{q['record_uid']}|{ts}|{ep}|x"] = _h(o["x"])
            out[f"aug|{q['record_uid']}|{ts}|{ep}|mask"] = _h(o["mask"])
    import cv2
    import albumentations as A
    return {"python": sys.executable, "cv2": cv2.__version__, "numpy": np.__version__,
            "albumentations": A.__version__, "n_digests": len(out), "digests": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run(a.config)
    with open(a.out, "x", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(a.out, res["n_digests"])
