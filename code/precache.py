"""Pre-decode every image once to a 256px uint8 memmap.

MEASURED PROBLEM: the pipeline runs at ~30 img/s while the GPU alone can do
~650 img/s -- i.e. ~5% GPU utilisation. The bottleneck is JPEG decode +
resize of full-resolution files (HAM10000 is 600x450, Derm7pt larger), not
compute. A bigger GPU would therefore buy almost nothing; removing the decode
does.

Images are stored at 256px (not 224) so the training-time RandomResizedCrop
still has margin to crop from. Degradation factors are applied AFTER this
cache, on the 256px array, which is what they were calibrated on anyway.

  python code/precache.py            # build every split
  python code/precache.py --check    # verify an existing cache
"""
import argparse, os, sys, json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
import data as D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = f"{ROOT}/cache"
SIZE = 256


def sources():
    clin, derm = D.derm7pt()
    return {
        "ham_train": D.ham10000("train"),
        "ham_val": D.ham10000("val"),
        "ham_test": D.ham10000("test"),
        "derm7pt_clinical": clin,
        "derm7pt_dermoscopy": derm,
        "pad_ufes_20": D.pad_ufes_20(),
        "ddi": D.ddi(),
    }


def build(name, recs, want_mask=True):
    os.makedirs(CACHE, exist_ok=True)
    n = len(recs)
    img_path = f"{CACHE}/{name}_img.npy"
    meta_path = f"{CACHE}/{name}_meta.json"
    has_mask = want_mask and any(r.mask for r in recs)

    arr = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8,
                                    shape=(n, SIZE, SIZE, 3))
    marr = None
    if has_mask:
        marr = np.lib.format.open_memmap(f"{CACHE}/{name}_mask.npy", mode="w+",
                                         dtype=np.uint8, shape=(n, SIZE, SIZE))
    labels, groups, mask_ok = [], [], []
    for i, r in enumerate(recs):
        img = D.load_rgb(r.img)
        arr[i] = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        labels.append(int(r.label))
        groups.append(str(r.group))
        if has_mask:
            if r.mask:
                m = cv2.imread(r.mask, cv2.IMREAD_GRAYSCALE)
                marr[i] = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
                mask_ok.append(True)
            else:
                marr[i] = 0
                mask_ok.append(False)
        if (i + 1) % 500 == 0:
            print(f"  {name}: {i+1}/{n}", flush=True)
    arr.flush()
    if marr is not None:
        marr.flush()
    json.dump({"n": n, "size": SIZE, "labels": labels, "groups": groups,
               "has_mask": has_mask, "mask_ok": mask_ok,
               "meta": [r.meta for r in recs]},
              open(meta_path, "w"))
    print(f"  {name}: {n} images -> {os.path.getsize(img_path)/1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    src = sources()
    if a.only:
        src = {a.only: src[a.only]}
    if a.check:
        for name, recs in src.items():
            p = f"{CACHE}/{name}_img.npy"
            if not os.path.exists(p):
                print(f"  MISSING {name}")
                continue
            arr = np.load(p, mmap_mode="r")
            meta = json.load(open(f"{CACHE}/{name}_meta.json"))
            ok = len(arr) == len(recs) == meta["n"]
            print(f"  {name:20s} n={len(arr):5d} {'OK' if ok else 'MISMATCH'} "
                  f"mel={sum(meta['labels'])}")
        return
    for name, recs in src.items():
        build(name, recs)
    print("\ncache ->", CACHE)


if __name__ == "__main__":
    main()
