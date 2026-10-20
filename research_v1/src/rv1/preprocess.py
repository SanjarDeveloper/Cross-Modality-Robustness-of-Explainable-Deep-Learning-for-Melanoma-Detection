"""Deterministic evaluation preprocessing, prep_v1_cached256 (Stage 2B2C).

No augmentation, no perturbation, no model. Numpy output only (no torch).

Legacy cache (code/precache.py, inspected, read-only)
----------------------------------------------------
  image: cv2.imread(path, IMREAD_COLOR) -> BGR uint8 (alpha dropped, EXIF applied
         by OpenCV; all observed EXIF orientations are 1) -> cvtColor BGR2RGB ->
         cv2.resize((256,256), INTER_AREA) -> uint8 array (n, 256, 256, 3), RGB order.
  mask : cv2.imread(path, IMREAD_GRAYSCALE) -> cv2.resize((256,256), INTER_NEAREST)
         -> uint8 (n, 256, 256).  NOT used here (not mask_interp_v1; see masks below).
  row i of cache <name> == manifest row with cache_name=<name>, cache_index=i.

Evaluation image path (this module)
-----------------------------------
  cached uint8 RGB (256,256,3)
  -> cv2.resize(dsize=(224,224), interpolation=cv2.INTER_LINEAR)   [explicit bilinear]
  -> float32 / 255.0
  -> (x - MEAN) / STD, per channel, MEAN/STD float32 ImageNet constants (below)
  -> transpose to CHW, float32, shape (3, 224, 224), C-contiguous.
  This matches the ORDER of the legacy loader (code/data.py: CachedDataset with
  Albumentations Resize(224) -> /255 -> normalise). Albumentations is not used;
  equality of its Resize with the explicit cv2 call is checked separately as a
  diagnostic, not assumed. raw->224 is NOT equivalent to cached256->224.

Evaluation mask path (HAM only)
-------------------------------
  source mask -> decode.interpret_mask (mask_interp_v1, bool, original size)
  -> uint8 {0,1} -> cv2.resize((256,256), INTER_NEAREST_EXACT)
  -> cv2.resize((224,224), INTER_NEAREST_EXACT) -> bool (224,224).
  Both resizes stretch the full frame, exactly as the image path does (no crop,
  no padding). INTER_NEAREST_EXACT (pixel-centre aligned, like INTER_AREA and
  INTER_LINEAR) is chosen instead of legacy INTER_NEAREST, whose sampling is not
  centre-aligned. Masks are never normalised or interpolated bilinearly.

Importing this module has no side effects; cv2/numpy are imported lazily.
"""
import json
import os

from . import decode as DC

PREP_VERSION = "prep_v1_cached256"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CACHE_SIZE = 256
INPUT_SIZE = 224


class CacheError(ValueError):
    pass


def to_model_input(img256):
    """uint8 RGB (256,256,3) -> float32 normalised CHW (3,224,224)."""
    import cv2
    import numpy as np
    if img256.dtype != np.uint8 or img256.shape != (CACHE_SIZE, CACHE_SIZE, 3):
        raise CacheError(f"expected uint8 (256,256,3), got {img256.dtype} {img256.shape}")
    x = cv2.resize(np.ascontiguousarray(img256), (INPUT_SIZE, INPUT_SIZE),
                   interpolation=cv2.INTER_LINEAR)
    x = x.astype(np.float32) / np.float32(255.0)
    x = (x - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)


def mask_to_input_grid(lesion):
    """bool mask (H,W) at source resolution -> bool (224,224) via 256 (nearest-exact)."""
    import cv2
    import numpy as np
    m = lesion.astype(np.uint8)
    m = cv2.resize(m, (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_NEAREST_EXACT)
    m = cv2.resize(m, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST_EXACT)
    return m.astype(bool)


def rebuild_cache_entry(path):
    """Reconstruction of the documented legacy cache-building step (for checks)."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise DC.DecodeError(f"cv2_unreadable: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)


class EvalReader:
    """Manifest-backed, read-only access to one legacy cache.

    rows: manifest rows (dicts, from decode.read_manifest) of ONE dataset; only rows
    with row["cache_name"] == cache_name are used, in manifest order.
    """

    def __init__(self, rows, cache_name, cache_dir, datasets_root=None, want_mask=False):
        import numpy as np
        self.cache_name, self.want_mask = cache_name, want_mask
        self.datasets_root = datasets_root
        self.rows = [dict(r) for r in rows if r.get("cache_name") == cache_name]
        if not self.rows:
            raise CacheError(f"no manifest rows for cache {cache_name!r}")
        ip = os.path.join(cache_dir, f"{cache_name}_img.npy")
        with open(os.path.join(cache_dir, f"{cache_name}_meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.img = np.load(ip, mmap_mode="r")          # read-only memory map
        n = self.img.shape[0]
        if self.img.dtype != np.uint8 or self.img.shape[1:] != (CACHE_SIZE, CACHE_SIZE, 3):
            raise CacheError(f"{cache_name}: bad cache array {self.img.dtype} {self.img.shape}")
        if self.meta.get("n") != n or self.meta.get("size") != CACHE_SIZE:
            raise CacheError(f"{cache_name}: meta n/size {self.meta.get('n')}/"
                             f"{self.meta.get('size')} != array {n}/{CACHE_SIZE}")
        idx = []
        for r in self.rows:
            try:
                i = int(r["cache_index"])
            except (KeyError, ValueError):
                raise CacheError(f"{r.get('record_uid')}: missing/invalid cache_index") from None
            if not 0 <= i < n:
                raise CacheError(f"{r['record_uid']}: cache_index {i} out of range [0,{n})")
            if int(self.meta["labels"][i]) != int(r["label_mel"]):
                raise CacheError(f"{r['record_uid']}: label mismatch with cache meta row {i}")
            if str(self.meta["groups"][i]) != str(r["cluster_id"]):
                raise CacheError(f"{r['record_uid']}: group mismatch with cache meta row {i}")
            idx.append(i)
        if sorted(idx) != list(range(n)):
            raise CacheError(f"{cache_name}: manifest cache_index values do not cover 0..{n - 1} "
                             "exactly once")
        if want_mask and datasets_root is None:
            raise CacheError("want_mask requires datasets_root")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        import numpy as np
        r = self.rows[k]
        x = to_model_input(np.asarray(self.img[int(r["cache_index"])]))
        item = {"x": x, "record": dict(r), "prep_version": PREP_VERSION,
                "mask": None, "mask_meta": None, "mask_interp_version": None}
        if self.want_mask:
            mp = r.get("mask_path", "")
            if not mp:
                raise CacheError(f"{r['record_uid']}: no mask_path")
            les, mm = DC.interpret_mask(os.path.join(self.datasets_root, *mp.split("/")))
            item.update(mask=mask_to_input_grid(les), mask_meta=mm,
                        mask_interp_version=mm["version"])
        return item
