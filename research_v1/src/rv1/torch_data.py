"""Manifest-backed Torch Dataset / DataLoader integration (Stage 2C1). No model.

Design
------
* The FULL cache mapping is validated first (preprocess.EvalReader over all manifest
  rows of that cache); only then is a subset of record_uids selected. The subset
  need not cover the cache.
* Cache access: a read-only np.load(mmap_mode="r") memory map, opened lazily in the
  process that reads (parent or worker). __getstate__ drops the map, so only paths,
  small row dicts and scalars are pickled to Windows spawn workers; no cache array
  is pickled or loaded into RAM.
* Modes: "eval" -> preprocess.to_model_input (independent of seed/epoch);
  "train" -> augment.augment(recipe, train_seed, epoch, image_uid).
* Epoch: an attribute of the dataset. DataLoader workers receive a pickled COPY of
  the dataset when an iterator is created. This module therefore uses
  persistent_workers=False and make_epoch_loader() builds a new DataLoader (hence new
  workers) per epoch AFTER setting dataset.epoch, so the epoch reaches every worker.
* Shuffling: an explicit torch.Generator seeded with
  stable_seed(sampler_seed, "sampler", epoch) (< 2**63) per epoch. Shuffle order is a
  sampler property; per-record content is an augmentation property keyed by
  (train_seed, epoch, image_uid). They are independent.
* Items: {"x": float32 tensor (3,224,224), "label": int64 tensor (), "record": dict
  (unchanged copy), "mask": bool tensor (224,224) or None, "mask_flags": list}.
  Masks are optional diagnostic metadata, never model input. collate() stacks x and
  labels and keeps records/masks/flags as lists aligned with the batch order.
"""
import os

from . import augment as AU, decode as DC, preprocess as PP
from .seeds import stable_seed


def _torch():
    import torch
    return torch


class ManifestDataset:
    """torch-compatible map-style dataset (implements __len__/__getitem__)."""

    def __init__(self, rows_all, cache_name, cache_dir, datasets_root, record_uids=None,
                 mode="eval", recipe="aug_std_v1", train_seed=None, epoch=0, want_mask=False):
        if mode not in ("eval", "train"):
            raise ValueError(f"mode must be 'eval' or 'train', got {mode!r}")
        if mode == "train" and not isinstance(train_seed, int):
            raise ValueError("train mode requires an int train_seed")
        full = PP.EvalReader(rows_all, cache_name, cache_dir)       # full-mapping validation
        by_uid = {r["record_uid"]: r for r in full.rows}
        uids = [r["record_uid"] for r in full.rows] if record_uids is None else list(record_uids)
        if len(set(uids)) != len(uids):
            raise ValueError("duplicate record_uid in subset")
        missing = [u for u in uids if u not in by_uid]
        if missing:
            raise ValueError(f"record_uid(s) not in cache {cache_name!r}: {missing[:5]}")
        self.rows = [dict(by_uid[u]) for u in uids]
        self.img_path = os.path.join(cache_dir, f"{cache_name}_img.npy")
        self.cache_name, self.datasets_root = cache_name, datasets_root
        self.mode, self.recipe, self.train_seed, self.epoch = mode, recipe, train_seed, epoch
        self.want_mask = want_mask
        self._img = None                         # opened lazily per process
        del full

    def __getstate__(self):
        s = dict(self.__dict__)
        s["_img"] = None                         # never pickle the memory map
        return s

    def _cache(self):
        if self._img is None:
            import numpy as np
            self._img = np.load(self.img_path, mmap_mode="r")
        return self._img

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import numpy as np
        torch = _torch()
        r = self.rows[i]
        img256 = np.asarray(self._cache()[int(r["cache_index"])])
        m256, flags = None, []
        if self.want_mask:
            les, mm = DC.interpret_mask(os.path.join(self.datasets_root, *r["mask_path"].split("/")))
            m256, flags = AU.mask_to_cache_grid(les), list(mm["flags"])
        if self.mode == "eval":
            out = AU.augment(img256, "aug_none", mask256=m256, mask_flags=flags)
        else:
            out = AU.augment(img256, self.recipe, self.train_seed, self.epoch, r["image_uid"],
                             m256, flags)
        return {"x": torch.from_numpy(out["x"]),
                "label": torch.tensor(int(r["label_mel"]), dtype=torch.int64),
                "record": dict(r),
                "mask": None if out["mask"] is None else torch.from_numpy(out["mask"]),
                "mask_flags": out["mask_flags"]}


def collate(items):
    torch = _torch()
    return {"x": torch.stack([it["x"] for it in items]),
            "label": torch.stack([it["label"] for it in items]),
            "record": [it["record"] for it in items],
            "mask": [it["mask"] for it in items],
            "mask_flags": [it["mask_flags"] for it in items]}


def make_epoch_loader(dataset, epoch, batch_size, shuffle, sampler_seed, num_workers=0):
    """New DataLoader for one epoch (persistent_workers=False, explicit generator)."""
    torch = _torch()
    dataset.epoch = epoch
    g = torch.Generator()
    g.manual_seed(stable_seed(sampler_seed, "sampler", epoch))
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=g,
        num_workers=num_workers, persistent_workers=False, drop_last=False,
        collate_fn=collate, pin_memory=False)
