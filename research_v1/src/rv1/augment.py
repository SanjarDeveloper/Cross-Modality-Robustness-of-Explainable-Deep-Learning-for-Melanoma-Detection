"""Deterministic training augmentation (Stage 2B2D). No model, no torch.

Recipes (Albumentations 2.0.8, the version installed both globally and in
research_v1/.venv_prep; every default is resolved explicitly below)
----------------------------------------------------------------------------
Input to every recipe: the legacy 256x256 RGB uint8 cache row (see preprocess.py)
and, optionally, a HAM mask on the 256 grid built exactly like preprocess.py:
mask_interp_v1 -> uint8 {0,1} -> cv2.resize((256,256), INTER_NEAREST_EXACT).

aug_none
  Identical to evaluation preprocessing: preprocess.to_model_input(img256) and
  preprocess.mask_to_input_grid semantics (256 -> 224, INTER_NEAREST_EXACT).
  Seed and epoch are ignored.

aug_std_v1  (parameter list of the legacy train_aug in code/data.py:LesionDataset._build)
  1. RandomResizedCrop(size=(224,224), scale=(0.7,1.0), ratio=(0.75,1.3333333333333333),
     interpolation=INTER_LINEAR, mask_interpolation=INTER_NEAREST_EXACT
     [legacy default was INTER_NEAREST; changed to the centre-aligned variant used
     for evaluation masks], p=1.0)
  2. HorizontalFlip(p=0.5)
  3. VerticalFlip(p=0.5)
  4. ShiftScaleRotate(shift_limit=0.05 -> (-0.05,0.05) in x and y,
     scale_limit=0.1 -> scale factor (0.9,1.1), aspect ratio kept,
     rotate_limit=20 -> (-20,20) degrees, interpolation=INTER_LINEAR,
     border_mode=BORDER_CONSTANT (0), fill=0 (black), fill_mask=0,
     mask_interpolation=INTER_NEAREST, rotate_method="largest_box", p=0.7)
     [Albumentations 2.x implements this as a special case of Affine and emits
     a deprecation warning; kept for fidelity to the legacy parameter list.]
  5. ColorJitter(brightness=(0.8,1.2), contrast=(0.8,1.2), saturation=(0.8,1.2),
     hue=(-0.05,0.05), p=0.5)  -- image only (Albumentations applies colour
     transforms to images, never to masks); the order of the four sub-operations
     is itself randomised by the library from the seeded generator.
  Then: float32 / 255 -> (x - MEAN) / STD -> CHW float32 (3,224,224), same as
  preprocess.to_model_input's normalisation step.

aug_heavy_v1  (Stage 9D, Addendum K: legacy code/train.py:_heavy_tf, order and explicit parameters kept;
  every default resolved for Albumentations 2.0.8; frozen after the standard-model results)
  1. RandomResizedCrop(size=(224,224), scale=(0.5,1.0), ratio=(0.75,1.3333333333333333),
     interpolation=INTER_LINEAR, mask_interpolation=INTER_NEAREST_EXACT, p=1.0)
  2. HorizontalFlip(p=0.5)   3. VerticalFlip(p=0.5)
  4. ShiftScaleRotate(shift_limit=0.1, scale_limit=0.25 -> factor (0.75,1.25), rotate_limit=45,
     interpolation=INTER_LINEAR, border_mode=BORDER_CONSTANT, fill=0, fill_mask=0,
     mask_interpolation=INTER_NEAREST, rotate_method="largest_box", p=0.8)
  5. ColorJitter(brightness=(0.6,1.4), contrast=(0.6,1.4), saturation=(0.6,1.4), hue=(-0.15,0.15), p=0.8)
  6. OneOf([GaussianBlur(blur_limit=(3,9), sigma_limit=(0.5,3.0)),
            MotionBlur(blur_limit=(3,9), allow_shifted=True, angle_range=(0,360), direction_range=(-1,1)),
            Defocus(radius=(3,8), alias_blur=(0.1,0.5))], p=0.5)   children p=0.5 each -> equal weights 1/3
  7. OneOf([GaussNoise(std_range=(0.2,0.44), mean_range=(0,0), per_channel=True, noise_scale_factor=1),
            ImageCompression(compression_type="jpeg", quality_range=(30,90))], p=0.4)   weights 1/2 each
  8. RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3),
     brightness_by_max=True, ensure_safe_range=False, p=0.6)
  Photometric transforms (5-8) never modify masks. Then the same normalisation as aug_std_v1.
  OneOf: activation from its own py_random, child choice from its own generator; a selected child is
  applied with force_apply=True (child p is only a selection weight).

Seeding (stateless)
  aug_seed = seeds.stable_seed(train_seed, epoch, image_uid)   in [0, 2**63-1]
  A NEW Compose is built for every sample. Its own generator and each child's
  generator are seeded with DISTINCT sub-seeds derived from aug_seed (see
  _seed_pipeline; nested OneOf children get their own sub-seeds too, because
  OneOf.set_random_seed propagates ONE seed to all children; Albumentations' own Compose(seed=...) seeds every child
  identically, which correlates their random decisions). The integers (< 2**63)
  are passed unchanged to numpy.random.default_rng and random.Random, which accept
  them; no range conversion is needed. No transform object is shared between
  samples, so results do not depend on access order, previous samples, or workers.
  No global RNG and no Python hash() is used.
"""
from . import preprocess as PP
from .seeds import stable_seed

RECIPES = ("aug_none", "aug_std_v1", "aug_heavy_v1")
DEFERRED = ()


def _std_pipeline(seed):
    import warnings
    import albumentations as A
    import cv2
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="ShiftScaleRotate is a special case")
        pipe = A.Compose([
            A.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0),
                                ratio=(0.75, 1.3333333333333333),
                                interpolation=cv2.INTER_LINEAR,
                                mask_interpolation=cv2.INTER_NEAREST_EXACT, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20,
                               interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                               fill=0, fill_mask=0, mask_interpolation=cv2.INTER_NEAREST,
                               rotate_method="largest_box", p=0.7),
            A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2),
                          hue=(-0.05, 0.05), p=0.5),
        ], seed=None, strict=True)
    _seed_pipeline(pipe, seed)
    return pipe


def _heavy_pipeline(seed):
    import warnings
    import albumentations as A
    import cv2
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="ShiftScaleRotate is a special case")
        pipe = A.Compose([
            A.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST_EXACT, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.25, rotate_limit=45,
                               interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                               fill=0, fill_mask=0, mask_interpolation=cv2.INTER_NEAREST,
                               rotate_method="largest_box", p=0.8),
            A.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4),
                          hue=(-0.15, 0.15), p=0.8),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 9), sigma_limit=(0.5, 3.0), p=0.5),
                     A.MotionBlur(blur_limit=(3, 9), allow_shifted=True, angle_range=(0, 360),
                                  direction_range=(-1.0, 1.0), p=0.5),
                     A.Defocus(radius=(3, 8), alias_blur=(0.1, 0.5), p=0.5)], p=0.5),
            A.OneOf([A.GaussNoise(std_range=(0.2, 0.44), mean_range=(0.0, 0.0), per_channel=True,
                                  noise_scale_factor=1, p=0.5),
                     A.ImageCompression(compression_type="jpeg", quality_range=(30, 90), p=0.5)], p=0.4),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3),
                                       brightness_by_max=True, ensure_safe_range=False, p=0.6),
        ], seed=None, strict=True)
    _seed_pipeline(pipe, seed)
    return pipe


def _seed_pipeline(pipe, seed):
    """Give the Compose and EACH child transform its own derived generator.

    Compose(seed=s) / Compose.set_random_seed(s) in Albumentations 2.0.8 seed every
    child with the SAME integer, so each transform's first draws coincide and their
    apply/skip decisions become perfectly correlated (observed in Stage 2B2D: only 3
    of 16 flip/affine/colour combinations occurred over 400 seeds). Therefore:
      compose  : stable_seed(seed, "albumentations_compose")
      child i  : stable_seed(seed, "albumentations_transform", i)
    """
    import random
    import numpy as np
    from albumentations.core.composition import BaseCompose
    cs = stable_seed(seed, "albumentations_compose")
    pipe.seed = None
    pipe.random_generator = np.random.default_rng(cs)
    pipe.py_random = random.Random(cs)
    for i, t in enumerate(pipe.transforms):
        t.set_random_seed(stable_seed(seed, "albumentations_transform", i))
        if not isinstance(t, BaseCompose):
            continue
        for j, child in enumerate(t.transforms):                              # nested OneOf: distinct streams
            child.set_random_seed(stable_seed(seed, "albumentations_transform", i, "child", j))


def mask_to_cache_grid(lesion):
    """bool source-resolution mask -> uint8 {0,1} (256,256), as in preprocess.py."""
    import cv2
    import numpy as np
    return cv2.resize(lesion.astype(np.uint8), (PP.CACHE_SIZE, PP.CACHE_SIZE),
                      interpolation=cv2.INTER_NEAREST_EXACT)


def _normalise(img224):
    import numpy as np
    x = img224.astype(np.float32) / np.float32(255.0)
    x = (x - np.asarray(PP.MEAN, np.float32)) / np.asarray(PP.STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)


def augment(img256, recipe, train_seed=None, epoch=None, image_uid=None, mask256=None,
            mask_flags=()):
    """Return dict(x, mask, mask_flags, recipe, aug_seed).

    img256: uint8 RGB (256,256,3) cache row. mask256: optional uint8 {0,1} (256,256).
    Labels / record metadata are not touched here; callers keep them unchanged.
    """
    import cv2
    import numpy as np
    if recipe in DEFERRED:
        raise NotImplementedError(f"{recipe} is deferred (not specified by the protocol)")
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}")
    if img256.dtype != np.uint8 or img256.shape != (256, 256, 3):
        raise PP.CacheError(f"expected uint8 (256,256,3), got {img256.dtype} {img256.shape}")
    if mask256 is not None and (mask256.shape != (256, 256) or
                                not set(np.unique(mask256).tolist()) <= {0, 1}):
        raise ValueError("mask256 must be a {0,1} (256,256) array")
    flags = list(mask_flags)
    if recipe == "aug_none":
        m = None
        if mask256 is not None:
            m = cv2.resize(mask256.astype(np.uint8), (224, 224),
                           interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
        return {"x": PP.to_model_input(img256), "mask": m, "mask_flags": flags,
                "recipe": recipe, "aug_seed": None}
    if not isinstance(train_seed, int) or not isinstance(epoch, int) or not image_uid:
        raise ValueError(f"{recipe} requires int train_seed, int epoch and image_uid")
    seed = stable_seed(train_seed, epoch, image_uid)
    pipe = _std_pipeline(seed) if recipe == "aug_std_v1" else _heavy_pipeline(seed)
    kw = {"image": np.ascontiguousarray(img256)}
    if mask256 is not None:
        kw["mask"] = np.ascontiguousarray(mask256.astype(np.uint8))
    out = pipe(**kw)
    m = None
    if mask256 is not None:
        m = out["mask"].astype(bool)
        if mask256.any() and not m.any():
            flags.append("empty_after_augmentation")
    return {"x": _normalise(out["image"]), "mask": m, "mask_flags": flags,
            "recipe": recipe, "aug_seed": seed}


def augment_record(record, img256, recipe, train_seed=None, epoch=None, mask256=None,
                   mask_flags=()):
    """augment() plus an unchanged COPY of the manifest record (labels, split, role, IDs).

    image_uid (not record_uid) keys the seed, so two reference rows of one physical
    file receive the same augmentation for the same (train_seed, epoch).
    """
    out = augment(img256, recipe, train_seed, epoch, record.get("image_uid"), mask256,
                  mask_flags)
    out.update(record=dict(record), train_seed=train_seed, epoch=epoch,
               prep_version=PP.PREP_VERSION)
    return out
