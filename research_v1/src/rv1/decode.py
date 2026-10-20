"""Manifest-backed record reading and explicit image/mask decoding (Stage 2B2A).

Scope: read frozen manifest CSVs, validate them, and decode image/mask files with
Pillow. NO resizing, cropping, rotation, normalisation, augmentation, colour
correction or tensor construction happens here. OpenCV is not imported.

Decoding policy
---------------
* Image.open() then im.load(): pixels are decoded explicitly (lazy open alone
  does not decode). Pillow's default ImageFile.LOAD_TRUNCATED_IMAGES=False is
  kept, so truncated files raise instead of returning partial pixels.
* Original size (width, height), mode and format are recorded before any change.
* RGB: if the decoded mode is not "RGB", im.convert("RGB") is applied explicitly
  and the original mode is recorded in `converted_from` (e.g. RGBA drops alpha,
  CMYK/P/L are mapped by Pillow's standard conversion).
* EXIF orientation (tag 274) is recorded when present. It is NOT applied: the
  pixels are returned in stored orientation.
* Masks are decoded in their stored mode and are not converted, binarised or
  resized. Observed convention for HAM10000 segmentations (checked, not taken
  from the publisher): 8-bit single channel ("L") with values {0, 255};
  255 (white) = lesion. 402 HAM masks are palette ("P") PNGs whose raw index 0
  renders white and index 255 renders black, i.e. raw indices are INVERTED
  relative to their appearance. Therefore foreground_fraction is computed ONLY
  for mode "L"; for other modes the raw values and the palette entries of the
  present indices are recorded and the mask is flagged, not interpreted.
  decode_mask stays a RAW inspection function; semantic interpretation is done
  only by interpret_mask (versioned, mask_interp_v1, below).

Importing this module does not open files and imports numpy/PIL lazily.
"""
import csv
import os
import posixpath

REQUIRED_COLUMNS = ("record_uid", "dataset", "image_uid", "rel_path", "label_mel",
                    "cluster_id")
DATASETS = ("ham10000", "derm7pt", "pad_ufes_20", "ddi")
EXIF_ORIENTATION_TAG = 274


class ManifestError(ValueError):
    pass


class DecodeError(Exception):
    pass


def _check_rel(p):
    if not p or os.path.isabs(p) or p.startswith(("/", "\\")):
        return False
    n = posixpath.normpath(p.replace("\\", "/"))
    return not (n == ".." or n.startswith("../"))


def read_manifest(path, dataset):
    """Return (rows, columns). rows keep CSV order and every original field (str).

    Validates required columns, dataset value, unique record_uid, label in {0,1},
    relative in-root rel_path (and mask_path when non-empty). File existence is
    NOT checked here; decoding reports missing files as failures.
    Duplicate image_uid values are allowed (Derm7pt same-file pairs).
    """
    if dataset not in DATASETS:
        raise ManifestError(f"unknown dataset {dataset!r}")
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        rows = list(rd)
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise ManifestError(f"{path}: missing required column(s) {missing}")
    seen = set()
    for i, r in enumerate(rows):
        where = f"{path} row {i + 1} ({r.get('record_uid')!r})"
        if not r["record_uid"]:
            raise ManifestError(f"{where}: empty record_uid")
        if r["record_uid"] in seen:
            raise ManifestError(f"{where}: duplicate record_uid")
        seen.add(r["record_uid"])
        if r["dataset"] != dataset:
            raise ManifestError(f"{where}: dataset {r['dataset']!r} != {dataset!r}")
        if r["label_mel"] not in ("0", "1"):
            raise ManifestError(f"{where}: unsupported label_mel {r['label_mel']!r}")
        if not _check_rel(r["rel_path"]):
            raise ManifestError(f"{where}: invalid rel_path {r['rel_path']!r}")
        mp = r.get("mask_path", "")
        if mp and not _check_rel(mp):
            raise ManifestError(f"{where}: invalid mask_path {mp!r}")
    return rows, cols


def _open(path):
    from PIL import Image
    if not os.path.isfile(path):
        raise DecodeError("file_missing")
    try:
        im = Image.open(path)
        im.load()
    except Exception as e:  # noqa: BLE001 - reason is recorded, not swallowed
        raise DecodeError(f"decode_failed: {type(e).__name__}: {e}") from None
    return im


def decode_image(path):
    """Decode to an RGB uint8 array (H, W, 3) in stored orientation + metadata."""
    import numpy as np
    im = _open(path)
    meta = {"width": im.size[0], "height": im.size[1], "mode": im.mode,
            "format": im.format, "exif_orientation": None, "converted_from": None}
    try:
        o = im.getexif().get(EXIF_ORIENTATION_TAG)
        meta["exif_orientation"] = int(o) if o is not None else None
    except Exception as e:  # noqa: BLE001
        meta["exif_orientation"] = f"unreadable: {type(e).__name__}"
    if im.mode != "RGB":
        meta["converted_from"] = im.mode
        im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)
    if arr.shape != (meta["height"], meta["width"], 3):
        raise DecodeError(f"unexpected_array_shape {arr.shape}")
    return arr, meta


def decode_mask(path):
    """Decode a mask in its stored mode (no conversion) + metadata and value summary."""
    import numpy as np
    im = _open(path)
    arr = np.asarray(im)
    vals, counts = np.unique(arr, return_counts=True) if arr.ndim == 2 else (None, None)
    meta = {"width": im.size[0], "height": im.size[1], "mode": im.mode, "format": im.format,
            "ndim": int(arr.ndim)}
    if vals is not None:
        meta["values"] = [int(v) for v in vals]
        meta["foreground_fraction"] = (float((arr == 255).mean()) if im.mode == "L"
                                       else None)
    if im.mode == "P" and vals is not None:
        pal = im.getpalette() or []
        meta["palette_rgb_of_values"] = {int(v): tuple(pal[3 * int(v):3 * int(v) + 3])
                                         for v in vals}
    return arr, meta


def mask_flags(mask_meta, image_meta, expected_values=(0, 255), expected_mode="L"):
    """(critical, warnings) for one mask versus its image."""
    critical, warn = [], []
    if (mask_meta["width"], mask_meta["height"]) != (image_meta["width"], image_meta["height"]):
        critical.append("mask_image_size_mismatch")
    if mask_meta["mode"] != expected_mode:
        warn.append(f"unexpected_mode:{mask_meta['mode']}")
    vals = mask_meta.get("values")
    if vals is None:
        warn.append("non_2d_mask")
    else:
        if set(vals) - set(expected_values):
            warn.append("unexpected_values")
        if mask_meta["mode"] == expected_mode:   # value semantics defined for "L" only
            if 255 not in vals:
                warn.append("empty_foreground")
            if vals == [255]:
                warn.append("all_foreground")
    return critical, warn


def iter_records(rows, root):
    """Yield (row, absolute image path, decoded array, meta) per reference row.

    Every reference row is decoded and yielded separately, even when two rows
    share one physical file, so each item stays linked to its own record_uid.
    `root` is the directory rel_path is relative to (DATASETS/). Decoding errors
    are yielded as (row, path, None, {"error": reason}) rather than raised.
    """
    for r in rows:
        p = os.path.join(root, *r["rel_path"].split("/"))
        try:
            arr, meta = decode_image(p)
        except DecodeError as e:
            yield r, p, None, {"error": str(e)}
            continue
        yield r, p, arr, meta


# --------------------------------------------------------------------------
# Mask interpretation (Stage 2B2B). Separate from decode_mask (raw inspection).
MASK_INTERP_VERSION = "mask_interp_v1"


class MaskInterpretationError(Exception):
    pass


def interpret_mask(path):
    """Binary lesion mask under convention mask_interp_v1.

    Rule: the mask is RENDERED to 8-bit values, then validated; only the rendered
    values 0 (black = background) and 255 (white = lesion) are accepted.
      * mode "L": rendered value = stored value.
      * mode "P": each present index is mapped through THIS file's own RGB
        palette; the colour must be exactly (0,0,0) or (255,255,255). No palette
        is assumed or blindly inverted.
      * a transparency entry (im.info["transparency"]) is rejected for any mode
        (since Stage 2B2C; previously checked for P only).
      * any other mode is rejected.
    Unexpected grey values or colours raise MaskInterpretationError; nothing is
    thresholded. The source file is only read.
    Returns (lesion: bool ndarray HxW, meta).
    """
    import numpy as np
    im = _open(path)
    meta = {"version": MASK_INTERP_VERSION, "original_mode": im.mode,
            "width": im.size[0], "height": im.size[1], "palette_map": None}
    raw = np.asarray(im)
    if "transparency" in im.info:          # rejected for L and P alike
        raise MaskInterpretationError("transparency_present")
    if im.mode == "L":
        rendered = raw
    elif im.mode == "P":
        if im.palette is None or im.palette.mode != "RGB":
            raise MaskInterpretationError(
                f"unsupported_palette_mode:{getattr(im.palette, 'mode', None)}")
        pal = im.getpalette("RGB") or []
        lut, pmap = np.zeros(256, np.uint8), {}
        for v in np.unique(raw).tolist():
            rgb = tuple(pal[3 * v:3 * v + 3])
            if rgb == (255, 255, 255):
                lut[v] = 255
            elif rgb == (0, 0, 0):
                lut[v] = 0
            else:
                raise MaskInterpretationError(f"unsupported_palette_colour:index{v}={rgb}")
            pmap[int(v)] = list(rgb)
        meta["palette_map"] = pmap
        rendered = lut[raw]
    else:
        raise MaskInterpretationError(f"unsupported_mode:{im.mode}")
    vals = sorted(int(v) for v in np.unique(rendered))
    bad = [v for v in vals if v not in (0, 255)]
    if bad:
        raise MaskInterpretationError(f"unsupported_rendered_values:{bad[:10]}")
    lesion = rendered == 255
    frac = float(lesion.mean())
    flags = []
    if frac == 0.0:
        flags.append("empty_foreground")
    if frac == 1.0:
        flags.append("all_foreground")
    meta.update(rendered_values=vals, foreground_fraction=frac, flags=flags)
    return lesion, meta
