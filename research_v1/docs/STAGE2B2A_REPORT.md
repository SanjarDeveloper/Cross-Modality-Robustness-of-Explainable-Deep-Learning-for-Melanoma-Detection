# Stage 2B2A Report — Manifest-Backed Image and Mask Decoding

Date: 2026-09-27

## Status: **PASS**

- The final audit, run 2, found 0 critical problems.
- All 33 tests passed.
- The receipt verifies.
- No protected input was changed.

This result does **not** mean the data are ready for modelling. The mask convention (§5) still has to be resolved in Stage 2B2B.

## 1. Files created and changed

| Path | Change |
|---|---|
| `research_v1/src/rv1/decode.py` (new) | Manifest reader with validation, plus explicit Pillow decoding of images and masks |
| `research_v1/src/rv1/decode_audit.py` (new) | Full streaming decode audit. Writes the audit CSV, a summary and a receipt |
| `research_v1/src/rv1/config.py` | Adds the purpose `data_audit` and an optional `supporting_inputs` field ({name: relative path}, validated like other paths). Earlier configs load unchanged and keep the same hashes (tested) |
| `research_v1/src/rv1/provenance.py` | Receipts record `supporting_inputs` as {path, sha256}. No sidecar is required for these. `verify_receipt` checks the recorded hash and that the entries match the embedded config. Older receipts without the field still verify through this path |
| `research_v1/tests/test_decode.py` (new) | 10 focused tests (§3) |
| `research_v1/configs/data_audit_v1.json` (new) | Purpose `data_audit`. It lists the 5 frozen manifest CSVs (which have sidecars) and `manifest_card_v1.json` as a supporting input (bytes unchanged, hash recorded). The environment inventory is included. The scientific seeds and versions are unchanged (null where not applicable) |
| `research_v1/runs/data_audit/20260927T110835Z_7be7a6a8_c91ef52b/` | **Run 1. Superseded.** Its `mask_foreground_fraction` values for the 402 palette masks are raw-index based and therefore inverted (§5). Kept unchanged as a record. Do not use |
| **`research_v1/runs/data_audit/20260927T111157Z_7be7a6a8_5f26c521/`** | **Run 2. Final.** Contains `decode_audit.csv`, `summary.json`, `receipt.json` and `receipt.json.sha256` |
| `research_v1/runs/stage2b2a/` | Integrity snapshot script, snapshots before and after, and the comparison |
| `research_v1/docs/STAGE2B2A_REPORT.md` | This report |

## 2. Decoder and decoding policy

**Decoder**
- Decoding used Pillow **12.3.0**, which was already installed.
- **OpenCV was not imported** at any point in this stage.
- The two conflicting OpenCV distributions are still an open issue for preprocessing.

**Images**
- `Image.open` followed by an explicit `im.load()`.
- `LOAD_TRUNCATED_IMAGES = False` (the default), so truncated files raise an error.
- `MAX_IMAGE_PIXELS` stays at its default of 89,478,485.
- For every image the audit records the stored width, height, mode and format.
- The EXIF orientation tag (274) is recorded but **not applied**.
- A non-RGB image is converted with an explicit `convert("RGB")`, and the original mode is recorded in `converted_from`.
- Nothing is resized, cropped, rotated or colour-corrected, and no tensors are built.

**Masks**
- Masks are decoded in their stored mode, without conversion.
- The audit records each mask's values and, for palette masks, the palette colour of each value that occurs.
- Mask size is compared with the size of its image.

**Manifest reader** (`read_manifest`)
- Requires the columns `record_uid`, `dataset`, `image_uid`, `rel_path`, `label_mel` and `cluster_id`.
- Rejects any of the following:
  - duplicate or empty `record_uid`;
  - a wrong `dataset` value;
  - labels outside {0, 1};
  - `rel_path` or `mask_path` values that are absolute or escape the root.
- Keeps the CSV row order and every field.
- Allows the same `image_uid` to appear more than once (the Derm7pt same-file pairs).
- `iter_records` decodes each reference row separately and returns it together with its own manifest row.

## 3. Tests

```bash
set PYTHONDONTWRITEBYTECODE=1
python -B -m unittest discover -s research_v1/tests -p "test_*.py" -v
```

**Result: `Ran 33 tests in 4.920s — OK`.** This covers the 23 foundation tests, re-run and still passing, and the 10 new tests.

New tests, all using temporary synthetic fixtures:

1. RGB pixels are read exactly, and record identity, row order and `cluster_id` are kept. An RGBA image is converted to RGB explicitly and the conversion is recorded.
2. A missing file and a corrupt file are each reported with a reason.
3. A duplicate `record_uid` is rejected.
4. Other invalid manifests are rejected: an invalid label, a `rel_path` that escapes the root, an empty `record_uid`, and a wrong dataset.
5. Two paired reference rows that point to one physical file both decode, and each stays linked to its own record.
6. A valid "L" mask is accepted.
7. A missing mask file is detected, and so are a mask/image size mismatch and an empty foreground.
8. A palette mask is not interpreted: its fraction is `None`, the palette is recorded and the mask is flagged.
9. A change to a supporting input is detected by `verify_receipt`.
10. The earlier configs still validate, and their hashes are unchanged.

## 4. Audit results (run 2)

| Dataset | Image-reference rows | Unique physical files | Decoded OK | Modes / format |
|---|---|---|---|---|
| HAM10000 | 10015 | 10015 | 10015 | RGB JPEG |
| Derm7pt | 2022 | 2013 | 2013 | RGB JPEG |
| PAD-UFES-20 | 2298 | 2298 | 2298 | PNG: RGB 858, **RGBA 1440** |
| DDI | 656 | 656 | 656 | PNG: RGB 428, **RGBA 228** |
| **Total** | **14991** | **14982** | **14982** | — |

**Image checks**
- Every one of the 14,982 files has a SHA-256 that matches its manifest entry.
- No image is corrupt or unreadable.
- **RGBA:** all 1,668 RGBA images have a fully opaque alpha channel (alpha minimum = 255). This was checked with a separate read-only diagnostic. So dropping alpha during the explicit RGB conversion loses no image content.
- **EXIF orientation:** present only as value 1 (normal), in 228 DDI and 67 PAD files. There is no unusual orientation.
- **Dimensions:**
  - HAM: all 600×450.
  - Derm7pt: mostly 768×512 or 768×511, with a few other sizes.
  - PAD: square images of many sizes, from about 433 to 2268 px.
  - Full per-file values are in the audit CSV.

**HAM masks**

| Check | Result |
|---|---|
| Records with a non-empty `mask_path` | 10015 / 10015 |
| Mask files decoded | 10015 |
| Mask size equals image size | 10015 (no mismatch) |
| Mode "L", values {0, 255} | 9613 |
| **Mode "P" (palette)** | **402** (see §5) |
| All-foreground (value 255 only), "L" masks | **2**: `ISIC_0026042`, `ISIC_0029819`. Flagged, not changed |

**Runtime** (measured on this machine; this is not human effort):
- Run 2: 116.4 s. Run 1: 217.3 s, which included the first cold reads of the files.
- Peak working set about 186 MiB, measured with `psutil`.
- Tests: 4.9 s.

## 5. Mask-format convention finding (resolve in Stage 2B2B)

**What the 402 palette masks look like**
- Their raw index **0** has palette colour **white** (255, 255, 255).
- Their raw index **255** has palette colour **black** (0, 0, 0).
- So their raw index values are the reverse of what they look like.

**Read-only evidence of what they mean**

| Quantity | 5th / 50th / 95th percentile |
|---|---|
| Fraction of *rendered white* pixels in the P masks | 0.035 / 0.205 / 0.625 |
| Fraction of white pixels in the L masks | 0.039 / 0.223 / 0.639 |
| Fraction of *raw index 255* in the P masks | 0.375 / 0.795 / 0.965 |

The rendered-white fractions of the P masks match the L masks closely. This strongly suggests that **lesion = rendered white**, and that raw index 255 in the P masks means *background*.

**What this changes**
- Any code that reads raw mask values without applying the palette would invert these 402 masks.
- For comparison, legacy `cv2.imread(..., IMREAD_GRAYSCALE)` applies palettes and would render them correctly. That was not tested here.
- Run 1 computed foreground fractions from raw indices, so its values for these 402 masks are wrong. **Run 1 is superseded.**
- In run 2, `decode.py` computes foreground fractions only for mode "L". For other modes it records the raw values and their palette colours, flags the mask, and does not interpret it.

**Proposed convention for 2B2B (not yet adopted)**
- Interpret a mask by its rendered luminance: apply the palette, then treat value 255 (white) as lesion.
- Test it on these 402 files, together with an independent visual check of a sample.

## 6. Receipt and integrity

**Receipts**

| Receipt | Verification result |
|---|---|
| Run 2 (`…111157Z_7be7a6a8_5f26c521`) | `verify_receipt` → `[]` (OK) |
| Run 1 | Now reports source-hash differences for `decode.py` and `decode_audit.py`. These are the legitimate fixes above. It was not rewritten |
| The two earlier foundation-check receipts | Report source differences for `config.py` and `provenance.py`, which were legitimately extended in this stage. They were not rewritten |

The run 2 receipt records the following:
- the hashes of the manifests and their sidecars;
- the hash of the environment inventory;
- `supporting_inputs.manifest_card` with its recorded SHA-256;
- the source hashes;
- the decoder details (Pillow 12.3.0, `opencv_imported: false`);
- the hashes of `decode_audit.csv` and `summary.json`.

**Protected-input integrity** (`research_v1/runs/stage2b2a/integrity_comparison.json`)
- 171 files were hashed before and 172 after. The covered paths are the same as in Stage 2B1.1, plus `research_v1/runs/stage2b1_1`.
- **0 changed, 0 removed.** The only file added is `research_v1/configs/data_audit_v1.json`, which was expected.
- The cache `.npy` files have the same size and modification time as before.
- The dataset images themselves were re-hashed by the audit, and all 14,982 matched the manifest SHA-256.
- No `__pycache__` directory was created under `research_v1`.

## 7. Unresolved issues

1. **Palette mask convention** (§5). Must be decided and tested in 2B2B before any localisation or mask use.
2. **Two all-foreground "L" masks** (`ISIC_0026042`, `ISIC_0029819`). Their validity is unknown. They are flagged and not changed.
3. **OpenCV conflict**: two distributions are installed (4.11.0.86 and headless 5.0.0.93). This must be addressed before preprocessing.
4. **RGBA images**: 1,668 images have an opaque alpha channel, which is dropped explicitly during RGB conversion. The final preprocessing recipe must state this.
5. **Earlier carry-forward items still open**:
   - documentation of how identifiers are constructed;
   - the Secondary B threshold for adapt_val (9 melanomas), which is not estimable;
   - the policy for the primary model at seed 0 (legacy or new).

**Stage 2B2A is complete. No preprocessing, model loading or training was done.**
