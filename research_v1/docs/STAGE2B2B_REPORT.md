# Stage 2B2B Report — HAM Mask Interpretation

Date: 2026-09-27

## Status

| Aspect | Result |
|---|---|
| Successful decoding (from Stage 2B2A, re-confirmed here) | 10015 / 10015 masks |
| Deterministic interpretation under `mask_interp_v1` | **PASS**: 10015 interpreted, 0 rejected; 10013 `ok`, 2 `flagged` (all-foreground) |
| Visual review | **Artifact generated and visually reviewed by the agent** for the 20 selected records (4 pages). This was not an expert or clinical review. |
| Annotation validity | **Unresolved in general.** Visual agreement does not establish clinical annotation accuracy. The two all-white masks are unresolved for localisation. |
| Tests | 40 / 40 pass |
| Protected inputs | 0 changed, 0 removed |

No preprocessing, resizing, model loading or training was done.

## 1. Evidence for the convention

**Publisher source** (local copy: `DATASETS/ham10000/dataverse_meta.json`; dataset DOI https://doi.org/10.7910/DVN/DBW86T; file entry `HAM10000_segmentations_lesion_tschandl.zip`).

What the source establishes:
- A single dermatologist (P. Tschandl) created **binary** segmentation masks for all 10015 images.
- The masks were initialised with a segmentation network (Tschandl et al., *Computers in Biology and Medicine* 2019, doi:10.1016/j.compbiomed.2018.11.010).
- They were then verified, corrected or replaced in FIJI.
- The source asks users to cite Tschandl et al., *Nature Medicine* 2020 (https://www.nature.com/articles/s41591-020-0942-0).

What the source does **not** establish:
- It does **not** say which colour or value means lesion.
- It does **not** mention palette-format files.
- No other local documentation on this was found.

**Therefore the white-means-lesion convention is not guaranteed by the publisher.** It rests on the following evidence:

1. **Rendered values.** Every mask renders to exactly two values, {0, 255} (black and white), except two masks that are entirely white. This agrees with the publisher's description of the masks as binary.
2. **Foreground size distributions.** The white fraction has the same distribution in both formats (see §2).
   - This **supports** the convention but **does not prove it**. The Stage 2B2A report said this evidence "strongly suggests" the convention, which overstated it. It should be read as supporting evidence only.
3. **Visual review.** In all 18 sampled non-degenerate masks (12 P, 6 L), the white region lay on the visible lesion (§3).
4. **Independent reviewer's inspection** of ISIC_0024324 (P) gave the same result. That is supporting evidence, not validation of all 402 P masks.

Stage 2B2A mentioned a comparison with OpenCV (`cv2.imread` with its palette handling). **That comparison was never run and remains unverified.** OpenCV was not imported in this stage either.

## 2. Convention `mask_interp_v1` and full validation

**New function: `decode.interpret_mask`.** The raw inspection function `decode_mask` is unchanged. The new function works as follows:
1. Render the mask:
   - "L" masks: the stored value is used as the rendered value.
   - "P" masks: each index that occurs is mapped through **that file's own RGB palette**. Each colour must be exactly black (0,0,0) or white (255,255,255). A palette transparency entry is rejected.
   - Any other mode is rejected.
2. Accept only rendered values 0 and 255. Any other grey value raises an error; nothing is thresholded.
3. White = lesion, black = background.
4. Empty and all-white masks are flagged.

Masks are never inverted blindly, and source files are only read.

**Results for 10015 HAM masks** (`mask_interpretation.csv`, `summary.json`):

| Original mode | Masks | Rendered values | Notes |
|---|---|---|---|
| L | 9613 | {0,255}: 9611; {255} only: 2 | — |
| P | 402 | {0,255}: 402 | Every one of the 402 files has the same palette mapping: index 0 → white, index 255 → black. Each file's palette was read individually. |

**White (lesion) fraction by mode:**

| Mode | min | p5 | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|---|
| L (n = 9613) | 0.0039 | 0.039 | 0.120 | 0.223 | 0.375 | 0.639 | 1.0 |
| P (n = 402) | 0.0057 | 0.035 | 0.112 | 0.205 | 0.358 | 0.625 | 0.986 |

**Other checks:**
- No mask is empty.
- 0 masks were rejected: no unsupported colours, no transparency, no unsupported modes, no unexpected grey values.
- **Mask SHA-256.** Every mask's file hash is recorded in the new CSV, because the frozen manifest has no mask-content hash. After the audit, all 10015 were re-hashed and 0 differed.

## 3. Visual review

**Artifact:** `review_page_1.png` … `review_page_4.png`. Each row shows the image, the rendered mask, and a translucent overlay with a contour. The pages were made with matplotlib and Pillow (plotting only; no generative tools).

**How the sample was selected** (fixed and recorded in `review_selection.json`):
- Masks were split into tertiles of lesion fraction, separately for P masks and for L masks (the L group excludes the all-white masks).
- Within each tertile, records were ordered by `stable_seed("mask_review_v1", record_uid)`.
- The first 4 per tertile were taken for P masks and the first 2 per tertile for L masks.
- Both all-white masks were then added.

**Selected records:**

| Group | Records |
|---|---|
| P, small | ISIC_0024431, ISIC_0024360, ISIC_0024385, ISIC_0024396 |
| P, medium | ISIC_0024379, ISIC_0024635, ISIC_0024412, ISIC_0024444 |
| P, large | ISIC_0024692, ISIC_0024510, ISIC_0024644, ISIC_0024424 |
| L, small | ISIC_0034177, ISIC_0026230 |
| L, medium | ISIC_0024728, ISIC_0030578 |
| L, large | ISIC_0032079, ISIC_0029217 |
| All-white | ISIC_0026042, ISIC_0029819 |

**What I observed** (non-expert visual check):
- **No sampled P mask looked inverted.** In every one of the 18 non-degenerate rows, the white region covers the visibly pigmented or abnormal area, and the background is black.
- ISIC_0024385: the image shows two pigmented spots, and only the lower one is masked. This may be an annotation choice; I cannot judge whether it is correct.
- ISIC_0024424 (fraction 0.986) and ISIC_0024644 (fraction 0.709): the lesion appears to fill most of the field. The masks are plausible, but I did not judge them clinically.

## 4. All-white masks

| Record | Label | Diagnosis | Split | Mode | Observation |
|---|---|---|---|---|---|
| ISIC_0026042 | 0 (non-melanoma) | bkl | val | L | The abnormal-looking tissue appears to fill the frame |
| ISIC_0029819 | 1 (melanoma) | mel | train | L | Dark lesion over most of the frame; skin-like margins may be visible at the edges |

Both images **remain in the classification cohorts**, and both masks are unchanged.

**Localisation meaning is UNRESOLVED.** An all-white mask may mean "the lesion fills the field of view" or may be an annotation limitation. My visual check cannot decide between these.

**Proposed rule for later localisation analysis (not applied now; no localisation metrics were computed):**
1. **Primary analysis:** exclude masks flagged `all_foreground` from localisation averages. Report the number excluded and the record IDs (currently 2: one in val, one in train).
2. **Sensitivity analysis:** include them. Report both results.

Note: neither mask is in HAM *test*, the split used for the simulated-shift localisation arm. So under the current protocol they affect no planned test-set localisation metric. The rule is still needed if val or train masks are ever used.

## 5. Tests

```bash
set PYTHONDONTWRITEBYTECODE=1
python -B -m unittest discover -s research_v1/tests -p "test_*.py" -v
```

**Result: `Ran 40 tests in 4.472s — OK`.** This includes the 33 earlier tests, re-run, and 7 new tests in `test_masks.py`.

The new tests use synthetic masks with a known lesion rectangle:
1. An ordinary black-and-white L mask.
2. A reversed-palette P mask (HAM-like: 0 → white, 255 → black).
3. An ordinary-palette P mask.
4. Equivalent L and P masks give identical boolean arrays.
5. These are rejected: a grey value, a non-black/white palette colour, palette transparency, and RGB and LA modes.
6. Empty and all-white masks are flagged, in both L and P format.
7. Original files are byte-unchanged after interpretation, and `decode_mask` raw behaviour is unchanged.

The `stable_seed` encoding is unchanged.

## 6. Outputs and provenance

| Path | Content |
|---|---|
| `research_v1/src/rv1/decode.py` | Adds `MASK_INTERP_VERSION = "mask_interp_v1"`, `MaskInterpretationError` and `interpret_mask`. Existing functions are unchanged. |
| `research_v1/src/rv1/config.py` | Adds the purpose `mask_audit`. Earlier configs and hashes are unchanged (tested). |
| `research_v1/src/rv1/mask_audit.py` (new) | Audit script and review sheets |
| `research_v1/tests/test_masks.py` (new) | Tests |
| `research_v1/configs/mask_audit_v1.json` (new) | Supporting inputs, all recorded with their SHA-256: `manifest_card`, `DATASETS/ham10000/dataverse_meta.json`, and the final Stage 2B2A `decode_audit.csv` and `summary.json` |
| **`research_v1/runs/mask_audit/20260927T113537Z_c049da73_88aea4ea/`** | `mask_interpretation.csv`, `summary.json`, `review_selection.json`, `review_page_1..4.png`, `receipt.json` + `.sha256`. The receipt records the input and source hashes and the output hashes. **`verify_receipt` → OK** |
| `research_v1/runs/stage2b2b/` | Integrity snapshots before and after, and the comparison |

**Runtime** (measured machine time, not human effort): audit 26.0 s; tests 4.5 s.

**Integrity**
- 185 files were checked before and 186 after. The coverage is the same as before, plus `research_v1/runs/data_audit` and `research_v1/runs/stage2b2a`.
- **0 changed, 0 removed.** The only file added is `research_v1/configs/mask_audit_v1.json`, which was expected.
- The cache `.npy` size and modification times are unchanged.
- The mask files were verified unchanged by content hash (§2).
- Earlier receipts were not rewritten. They may now report source-hash differences for `decode.py` and `config.py`, which were legitimately extended in this stage.

## 7. Unresolved issues

1. **Annotation validity.** Visual agreement on 18 samples supports `mask_interp_v1`. It does not validate every mask, and it says nothing about clinical annotation accuracy.
2. **All-white masks.** Localisation meaning is unresolved. The proposed handling rule is in §4.
3. **The OpenCV comparison was never executed**, and the OpenCV distribution conflict is still open.
4. **Earlier items still open:** RGBA handling in the final preprocessing recipe; documentation of how identifiers are constructed; the Secondary B endpoint for adapt_val (not estimable); and the primary seed-0 model policy.

**Stage 2B2B is complete.**
