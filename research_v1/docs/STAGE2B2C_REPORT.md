# Stage 2B2C Report — Deterministic Evaluation Preprocessing (prep_v1_cached256)

Date: 2026-09-27

## Status

| Aspect | Result |
|---|---|
| Software tests (preprocess, masks, decode) in the project venv | **27 / 27 pass** |
| Foundation tests in the global interpreter (they need torch) | **23 / 23 pass** |
| Real-data sample check (29 records, every dataset, all 7 caches, both mask formats) | **PASS** — cache reconstruction exact 29/29 |
| Visual alignment artifact | Generated and **visually reviewed by the agent**: 3 P-mode and 3 L-mode masks align with the lesion on the 224 input |
| Scientific / model validity | **Not established by this stage.** It shows only that inputs are built as documented. No model was loaded, and no augmentation, inference or training was done |
| Protected inputs | 0 changed, 0 removed |

## 1. The legacy cache, as inspected (not assumed)

The facts below come from `code/precache.py`, `code/data.py` and the cache files. All were read-only.

| Item | Finding |
|---|---|
| Decoder / channel order | `cv2.imread(path, IMREAD_COLOR)` (BGR; alpha dropped; OpenCV applies EXIF orientation, but every observed orientation is 1) → `cvtColor(BGR2RGB)`. **The cache is stored in RGB order.** |
| Resize | `cv2.resize((256,256), INTER_AREA)`: the full frame is stretched, with no crop or padding |
| Array | `cache/<name>_img.npy`, `uint8`, shape `(n, 256, 256, 3)`, values 0–255 |
| Masks (HAM only) | `cache/ham_*_mask.npy`, `uint8 (n,256,256)`, made with `cv2.imread(IMREAD_GRAYSCALE)` + `INTER_NEAREST`, values {0,255}. **Not used by v1** (see §3) |
| Row mapping | Cache row *i* = the legacy loader's record *i*. Each manifest row gives its own `cache_name` + `cache_index` |
| Metadata | `<name>_meta.json`: `n`, `size`=256, `labels`, `groups`, `has_mask`, `mask_ok`, `meta` |
| Legacy step from 256 to model input | `CachedDataset`: Albumentations `Resize(224,224)` (default `INTER_LINEAR`) → `astype(float32)/255` → `(x − MEAN)/STD` → HWC→CHW |

**Reconstruction check.** For all 29 sampled records, rebuilding the cache entry from the raw file with the documented steps (in the venv, OpenCV 4.11.0) produced a **byte-identical** match: maximum absolute difference 0. The sample includes RGBA-source PNGs from PAD and DDI.

This supports, but does not prove for every row, that the cache was built as documented with an equivalent OpenCV build.

**No contradiction with the planned recipe was found.**

## 2. Evaluation contract (`rv1/preprocess.py`, `PREP_VERSION = "prep_v1_cached256"`)

The image path, step by step:

1. Legacy cache row: `uint8` RGB `(256,256,3)`, opened with a read-only `np.load(mmap_mode="r")`.
2. `cv2.resize(dsize=(224,224), interpolation=cv2.INTER_LINEAR)`, called explicitly.
3. `float32 / 255.0`.
4. `(x − MEAN) / STD`, with MEAN = (0.485, 0.456, 0.406) and STD = (0.229, 0.224, 0.225) as float32 and channels in R, G, B order.
5. Transpose to CHW. The output is `float32`, shape `(3, 224, 224)`, C-contiguous, as a numpy array. No torch tensor is created.

The order of operations is the same as the legacy loader.

- **Diagnostic:** on all 29 samples, Albumentations 2.0.8 `Resize(224,224)` was byte-identical to the explicit cv2 call. Albumentations is not used in the v1 path.
- **raw→224 is not equivalent:** a direct raw→224 `INTER_AREA` input differs from cached256→224. The maximum absolute difference in normalised units was 0.12–1.06 across samples, and the mean was 0.006–0.046. This was a diagnostic only. It was not required to match, and no claim of equivalence is made.

**Reader.** `EvalReader(rows, cache_name, cache_dir, datasets_root, want_mask)` uses the frozen manifest rows (`read_manifest`). It fails with `CacheError` in any of these cases:

- the dtype or shape is wrong;
- the meta `n` or `size` does not match the array;
- a `cache_index` is invalid or out of range;
- a label or group differs from the cache meta;
- the indices do not cover `0..n−1` exactly once.

Each item returns:

- `x`;
- an unchanged copy of the record (`record_uid`, `image_uid`, `pair_id`, `cluster_id` kept separately);
- `prep_version`;
- the optional mask together with `mask_meta` (including the empty or all-foreground flags) and `mask_interp_version`.

No record is ever dropped.

## 3. Masks

- **Path:** source mask → `decode.interpret_mask` (`mask_interp_v1`) → `uint8 {0,1}` → `cv2.resize((256,256), INTER_NEAREST_EXACT)` → `cv2.resize((224,224), INTER_NEAREST_EXACT)` → boolean `(224,224)`. Masks are never normalised or bilinearly interpolated.
- **Geometry:** the mask goes through the same two full-frame stretches as the image (raw→256→224).
- **`INTER_NEAREST_EXACT` instead of the legacy `INTER_NEAREST`:** the EXACT variant samples at pixel centres, which matches the centre-aligned `INTER_AREA` and `INTER_LINEAR` used for the image. A test checks it exactly against an independently derived centre-index mapping, and to within 1 output pixel of the continuous geometry.
- **Diagnostic against the legacy cached masks** (`cv2` grayscale + `INTER_NEAREST`, binarised at >127): the v1 256-grid masks differ on 0.12–0.59% of pixels. This includes the 3 sampled palette (P) masks, which rules out a legacy inversion for those 3 files.

  This is the first *executed* OpenCV palette comparison. It covers only 3 P masks and is not a full verification. The Stage 2B2B statement that an OpenCV comparison had never been run was correct at that time.
- **Transparency fix:** `interpret_mask` now rejects a transparency entry for **L masks as well as P masks** (error `transparency_present`). The raw `decode_mask` behaviour is unchanged. A regression test uses an L PNG that really carries tRNS.

## 4. OpenCV environment (actual changes)

**Global inspection** (read-only, unchanged). There are actually **four** OpenCV distributions sharing one `cv2` package directory. Earlier reports said two; that was an under-count.

| Distribution | Version |
|---|---|
| opencv-python | 4.11.0.86 |
| opencv-python-headless | 5.0.0.93 |
| opencv-contrib-python | 5.0.0.93 |
| opencv-contrib-python-headless | 5.0.0.93 |

The imported `cv2` reports 4.11.0 (non-headless, non-contrib). Many packages depend on OpenCV (albumentations, albucore, grad-cam, mediapipe, ultralytics, and others). Nothing global was uninstalled or upgraded.

**New project-local venv:** `research_v1/.venv_prep`.

- Built from the recorded interpreter, Python 3.10.11.
- CPU-only packages were installed into it. Exactly **one** OpenCV distribution is present: `opencv-python-headless 4.11.0.86`, `cv2.__version__` = 4.11.0, headless = True.
- Also installed: numpy 2.2.6, Pillow 12.3.0, albumentations 2.0.8 and albucore 0.0.24 (both used only for the diagnostic), plus their dependencies.
- The full freeze is in `research_v1/runs/stage2b2c/venv_prep_freeze.txt`.
- No torch or other GPU packages were installed.

Version 4.11.0 was chosen because it is the OpenCV that the global interpreter actually loads. Headless and non-headless builds are not claimed to be byte-equivalent in general. For the sampled rows, however, the cache reconstruction was exact (§1).

## 5. Identifier construction (actual, frozen, not rewritten)

| Dataset | `record_uid` | `image_uid` | Uniqueness scope |
|---|---|---|---|
| HAM10000 | `ham10000:<image_id>` | same as `record_uid` | unique within the dataset; `image_id` comes from metadata |
| Derm7pt | `derm7pt:<pair_id>:<modality>` | `derm7pt:<relative path under DATASETS>` (path-based) | `record_uid` unique; `image_uid` is shared by the 9 same-file pairs |
| PAD-UFES-20 | `pad_ufes_20:<file basename>` | same as `record_uid` | unique within the dataset |
| DDI | `ddi:<file basename>` | same as `record_uid` | unique within the dataset |

- `pair_id` for Derm7pt is the `meta.csv` `case_id` when it is non-empty (27 cases), and `case<case_num>` otherwise.
- `cluster_id` is the lesion ID for HAM, the `pair_id` for Derm7pt, the `patient_id` for PAD, and the `DDI_ID` for DDI.
- The dataset prefix makes every ID unique across datasets.

## 6. Tests

**In the venv:**

```bash
research_v1\.venv_prep\Scripts\python.exe -B -m unittest discover -s research_v1/tests -p "test_[pmd]*.py"
```

Result: **27 tests, OK**. The run used `-W error::ResourceWarning`.

**In the global interpreter:**

```bash
python -B -m unittest discover -s research_v1/tests -p "test_foundations.py"
```

Result: **23 tests, OK**.

**Not run, and why:**

- `test_foundations.py` fails **in the venv** by design (2 errors and 1 failure: T3, which needs torch, and T9, which imports torch). Torch is deliberately absent from this CPU-only preprocessing venv. These tests do not use OpenCV, and they pass in the global interpreter.
- `test_preprocess.py` was **not** run in the global interpreter, because of its ambiguous OpenCV installation.

**New `test_preprocess.py` (9 tests):**

- channel order from synthetic red, green and blue source files through the documented cache build;
- exact normalisation arithmetic on constant images, and on grey 128;
- dtype, shape and contiguity;
- identity with the explicit cv2 `INTER_LINEAR` reference;
- wrong inputs are rejected;
- masks stay boolean, and image–mask alignment follows the exact centre mapping, within ≤ 1 pixel of the continuous geometry;
- rendered-equivalent L and P masks stay equivalent after resizing;
- repeated reads are identical and metadata is kept;
- six cache-index and metadata mismatch cases raise errors;
- both records of a Derm7pt same-file pair are kept, and identical cached pixels give identical inputs.

**`test_masks.py`:** +1 test (L-mask transparency). The old P-transparency test was updated to the new error text.

## 7. Real-data sample check and artifact

**Selection** is by `stable_seed("prep_check_v1", record_uid)`:

- 3 records from each of the 7 caches;
- 3 P-mode and 3 L-mode HAM-test masks (the L picks coincided with the `ham_test` cache sample);
- 2 PAD and 2 DDI images from RGBA sources;
- the Derm7pt same-file pair `case827`.

After removing duplicates, this gives 29 records. The full per-record results are in `sample_check.json`.

**Results:**

- All inputs are `float32 (3,224,224)` and finite.
- Repeated reads are identical.
- Cache vs reconstruction: exact for 29/29.
- Albumentations vs explicit cv2: identical for 29/29.
- The same-file pair gives identical inputs, and both records are kept.

**Run directories:**

- **Final:** `research_v1/runs/prep_check/20260927T130514Z_942f5ccf_0403c63c/`. Contains `sample_check.json`, `alignment_sheet.png` and a receipt. `verify_receipt` → **OK**.
  - The receipt records the manifests and their sidecars, the environment inventory, and the source files, including the legacy `code/precache.py`, `data.py` and `labels.py`.
  - It also records these supporting inputs: the manifest card, the Stage 2B2B mask CSV, the Stage 2B2A decode CSV, the venv freeze, and **all 17 legacy cache files** (arrays and metadata, by content hash).
  - It records the output hashes and the versions of cv2, numpy and albumentations.
- **Superseded:** `…130447Z_942f5ccf_21b178a8/`. Its checks are identical, but its alignment sheet showed only P masks because of the de-duplication. It is kept unchanged.

**Runtime** (measured machine time): sample check 4.1 s; venv tests 0.3 s; foundation tests 4.6 s. Creating the venv and installing packages was not timed.

## 8. Integrity

- `research_v1/runs/stage2b2c/` holds the before/after snapshots and the comparison: 200 files before, 201 after.
- **0 changed, 0 removed.** The only file added is `research_v1/configs/prep_check_v1.json`, which was expected.
- The cache `.npy` size and modification time are unchanged, and the cache contents are hashed in the receipt.
- The global OpenCV set is unchanged.
- New, as expected: `research_v1/.venv_prep/` and the run directories.
- Earlier receipts were not rewritten. They may show source-hash differences for `decode.py` and `config.py`, which were legitimately edited in this stage.

## 9. Remaining issues

1. The global OpenCV conflict (four distributions) remains. The production preprocessing path must run in `.venv_prep`, or in an equivalent single-OpenCV environment.
2. The venv has no torch. Any stage that combines this preprocessing with a model needs an environment decision: for example, a model environment with exactly one OpenCV, and with the equivalence re-checked by this same sample check.
3. Exact cache reconstruction was verified on a 29-record sample, not on every row. Full-cache verification could be added if required.
4. The palette-mask agreement with the legacy OpenCV masks was checked on 3 P masks only.
5. **Still open from earlier stages:**
   - the localisation rule for the all-white masks;
   - the RGBA alpha drop, which must be stated in Methods (all observed alpha channels are opaque);
   - the Secondary B endpoint for adapt_val, which is not estimable;
   - the choice of seed-0 primary model (legacy or new);
   - the training augmentation, which is not implemented.

**Stage 2B2C is complete. No augmentation or training.**
