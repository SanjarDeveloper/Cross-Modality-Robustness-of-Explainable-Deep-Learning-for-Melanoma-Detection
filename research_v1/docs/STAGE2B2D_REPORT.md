# Stage 2B2D Report — Deterministic Training Augmentation

Date: 2026-09-27

## Status

| Aspect | Result |
|---|---|
| `aug_none`, `aug_std_v1` implemented | Yes |
| `aug_heavy_v1` | **Deferred.** It is not sufficiently specified (§1) |
| Tests in `research_v1/.venv_prep` (augment, preprocess, masks, decode) | **39 / 39 pass** |
| Tests in the global interpreter (foundation tests; they need torch) | **23 / 23 pass** |
| Real-data review, HAM **train** only | Artifact generated **and visually reviewed by the agent** (§5) |
| Torch DataLoader worker integration | **Not tested.** Deferred to the model-environment stage |
| Model / training | None. **Training is not authorised** |
| Protected inputs | 0 changed, 0 removed |

The decisions made at this stage are recorded in `PROTOCOL_ADDENDUM_2026-09-27_A.md`:
- the primary seed-0 model will be newly trained with v1;
- legacy seed 0 is not pooled with the new seeds;
- how the all-white masks are handled;
- the heavy recipe is deferred.

## 1. Recipes

### Sources inspected

- `code/data.py:LesionDataset._build` (legacy `train_aug`).
- `code/train.py:_heavy_tf`.
- The Albumentations 2.0.8 signatures and resolved attributes, read in the venv. The same version is installed globally.

### `aug_none`

- Image: `preprocess.to_model_input(img256)`.
- Mask: 256 → 224 with `INTER_NEAREST_EXACT`.
- Seed and epoch are ignored.
- A test confirms the output is **byte-identical** to evaluation preprocessing.

### `aug_std_v1`

The input is the legacy 256×256 RGB `uint8` cache row. The HAM mask is optional: `mask_interp_v1` → `{0,1}` → `cv2.resize((256,256), INTER_NEAREST_EXACT)`, which is the same source-to-256 mapping as in Stage 2B2C.

Transforms, applied in this order:

1. **RandomResizedCrop** → output (224, 224).
   - scale (0.7, 1.0); ratio (0.75, 1.3333333333333333).
   - Image interpolation: `INTER_LINEAR`.
   - Mask interpolation: **`INTER_NEAREST_EXACT`**. The legacy default was `INTER_NEAREST`; this change is documented.
   - p = 1.0.
2. **HorizontalFlip**, p = 0.5.
3. **VerticalFlip**, p = 0.5.
4. **ShiftScaleRotate**, p = 0.7.
   - Shift (−0.05, 0.05) in x and y.
   - Scale factor (0.9, 1.1), with the aspect ratio kept.
   - Rotation (−20, 20)°.
   - Image interpolation: `INTER_LINEAR`. Mask interpolation: `INTER_NEAREST`.
   - Border: `BORDER_CONSTANT` with fill 0 (black); mask fill 0.
   - `rotate_method = "largest_box"`.
   - Albumentations 2.x implements this as a special case of Affine and emits a deprecation warning, which is suppressed for this transform only.
5. **ColorJitter**, p = 0.5, applied to the **image only**.
   - brightness (0.8, 1.2), contrast (0.8, 1.2), saturation (0.8, 1.2), hue (−0.05, 0.05).
   - The library chooses the order of the four sub-operations randomly, using the seeded generator.
6. Normalisation: `float32 / 255` → `(x − MEAN) / STD` → CHW `float32 (3,224,224)`. This is the same step as in evaluation.

The recipe keeps the legacy parameter list. The only changes are the two documented ones (the mask interpolation in step 1 and the seeding, §2). Recipe choices were not informed by validation or test performance, because no model is involved.

### `aug_heavy_v1` — deferred

The legacy `_heavy_tf` contains GaussianBlur, MotionBlur, Defocus, GaussNoise and ImageCompression. These overlap the RQ2 perturbation factors. The protocol does not say whether that overlap is an intended experimental treatment, and it does not fix the transforms' parameters.

Inventing that treatment here would be a new scientific decision, so the recipe raises `NotImplementedError`.

**Consequence:** the RQ3 augmentation factor is blocked until `aug_heavy_v1` is specified and approved.

## 2. Seeding (stateless)

- `aug_seed = stable_seed(train_seed, epoch, image_uid)`, using the unchanged `rv1.stable_seed/1` encoding. The value lies in [0, 2⁶³−1].
- **A new pipeline is built for every sample**, so no transform object is shared between samples.

### Defect found and fixed

Albumentations 2.0.8 offers `Compose(seed=s)` and `Compose.set_random_seed(s)`. Both give **every** child transform its own generator, but seeded with the **same** `s`.

The effect: each transform's first draw is identical, so the apply/skip decisions become coupled.

- **Observed:** over 400 seeds, only 3 of the 16 possible combinations occurred. HorizontalFlip, VerticalFlip and ColorJitter always fired together.

**Fix (`_seed_pipeline`):**
- The Compose generator is seeded with `stable_seed(aug_seed, "albumentations_compose")`.
- Child *i* is seeded with `stable_seed(aug_seed, "albumentations_transform", i)`.

**Result after the fix:** over 1600 seeds, all 16 combinations occur. The observed marginal frequencies are ShiftScaleRotate 0.687, ColorJitter 0.497, HorizontalFlip 0.497 and VerticalFlip 0.479, against configured values of 0.7 / 0.5 / 0.5 / 0.5.

**Seed range:** no conversion is needed. The integers are passed unchanged to `numpy.random.default_rng` and `random.Random`.

**Not used:** Python's `hash()`, the global RNGs, or unseeded Albumentations randomness.

`augment_record` keys the seed on `image_uid`. The two reference rows of a Derm7pt same-file pair therefore receive the same augmentation for the same (train_seed, epoch). This is relevant only if such rows are ever trained on; under the protocol, only HAM train is used for training.

## 3. Geometry and masks

- **Joint transforms:** the same sampled geometric transform is applied to the image and the mask, with bilinear interpolation for the image and nearest-neighbour for the mask. Colour transforms touch the image only.
- **Masks:**
  - Masks stay boolean.
  - Existing flags (for example `all_foreground`) are carried forward.
  - `empty_after_augmentation` is added when a non-empty mask becomes empty after cropping or warping. Crops are never re-sampled.
  - Masks are diagnostic metadata only and are not model input.
- **Records:** `augment_record` returns an unchanged copy of the manifest record (IDs, label, split, role). The input record is not mutated.
- **Constant-fill borders:** rotation and shift leave black corners (fill 0) where the frame is empty. When ColorJitter's contrast step runs afterwards, these corners can become dark grey. This is inherited legacy behaviour, visible in the review sheet, and it is documented here rather than changed.

## 4. Tests

**Commands and results:**

```bash
research_v1\.venv_prep\Scripts\python.exe -B -W ignore -W error::ResourceWarning -m unittest discover -s research_v1/tests -p "test_[apmd]*.py"
```

→ **Ran 39 tests — OK**. This covers 12 augment, 9 preprocess, 8 mask and 10 decode tests. Interpreter: `research_v1/.venv_prep` (Python 3.10.11, single OpenCV).

```bash
python -B -m unittest discover -s research_v1/tests -p "test_foundations.py"
```

→ **Ran 23 tests — OK**. Interpreter: global Python 3.10.11, with torch.

The suite is split across **two interpreters**. It was not run as a single combined pass.

**New `test_augment.py` (12 tests):**
- The same (seed, epoch, image) gives an identical image and mask.
- Identical digests across independent subprocesses with `PYTHONHASHSEED` = 0 and 999, and equal to the in-process digests.
- Access order (forward vs reversed) does not change the per-sample outputs.
- Seeds 0 and 1 differ on at least 18 of 20 fixtures. Every individual draw is not required to differ.
- Coupling regression: at least 14 of 16 transform combinations occur over 200 seeds.
- Image–mask alignment: a white square and its mask have IoU ≥ 0.9 over 40 seeds.
- ColorJitter leaves the mask unchanged: the full pipeline gives the same mask as the geometric-only pipeline with the same per-index sub-seeds, while the images do differ.
- Masks stay boolean; existing flags are carried; every crop-induced empty mask is flagged, and only those; a non-{0,1} mask is rejected.
- The record and label are unchanged, and the input record is not mutated.
- `aug_none` equals evaluation preprocessing for both the image and the mask, for any train_seed or epoch.
- Evaluation input is unaffected by earlier augmentation calls, and the input array is not mutated.
- The deferred recipe, an unknown recipe and a missing seed each raise an error.

**A mistake I made:** the first version of the ColorJitter/mask test reused the full pipeline's transform objects, whose RNG state had already advanced. The test was corrected to build fresh objects. The implementation was not changed.

**Not tested:** torch DataLoader integration with `num_workers > 0` and worker-process seeding (torch is not in this environment). This is **deferred** to the model-environment stage. CPU subprocess independence was tested instead.

## 5. Real-data review (HAM train only)

**Sample:** the first 3 P-mode and 3 L-mode HAM **train** masks with status `ok`, ordered by `stable_seed("aug_review_v1", record_uid)`:

| Mode | Records |
|---|---|
| P | ISIC_0024596, ISIC_0024658, ISIC_0024562 |
| L | ISIC_0032692, ISIC_0033125, ISIC_0029445 |

Each is shown with `aug_none` and with `aug_std_v1` at (train_seed, epoch) = (0,0), (0,1) and (1,0).

**Findings from my visual review:**
- In all 18 augmented views, the mask overlay follows the lesion through crop, flip, rotation and shift.
- The constant-fill borders from rotation are visible.
- No mask became empty, and no mask was misaligned.
- Records were unchanged in every output.

This is a non-expert software check. It says nothing about clinical annotation quality or model behaviour.

**Output:** `research_v1/runs/aug_review/20260927T131330Z_c6b78b1b_2655dc61/`, containing `review_sheet.png`, `review.json` and the receipt. `verify_receipt` → **OK**.

The receipt records:
- the manifests, the environment inventory and the sources, including legacy `code/data.py`, `train.py` and `precache.py`;
- supporting inputs: the manifest card, the mask and decode audit CSVs, the venv freeze, and the `ham_train` cache files;
- the output hashes;
- the recipe and version details.

**Runtime:** the review run and the tests each took a few seconds. The test suites took 4.8 s in the venv and 4.5 s globally.

## 6. Files

| Path | Change |
|---|---|
| `src/rv1/augment.py` | New |
| `src/rv1/aug_review.py` | New |
| `tests/test_augment.py` | New |
| `src/rv1/config.py` | Adds purpose `aug_review`. Earlier configs are unaffected (tested) |
| `configs/aug_review_v1.json` | New |
| `docs/PROTOCOL_ADDENDUM_2026-09-27_A.md` | New |
| `runs/stage2b2d/` | Integrity evidence |

No environment changes: the `.venv_prep` freeze is identical to Stage 2B2C, and the global environment is untouched.

## 7. Integrity

- 215 files before, 217 after.
- **0 changed, 0 removed.**
- Added, both expected: `configs/aug_review_v1.json` and the protocol addendum.
- The cache `.npy` size and modification times are unchanged.
- No `__pycache__` was created.
- Earlier receipts were not rewritten. They may show source differences for `config.py`, which was legitimately extended.

## 8. Remaining model-environment checks

1. A model environment is needed with torch and exactly one OpenCV. In it, the Stage 2B2C sample check and these augmentation tests must be repeated. The output digests should equal those from `.venv_prep`.
2. Torch DataLoader integration still needs to be tested:
   - `num_workers` of 0 and > 0;
   - one pipeline per sample;
   - identical batches regardless of the number of workers;
   - no reliance on worker RNG.
3. `aug_heavy_v1` needs a specification before the RQ3 augmentation factor can run.
4. Still open from earlier stages:
   - the Secondary B endpoint for adapt_val, which is not estimable;
   - the full-cache reconstruction check (only a sample has been checked);
   - the global OpenCV conflict.

**Stage 2B2D is complete. No training.**
