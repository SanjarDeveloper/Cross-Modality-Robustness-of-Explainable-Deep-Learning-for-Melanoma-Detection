# Stage 9D Report — Definition and Validation of `aug_heavy_v1`

Date: 2026-10-04. All work in this stage used `research_v1/.venv_model` (Albumentations 2.0.8, OpenCV 4.11.0).

**Scope:** no heavy-model training or evaluation. HAM TRAIN only.

## Status

| Item | Result |
|---|---|
| Treatment definition | The legacy `code/train.py:_heavy_tf`, with order, explicit parameters and probabilities preserved. **Addendum K** and `configs/aug_heavy_v1_spec.json` (module SHA-256 `df6474a3…`) were frozen before the review and the timing check |
| Implementation | `src/rv1/augment.py`, extended with `aug_heavy_v1`. Every implicit default is resolved explicitly. Nested OneOf children get distinct sub-seeds |
| `aug_std_v1` and `aug_none` regression | **96/96 output digests identical** before and after the edit (24 HAM TRAIN images × `aug_none` plus 3 `aug_std_v1` seed/epoch pairs, image and mask) |
| Tests | New `test_augment_heavy.py` (7 tests). `test_augment.py` updated: one assertion that encoded the old *deferred* status (the old copy is kept). **Full suite 268/268** |
| Review | `runs/aug_heavy_review_9d/20261004T113020Z_fac166c0_7c72e964/`. Sheet inspected (§4); receipt verifies |
| Throughput (augmentation plus DataLoader only) | `aug_heavy_v1` takes about **1.6×** the time of `aug_std_v1` (§5) |
| Protected inputs | 0 changed, 0 removed |

## 1. Frozen recipe (Addendum K)

| # | Transform | Parameters | p |
|---|---|---|---|
| 1 | RandomResizedCrop | 224, scale (0.5, 1.0), ratio (3/4, 4/3), INTER_LINEAR, mask INTER_NEAREST_EXACT | 1.0 |
| 2–3 | HorizontalFlip, VerticalFlip | — | 0.5 each |
| 4 | ShiftScaleRotate | shift 0.1, scale 0.75–1.25, ±45°, INTER_LINEAR, BORDER_CONSTANT fill 0, mask fill 0, mask INTER_NEAREST, largest_box | 0.8 |
| 5 | ColorJitter | brightness, contrast and saturation (0.6, 1.4); hue ±0.15 | 0.8 |
| 6 | OneOf (equal weights) | GaussianBlur (3–9, σ 0.5–3); MotionBlur (3–9, shifted, 0–360°, direction ±1); Defocus (radius 3–8, alias 0.1–0.5) | 0.5 |
| 7 | OneOf (equal weights) | GaussNoise (std 0.2–0.44 of the maximum, mean 0, per channel); ImageCompression (JPEG, quality 30–90) | 0.4 |
| 8 | RandomBrightnessContrast | ±0.3 / ±0.3, brightness_by_max, ensure_safe_range False | 0.6 |

**Pipeline:** input is the existing cached-256 training path, and output goes through the same normalisation as `aug_std_v1` to (3, 224, 224) float32. There is no extra resize.

**Compatibility notes** (recorded before execution)
- The legacy code calls the Albumentations 2.x API (`quality_range=`), so its implicit defaults are the 2.x defaults; the installed 2.0.8 reproduces them.
- **GaussNoise:** the 2.x default is std 0.2–0.44 of the maximum value (about 51–112 grey levels). That is much stronger than the pre-1.4 default (variance 10–50, std about 3–7 grey levels). The installed 2.x behaviour is used, as the legacy code implies. See §4.
- **Mask interpolation:** INTER_NEAREST_EXACT for RandomResizedCrop, as in `aug_std_v1`, instead of the legacy default INTER_NEAREST.

## 2. Scientific interpretation (recorded in K3)

- **Timing:** the legacy recipe predates the v1 standard results. The v1 implementation details were frozen **after** Stages 9C1 and 9C2 and were not tuned in response to them.
- **Overlap with RQ2:** blur, colour/brightness, JPEG and crop/scale overlap RQ2 families; nuisance does not. RQ2 results on heavy models therefore measure robustness to **seen families**, not generalisation to unseen shift types.
- **Not leakage:** family overlap is not patient or image leakage. No test images, scores or labels are used, the RQ2 simulator is not imported, and matching transform families are not automatically leakage.
- **The treatment** is the whole package, not an isolated blur, colour, compression or crop effect.
- **Status:** the strategy comparison is exploratory and was specified retrospectively.

## 3. Determinism and tests (`tests/test_augment_heavy.py`, 7 tests)

**Determinism**
- The same seed, epoch and record give identical outputs, including **across processes with PYTHONHASHSEED 0 and 999**.
- Sample order has no effect.
- A different epoch gives a different output, reproducibly.

**Activation frequencies** (N = 2000 seeds; tolerance of 4 binomial SDs, fixed beforehand). All observed rates are within tolerance:

| Transform | Expected rate |
|---|---|
| Crop | 1.0 |
| Each flip | 0.5 |
| ShiftScaleRotate | 0.8 |
| ColorJitter | 0.8 |
| RandomBrightnessContrast | 0.6 |
| Each blur child | 0.5/3 |
| Each noise/JPEG child | 0.4/2 |
| Any blur | 0.5 |

**OneOf and independence**
- Each OneOf applies **at most one child**.
- **No shared RNG streams:** P(both flips) ≈ 0.25 and P(blur and noise group) ≈ 0.20, as expected for independent decisions. Without the per-child seeding fix, Albumentations would give all children the same seed.

**Contract**
- Outputs are (3, 224, 224) float32 and finite; masks are (224, 224) bool.
- The input is not mutated, and the label and record are unchanged.
- Image and mask geometry stay aligned.
- An empty mask after cropping is flagged `empty_after_augmentation`, and the sample is not replaced.

## 4. Bounded HAM TRAIN review (sample fixed before viewing)

**Sample:** the first record per combination of label and mask mode in stable-seed order:

| Record | Label | Mask mode |
|---|---|---|
| ISIC_0029487 | non-melanoma | L |
| ISIC_0024715 | non-melanoma | P |
| ISIC_0025097 | melanoma | L |
| ISIC_0024586 | melanoma | P |

The sheet shows the original and 6 heavy samples (epochs 0–5), with mask boundaries overlaid.

**Observations (technical, not clinical review)**
- **Strong Gaussian noise.** When GaussNoise is selected (p = 0.2), the image becomes heavily speckled and lesion texture is largely obscured. This follows from the 2.x default std range, see §1.
- **Colour.** Large hue and saturation shifts (yellow or purple casts) and washed-out images. In some samples the lesion becomes faint, for example ISIC_0024586 at epochs 1 and 5.
- **Geometry.** Rotation and shift leave black corners. RandomBrightnessContrast and ColorJitter run after the fill, so those corners can become grey or white.
- **Crop.** Strong zoom from scale 0.5. Large lesions can extend beyond the frame.
- **Masks.** Masks follow the geometry correctly, and photometric changes leave them unchanged. No empty-mask flags occurred in the sample.

**No defect was found.** These are properties of the frozen recipe, and no parameter was tuned.

**Decision point for you:** the GaussNoise strength is the installed-version default that the legacy code implies. If the intended legacy behaviour was the milder pre-1.4 noise, that would need an explicit, versioned amendment before heavy training.

## 5. Throughput and integration (`review_and_timing.json`)

**Setup**
- The same fixed 512-image HAM TRAIN sample.
- `ManifestDataset` + `make_epoch_loader`, batch 16, workers 0, two repeats per recipe, perf_counter timing.

**Results**

| Recipe | Images/s per repeat |
|---|---|
| `aug_std_v1` | 213, 301 (the first repeat includes warm-up) |
| `aug_heavy_v1` | 154, 158 |

- Mean time ratio heavy/std = **1.60**.
- All batches meet the model input contract: (B, 3, 224, 224) float32, finite.
- **Scope of this number:** this is augmentation overhead only. It is **not** a measured heavy-training time.
- **What the training loop does:** the existing loop runs at about 100–113 samples/s, GPU and logging bound, with augmentation inline (workers 0).
- **Projection (an estimate):** about 15–20 s extra per epoch. The heavy training/resume pilot will measure the real figure.

## 6. Stage 9C2 note

`docs/STAGE9C2_INTERPRETATION_NOTE_v1.md`:
- Blur's leading sensitivity loss has a ranking share of only 0.58 for EfficientNet-B3 s1.
- JPEG produced worst-case sensitivity losses of 17.1 and 10.6 pp in two EfficientNet-B3 runs.

## 7. Files

**Code, tests and config**
- Changed: `src/rv1/augment.py` (prior copy `runs/stage9d/augment_before.py.txt`) and `tests/test_augment.py` (prior copy kept).
- New: `src/rv1/aug_heavy_review.py`, `tests/test_augment_heavy.py`.
- New configs: `configs/aug_heavy_v1_spec.json`, `configs/aug_heavy_review_9d_v1.json`.

**Documents:** `docs/PROTOCOL_ADDENDUM_2026-10-04_K.md`, `docs/STAGE9C2_INTERPRETATION_NOTE_v1.md`.

**Run outputs:** `review_aug_heavy_v1.png`, `review_and_timing.json` and the receipt.

**Regression evidence:** `runs/stage9d/digests_{before,after}.json` and `regression_digests.py`.

**Integrity:** the before and after snapshots were compared. 0 changed and 0 removed; only the new documents were added. The cache `.npy` files are unchanged.

## 8. Next (not started)

1. A bounded heavy training/resume pilot: smoke U vs R on the frozen subset, then a short timing run per architecture.
2. The 9 heavy cells (3 architectures × 3 seeds), only after that verification passes, and after a decision on the GaussNoise strength if you wish to revisit it.

**Stage 9D is complete.**
