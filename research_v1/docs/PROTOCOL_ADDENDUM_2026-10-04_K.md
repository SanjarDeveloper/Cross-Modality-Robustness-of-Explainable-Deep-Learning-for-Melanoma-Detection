# Protocol Addendum K — 2026-10-04: `aug_heavy_v1` Definition (Stage 9D)

This addendum is frozen before the visual review and the timing check, and before any heavy-model training. Earlier documents are preserved.

## K1. Treatment definition

**Decision:** the legacy `code/train.py:_heavy_tf` is the treatment, with its order, explicit parameters and probabilities preserved. Every default left implicit in the legacy code is resolved for **Albumentations 2.0.8**, the installed version. The legacy code already uses the 2.x API (`quality_range`), so its defaults are the 2.x defaults.

**Explicit choices:**
- Input size 224 from the existing cached-256 path, with the existing normalisation.
- Mask interpolation in RandomResizedCrop is INTER_NEAREST_EXACT, as in `aug_std_v1`.
- The parameters of the ShiftScaleRotate border are written out explicitly. They equal the installed defaults.

| # | Transform | Parameters (all resolved) | p |
|---|---|---|---|
| 1 | RandomResizedCrop | size 224, scale (0.5, 1.0), ratio (3/4, 4/3), INTER_LINEAR, mask INTER_NEAREST_EXACT | 1.0 |
| 2–3 | HorizontalFlip, VerticalFlip | — | 0.5 each |
| 4 | ShiftScaleRotate | shift 0.1, scale factor 0.75–1.25, rotate ±45°, INTER_LINEAR, BORDER_CONSTANT, fill 0, mask fill 0, mask INTER_NEAREST, largest_box | 0.8 |
| 5 | ColorJitter | brightness, contrast and saturation (0.6, 1.4); hue ±0.15; sub-operation order randomised | 0.8 |
| 6 | OneOf | GaussianBlur (kernel 3–9, σ 0.5–3.0); MotionBlur (kernel 3–9, allow_shifted, angle 0–360, direction ±1); Defocus (radius 3–8, alias_blur 0.1–0.5). Equal weights (each child's p = 0.5) | 0.5 |
| 7 | OneOf | GaussNoise (std 0.2–0.44 of the maximum value, mean 0, per channel, scale factor 1); ImageCompression (JPEG, quality 30–90). Equal weights | 0.4 |
| 8 | RandomBrightnessContrast | ±0.3 / ±0.3, brightness_by_max True, ensure_safe_range False | 0.6 |

**OneOf semantics** (Albumentations 2.0.8):
- Activation is decided by the OneOf's own `py_random`.
- The child is chosen from its own generator, with probability proportional to the child's p.
- The chosen child is applied with `force_apply=True`.

**Masks:** photometric transforms (5–8) never touch masks. An empty mask after cropping is flagged `empty_after_augmentation`, and the sample is kept.

## K2. Seeding

- Stateless: `aug_seed = stable_seed(train_seed, epoch, image_uid)`.
- A new Compose is built per sample, with distinct sub-seeds for the Compose and for each top-level transform.
- **Nested OneOf children get their own sub-seeds:** `stable_seed(seed, "albumentations_transform", i, "child", j)`. Albumentations' `OneOf.set_random_seed` would otherwise give every child the same seed.
- No global RNG is used and no transform instance is shared.

## K3. Scientific interpretation

**Timing**
- The legacy recipe predates the v1 standard-model results.
- Its v1 implementation details were frozen **after** the Stage 9C1 and 9C2 results. They were not tuned in response to them.

**What the treatment is**
- Several training families overlap RQ2 evaluation families: blur, colour/brightness, JPEG, and crop/scale. Nuisance (glare and hair) has no counterpart.
- RQ2 results on heavy models therefore measure robustness to **seen perturbation families**. They are not evidence of generalisation to entirely unseen shift types.
- Family overlap is not patient or image leakage: no test images, scores or labels are used, and the RQ2 simulator is not imported.
- The treatment is the **whole augmentation package**. It is not an isolated blur, colour, compression or crop effect.

**Status:** the strategy comparison is exploratory and was specified retrospectively.
