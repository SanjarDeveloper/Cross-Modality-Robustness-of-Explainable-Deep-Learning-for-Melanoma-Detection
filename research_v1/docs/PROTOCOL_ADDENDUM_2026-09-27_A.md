# Protocol Addendum A — 2026-09-27

**Applies to:** `02_RESEARCH_PROTOCOL_V1_DRAFT_rev2.md`. That document and every earlier planning document are kept unchanged. This addendum takes precedence only where it states so explicitly.

**Status:** researcher decisions recorded in Stage 2B2D. **Training is still not authorised.**

## A1. Primary seed-0 model

The primary v1 seed-0 model (ResNet-50, `aug_std_v1`) will be **newly trained with the v1 pipeline**:

- `prep_v1_cached256` inputs;
- stateless augmentation seeding;
- the v1 threshold rule.

The legacy checkpoint `checkpoints/resnet50_standard_s0.pt` is used **only** in the separate legacy-reproduction (LR) analysis (02_rev2 §1.3).

## A2. No pooling across recipes

Legacy seed 0 must **not** be pooled with newly trained seeds 1–2 as though the three models followed the same training recipe. Seed-spread summaries use v1-trained models only.

## A3. HAM all-white masks

ISIC_0026042 (val) and ISIC_0029819 (train):

- **Classification:** both remain in the classification cohorts.
- **Localisation:** both are flagged `all_foreground`. For any later localisation analysis:
  - primary analysis: exclude them, and report the count and the IDs;
  - sensitivity analysis: include them.
- **Current effect:** neither image is in HAM test, so the planned test-set localisation metrics are currently unaffected.

## A4. Augmentation recipes (definition; see STAGE2B2D_REPORT.md for exact parameters)

**`aug_none`** is identical to evaluation preprocessing.

**`aug_std_v1`** uses the parameter list of the legacy `train_aug`, as interpreted by Albumentations 2.0.8, with two documented changes:

1. **RandomResizedCrop mask interpolation** is `INTER_NEAREST_EXACT` (centre-aligned). The legacy default was `INTER_NEAREST`.
2. **Seeding.** Each transform has its own stable sub-seed, derived from `stable_seed(train_seed, epoch, image_uid)`. This avoids Albumentations' identical-seed coupling.

**`aug_heavy_v1` is DEFERRED.** The protocol does not specify it sufficiently. The legacy `_heavy_tf` overlaps the RQ2 perturbation factors (blur, noise, JPEG), and whether that overlap is an intended experimental treatment has not been decided.

**Consequence for RQ3:** the "standard vs heavy augmentation" factor cannot be run until `aug_heavy_v1` is specified and approved. This does not block RQ1, RQ2, RQ4 or RQ5.
