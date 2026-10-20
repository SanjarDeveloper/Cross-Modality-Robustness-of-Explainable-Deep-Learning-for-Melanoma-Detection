# Stage 9A Report — RQ3 Architecture Readiness and Timing Pilots

Date: 2026-10-04. All work in this stage used `research_v1/.venv_model` (timm 1.0.28, torch 2.11).

**Scope:** HAM TRAIN and HAM VAL only. No main training, external evaluation, RQ2 sweep or XAI. No architecture was selected.

## Status

| Item | Result |
|---|---|
| Architecture contract | **Addendum J** (frozen before training) and `runs/stage9a/weights/weight_provenance.json` |
| EfficientNet-B3 | `efficientnet_b3.ra2_in1k`, Hub revision `0366a755…`, SHA-256 `279d2a53…0469` (equal to Hub LFS), 49.3 MB |
| DenseNet-121 | `densenet121.ra_in1k`, revision `92007b62…`, SHA-256 `eae900d8…da10` (equal to Hub LFS), 32.3 MB |
| Weight source | Copied from the existing local Hugging Face cache. **No new download.** The Hub LFS hashes were checked through a metadata query |
| Adapter | `src/rv1/arch_build.py`: a build function for the **existing** trainer. Classifier replaced by one logit; all non-classifier tensors verified (572 for EfficientNet-B3, 725 for DenseNet-121); all parameters trainable |
| Tests | New `test_arch_build.py` (4 tests). **Full suite 249/249**, run because `config.py` gained the two architecture names |
| Smoke and resume (fresh processes) | **Both PASS, and bit-identical.** Parameters, buffers, optimiser and probe probabilities all differ by 0.0; identical steps, order, digests and learning rates |
| Timing pilots | 2 epochs each on full HAM TRAIN (6938) with full HAM VAL passes (1529). Both completed; **no OOM** at batch 16 |
| Protected inputs | 0 changed, 0 removed. The primary ResNet-50 and all earlier runs are untouched |

## 1. Contract highlights (Addendum J)

**Input size**
- The study keeps its verified `prep_v1_cached256` → **224 px** path with ImageNet mean and std (identical to both pretrained configurations).
- EfficientNet-B3's recommended **288 px** input (test 320) is **not used**. This is a documented deviation that may disadvantage EfficientNet-B3, kept for a controlled comparison.
- DenseNet-121's recommended size is 224 px.

**Recipe** (Addendum C, unchanged)
- AdamW 3e-4 / 1e-4; BCE with `pos_weight` 7.8608 from HAM TRAIN.
- Batch 16, workers 0, FP32.
- Per-step cosine schedule; `aug_std_v1`.
- Validation under `xai_precision_v1`.
- Seed 0; sampler seed 20260927.

**Model structure**
- A fresh one-logit head, drawn after seeding. Head hashes: EfficientNet-B3 `43e409d7…`, DenseNet-121 `9c3aec44…`.
- Global average pooling (timm default). BatchNorm in normal train/eval mode.
- No ResNet-specific assumptions: the classifier names (`classifier.*`) and the provenance checks are per architecture.

## 2. Tests (`tests/test_arch_build.py`, CPU, real pinned weights)

| Area | What is checked |
|---|---|
| Head and outputs | Classifier → 1 output; forward gives (2, 1) and finite values |
| Training setup | All parameters trainable; more than 100 verified tensors; the ImageNet classifier tensors are discarded |
| Provenance | Altering one BatchNorm running mean in the reference makes the provenance check fail |
| Roles | Train and validation sets disjoint, with train mode and eval mode respectively |
| Training step | Finite loss and gradients for every parameter, and parameters update |
| Checkpoint | Save and reload with a strict load gives an identical state dict, including the BatchNorm buffers and the optimiser state |

## 3. Smoke and resume checks (`runs/rq3_smoke/`, `runs/stage9a/resume_compare_*.json`)

**Setup**
- Frozen smoke subset and probe (the Stage 2C2 lists), 2 epochs (32 steps).
- U = uninterrupted run. R = stop after epoch 1, then resume in a fresh process.
- The Stage 2C2 tolerances (1e-3) were frozen in J3.

**Result:** both architectures **PASS**:
- parameters and buffers max |Δ| = 0.0 (574 and 727 tensors);
- optimiser state 0.0; scheduler equal; RNG digests equal;
- probe probabilities and AUROC identical;
- fingerprints equal; no warnings.

These are implementation checks, not research results.

## 4. Timing pilots (measured; `runs/rq3_pilot/{effb3,dn121}/`)

| | EfficientNet-B3 | DenseNet-121 | ResNet-50 (Stage 3A, for reference) |
|---|---|---|---|
| Model and dataset build | 1.1 s | 1.2 s | 1.3 s |
| Epoch (train + validation + checkpoint) | 84.2 s | 84–88 s | 67–72 s |
| Training loop (incl. step logging) | 78.2–78.6 s | 78.7–81.3 s | 61.5–65.6 s |
| Step logging (SHA-256 + fsync) | about 19 s | about 16–17 s | about 20 s |
| Validation (full HAM VAL, highest FP32) | 5.0–5.4 s | 4.7–5.8 s | 4.6–5.0 s |
| Checkpoint commit | 0.6 s | 0.4 s | about 1 s |
| Peak CUDA memory (allocated / reserved) | 2865 / 3084 MiB | 2163 / 2504 MiB | 1718 / 1922 MiB |
| 2-epoch run wall-clock | 172.5 s | 175.1 s | — |
| Disk per checkpoint | about 124 MB | about 81 MB | about 275 MB |

**Thermal:** nvidia-smi was sampled every 30 s (12 samples). The GPU ranged from 56 to 72 °C. This is a short run and says nothing about long-run thermal stability.

**Scores:** the pilot validation AUROCs are recorded in the run logs but were **not used** for any decision. Pilot checkpoints are not final models.

## 5. RQ3 matrix inventory and cost projection

**Matrix** (02_rev2 §5): 3 architectures × 2 strategies × 3 seeds = **18 cells**.

| | Standard (`aug_std_v1`) | Heavy (`aug_heavy_v1`) |
|---|---|---|
| ResNet-50 | **seeds 0, 1, 2: trained** (Stages 3A and 6A) | 0/3: **blocked** (recipe unspecified) |
| EfficientNet-B3 | 0/3 (pilot only) | 0/3: blocked |
| DenseNet-121 | 0/3 (pilot only) | 0/3: blocked |

**Projection** (from the measured pilot epochs; an estimate, not a measurement):

| Item | Estimate |
|---|---|
| One 8-epoch run, EfficientNet-B3 or DenseNet-121 | about 8 × 86 s ≈ **11.5 min** (plus about 2 s of setup) |
| 6 remaining standard cells | about **69 min** of training. Disk: about 1.0 GB per EfficientNet-B3 run and 0.65 GB per DenseNet-121 run, so about 5 GB |
| Per-model evaluation (RQ3: E1.1–E1.5 and the RQ2 estimands) | HAM VAL threshold plus HAM TEST, Derm7pt, PAD and DDI inference is about 1–2 min. RQ2 sweep: 6.2 min measured for ResNet-50, about 7–8 min projected for the others. About **9–10 min per model**, so about 1 h for the 6 new models |
| 9 heavy cells | Training about 1.7–2 h if recipe overhead is small. Cannot be estimated until `aug_heavy_v1` exists |

**Budget:** the measured costs fit overnight-scale budgets, so no §R reduction is needed on cost grounds for the **standard** cells. The heavy cells are blocked by specification, not by compute.

## 6. Heavy-augmentation recipe: unresolved

**Definitions on record**
- **Legacy `code/train.py:_heavy_tf`:**
  - RandomResizedCrop (scale 0.5–1);
  - horizontal and vertical flips; ShiftScaleRotate (0.1, 0.25, 45°);
  - ColorJitter (0.4, 0.4, 0.4, 0.15);
  - one of GaussianBlur (3–9) / MotionBlur (9) / Defocus (3–8);
  - one of GaussNoise / ImageCompression (quality 30–90);
  - RandomBrightnessContrast (0.3, 0.3).
- **Protocol:** 02_rev2 §5 names "heavy augmentation" without parameters.
- **Status:** Addendum A and Stage 2B2D mark `aug_heavy_v1` as **deferred**.

**Overlap with the RQ2 perturbations**
- Blur ↔ blur.
- ColorJitter and brightness/contrast ↔ illumination.
- ImageCompression ↔ JPEG.
- RandomResizedCrop and ShiftScaleRotate partly ↔ scale.
- Nuisance (glare and hair) has **no** counterpart.

**Interpretation**
- Training on perturbation *types* that resemble the test perturbations measures robustness to **seen** shift types. That differs from generalisation to **unseen** shifts, and RQ2-on-heavy results must be labelled accordingly.
- Overlap does not automatically invalidate the experiment. It changes the question being answered. The legacy docstring explicitly avoids importing the simulator itself.

**Decision needed (not taken here):** specify `aug_heavy_v1` (parameters, ordering, seeding), and decide whether the overlap is an intended treatment. Options include keeping the legacy recipe with explicit labelling, or a variant that excludes the overlapping families.

**Not done:** no heavy recipe was implemented or run.

## 7. Recommended next bounded task (proposal only)

**Stage 9B: train the 6 remaining standard cells.**
- EfficientNet-B3 and DenseNet-121, seeds 0, 1 and 2.
- 8 epochs, Addendum C recipe, `sampler_seed` = 20260927 + seed (Addendum I).
- Selected checkpoints verified as in Stage 6A.
- Measured cost: about 70 min of training.

**Stage 9C:** per-model evaluation (thresholds, E1.1–E1.5, RQ2).

**Separately:** a decision and specification for `aug_heavy_v1`, before any heavy cell is run.

No matrix cell is dropped. §R reduction 2 ("heavy for ResNet-50 only") would apply only if the heavy runs later prove infeasible.

## 8. Stage 8C clarification

`docs/STAGE8C_INTERPRETATION_NOTE_v1.md`:
- Blur AUROC also declined, with CIs excluding 0 from s = 0.5.
- JPEG effects were not limited to quality 18: sensitivity rose and false positives increased at several levels.
- "No crossing" refers only to the tested grid and the bootstrap replicates.

## 9. Files

**Code, tests and config**
- New: `src/rv1/arch_build.py`, `tests/test_arch_build.py`.
- New configs: `configs/rq3_{smoke,pilot}_{effb3,dn121}_v1.json`.
- Changed: `src/rv1/config.py`, which adds `efficientnet_b3` and `densenet121` to the architecture enum. The prior copy is `runs/stage9a/config_before.py.txt`.

**Documents:** `docs/PROTOCOL_ADDENDUM_2026-10-04_J.md`, `docs/STAGE8C_INTERPRETATION_NOTE_v1.md`.

**Runs**
- `runs/rq3_smoke/{effb3,dn121}/{U,R}` (816 MB);
- `runs/rq3_pilot/{effb3,dn121}` (410 MB), with receipts;
- `runs/stage9a/`: weights and provenance, resume comparisons, stdout, GPU samples and integrity.

**Integrity:** 1318 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged. No `__pycache__` was created.

**Stage 9A is complete.**
