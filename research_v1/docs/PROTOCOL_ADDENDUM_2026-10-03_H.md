# Protocol Addendum H — 2026-10-03: RQ5 PAD Adaptation Contract (Stage 5A)

This addendum is frozen **before** adaptation training. Earlier documents are preserved.

It authorises one adaptation run, with training seed 0. It does **not** authorise:
- PAD eval inference;
- threshold selection;
- DDI;
- XAI;
- any recipe search.

## H1. Protocol-defined settings (02_rev2 §7.2)

| Item | Setting |
|---|---|
| Starting point | Primary model: Stage 3A `best.json` → `ckpt_epoch8.pt`, SHA-256 `303d21e7…03dd`. **All parameters and buffers** (including BatchNorm running statistics) are loaded strictly from the checkpoint |
| Trainable layers | All |
| Optimiser | AdamW, lr 1e-5, weight decay 1e-4 |
| Epochs | 5 |
| Schedule | Cosine over the 5 epochs |
| Augmentation | `aug_std_v1` |
| Training seed | 0 |
| Selection | Highest image-level adapt_val AUROC after each epoch; ties keep the earlier epoch |
| Fallback | If the AUROC cannot be computed, use the epoch-5 checkpoint and record this |

**Optimiser and scheduler state.** A **new** optimiser and a **new** scheduler are created. The HAM optimiser and scheduler states stored in the base checkpoint are **not** loaded.

## H2. Retrospective implementation decisions (not preregistered; protocol silent)

None of these is derived from PAD eval, which has not been inferred on with any adapted model.

| Item | Decision |
|---|---|
| Loss | `BCEWithLogitsLoss` |
| `pos_weight` | n_neg / n_pos on **adapt_train only**: 788/17 = 46.35 |
| Batch, workers | 16; 0 |
| Precision | FP32, no AMP |
| Sampler | Ordinary seeded shuffle, the same mechanism and `sampler_seed` (20260927) as Stage 3A. No weighted sampler |
| Schedule granularity | Per-step `CosineAnnealingLR`, `T_max` = 5 × ⌈805/16⌉ = 255. This is the existing trainer, as in Stage 3A |
| Training backend | `pytorch_default_tf32conv`, as in Stage 3A |
| Validation precision | `xai_precision_v1` |
| BatchNorm | Normal `train()` behaviour; the running statistics update |

**Selection fallback.** The adapt_val set is fixed at 343 images with 9 melanoma, so its AUROC cannot be undefined for lack of a class. The existing trainer raises `ValidationUndefined` only for single-class input or non-finite outputs. Non-finite outputs are an implementation failure: training stops and the failure is reported, and the epoch-5 fallback is **not** used to hide it.

## H3. Data (frozen manifest; `rq5_role`)

| Role | Images | Patients | Melanoma images | Melanoma patients | List |
|---|---|---|---|---|---|
| adapt_train | 805 | 482 | 17 | 13 | `configs/pad_adapt_train_v1.csv` |
| adapt_val | 343 | 204 | 9 | 5 | `configs/pad_adapt_val_v1.csv` |
| eval | 1150 | 687 | 26 | 18 | **Excluded** from training and selection |

- **Patient roles are disjoint.**
- **Mixed-diagnosis patients:** per-image labels are retained.
- **Selection instability:** with 9 melanoma images from 5 patients, checkpoint selection **may be unstable**. The split is not enlarged or replaced.

## H4. Reload verification tolerance (frozen now)

The selected checkpoint is reloaded in a fresh process, and adapt_val is re-evaluated with the same batch size, order and precision policy.

**Acceptance:**
- identical record IDs and labels;
- maximum |Δp| ≤ 1e-6 against the probabilities saved during training (0 is expected);
- the AUROC reproduced.
