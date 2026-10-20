# Protocol Addendum I — 2026-10-03: ResNet-50 Seed Replicates 1 and 2 (Stage 6A)

This addendum is frozen **before** training. Earlier documents are preserved.

It authorises two HAM training runs, with `train_seed` 1 and `train_seed` 2. They exactly replicate the Addendum C recipe and serve only to show seed spread (02_rev2 §2/§3).

**Seed 0 remains the prespecified primary model.** Seeds are never pooled as extra cases.

## I1. Recipe (identical to Addendum C / `train_main_resnet50_s0_v1.json`)

| Item | Setting |
|---|---|
| Pretrained weights | `resnet50.a1_in1k`, file SHA-256 `773525d5…41bb`. All backbone tensors are verified against the file. **Each run starts from the original pretrained backbone**, not from seed 0 or the PAD-adapted checkpoint |
| Head | A fresh one-logit head |
| Optimiser and scheduler | Fresh |
| Train / val lists | `configs/pilot_ham_train_full_v1.csv` (6938) and `configs/main_ham_val_full_v1.csv` (1529), identical lists |
| Training | AdamW 3e-4 / 1e-4; BCE with `pos_weight` from HAM TRAIN; batch 16; workers 0; FP32; 8 epochs; per-step cosine (`T_max` 3472); `aug_std_v1`; `prep_v1_cached256` |
| Backend and validation | `pytorch_default_tf32conv` backend; validation under `xai_precision_v1` |
| Selection | Highest HAM VAL AUROC; ties keep the earlier epoch |

**No change after results:** epochs and hyperparameters are not changed after any validation result is seen. No seed is preferred.

## I2. Seed policy, with one discrepancy resolved explicitly

**Plan 03_rev2 §8:** the DataLoader generator is seeded from `stable_seed(train_seed, "loader")`.

**Stage 3A as run (Addendum C):** used a separate config field, `sampler_seed = 20260927`, through `stable_seed(sampler_seed, "sampler", epoch)`.

These two definitions differ, and no single rule reproduces both.

**Decision for Stage 6A** (a retrospective implementation decision; not preregistered):
- `sampler_seed = 20260927 + train_seed`.
- Seed 0 → 20260927, which matches Stage 3A exactly.
- Seed 1 → 20260928; seed 2 → 20260929.
- This makes the shuffling stream seed-dependent, as the intent of plan §8 requires, while leaving the seed-0 run valid.
- The literal `stable_seed(train_seed, "loader")` formula is **not** used; that is a documented deviation.

**RNG streams that change with `train_seed`:**

| Stream | How it depends on `train_seed` |
|---|---|
| `seed_everything(train_seed)` (Python, NumPy, Torch, CUDA) | Gives the **head initialisation**, which is drawn after seeding |
| Augmentation | Per-transform sub-seeds from `stable_seed(train_seed, epoch, image_uid)` |
| Batch order | Through `sampler_seed`, by the rule above |

**Fixed by design:**
- `split_seed` 0: the HAM lesion split and the record lists;
- the pretrained backbone weights;
- the validation order and batch;
- `CUBLAS_WORKSPACE_CONFIG`;
- determinism flags;
- all hyperparameters.

## I3. Reload verification tolerance (frozen now)

Each selected checkpoint is reloaded in a fresh process and HAM VAL is re-evaluated with the same order, batch 16 and `xai_precision_v1`.

**Acceptance:**
- identical IDs and labels;
- maximum |Δp| ≤ 1e-6 (0 expected);
- the AUROC reproduced.
