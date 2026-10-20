# Protocol Addendum C — 2026-09-27: Primary ResNet-50 Recipe (v1, seed 0)

This recipe is frozen **before** execution. Earlier documents are preserved.

It authorises **one** training run: ResNet-50, `train_seed = 0`.

It does **not** authorise:
- seeds 1–2;
- other architectures;
- RQ evaluation;
- test-set or external-data inference;
- full XAI.

Eight epochs is a bounded, prespecified recipe. It is not a claim that eight epochs is optimal. No early stopping and no performance-driven extension are allowed. The recipe is not changed in response to pilot, probe or smoke scores.

## C1. Model and data

| Item | Value |
|---|---|
| Model | timm `resnet50`, pretrained tag `resnet50.a1_in1k`, one melanoma logit |
| Weights | Hugging Face `timm/resnet50.a1_in1k`, commit `767268603ca0cb0bfe326fa87277f19c419566ef`, `model.safetensors`, SHA-256 `773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb`. The hash is verified before loading, and all backbone tensors are checked to be equal to the file |
| Head | New `nn.Linear` head (timm default initialisation), drawn after `seed_everything(0)`. **No** smoke or pilot weights are used as initialisation |
| Train | The complete frozen HAM TRAIN split: 6938 images (`configs/pilot_ham_train_full_v1.csv`, manifest order), with `aug_std_v1` |
| Validation | The complete frozen HAM VAL split: 1529 images (`configs/main_ham_val_full_v1.csv`), with `prep_v1_cached256` (evaluation path) |
| Not used | HAM TEST, Derm7pt, PAD-UFES-20 and DDI |

## C2. Optimisation

| Item | Value |
|---|---|
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4 |
| Loss | `BCEWithLogitsLoss`, `pos_weight` = n_neg/n_pos computed from HAM TRAIN only (6155/783 ≈ 7.861) |
| Batch size | 16 |
| `num_workers` | 0 |
| Epochs | 8 (total budget fixed from the start) |
| Scheduler | Per-step `CosineAnnealingLR`, `T_max` = 8 × 434 = 3472 |
| Seeds | `train_seed` 0; `sampler_seed` 20260927 |

## C3. Precision (training and evaluation are separate)

**Training.** Policy `train_backend = pytorch_default_tf32conv`, set explicitly at the start and recorded:
- `cudnn.allow_tf32 = True`;
- `cuda.matmul.allow_tf32 = False`;
- float32 matmul precision `"highest"`;
- `cudnn.deterministic = True` and `benchmark = False` (from `seed_everything`);
- FP32 tensors, **no AMP**.

This reproduces the policy tested in the Stage 2C2 and 2C3 training runs.

**Validation and later v1 evaluation.** `xai_precision_v1` (Addendum B), scoped:
- TF32 off, "highest", deterministic;
- the training settings are restored afterwards, and this is checked.

This keeps validation forward passes aligned with the later explanation forward passes.

## C4. Checkpoint selection

- After each epoch, the **full** HAM VAL set is evaluated in `eval()` mode under `no_grad`.
- Selection rule: the **highest image-level AUROC**; a tie keeps the **earlier** epoch.
- If the AUROC is undefined or non-finite, training stops and the problem is reported. There is no fallback.
- Two separate pointers are kept:
  - `latest.json`: for recovery;
  - `best.json`: for scientific evaluation, pointing to an immutable epoch checkpoint and its SHA-256.
- The best-so-far state is stored inside each checkpoint and survives resume.
- Per-image validation probabilities and record IDs are stored for every epoch.
- No decision threshold is computed at this stage.
- The validation results are checkpoint-selection results only. They are **not** independent test performance.

## C5. Durability and bounds

- Existing durable logging, epoch-boundary recovery and receipts are used.
- The run stops on a non-finite loss, OOM, failed integrity check, undefined validation AUROC, or lack of progress.
- The run is bounded to 30 minutes of wall-clock time. Batch size, precision and the epoch budget are never altered silently.
- No other GPU work runs during training.
