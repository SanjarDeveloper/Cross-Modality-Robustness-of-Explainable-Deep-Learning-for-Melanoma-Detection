# Stage 9B Report — RQ3 Standard-Augmentation Training (EfficientNet-B3, DenseNet-121; Seeds 0–2)

Date: 2026-10-04. All work in this stage used `research_v1/.venv_model`.

**Records:** `runs/stage9b/`. Model cards: `runs/stage9b/MODEL_CARDS_rq3_standard_9b.md`.

## Status

| Item | Result |
|---|---|
| Six contracts frozen before launch | `configs/train_main_{effb3,dn121}_s{0,1,2}_v1.json` |
| Recipe comparison across all **nine** standard cells | `recipe_comparison_9cells.json`. The only differences are in architecture and pretraining (arch, model ID, weight file and hash), seed (`train_seed`, `sampler_seed`), description and supporting inputs. **No other recipe difference** |
| Structural defaults recorded | timm 1.0.28: dropout 0, no drop-path, average pooling. BatchNorm layers: EfficientNet-B3 78, DenseNet-121 121, ResNet-50 53 |
| Preflight | New `test_rq3_main_configs.py` (2 tests). `test_arch_build` (4) and `test_best_selection` (5) also pass |
| Preflight checks covered | Architecture, seed and sampler bindings; pinned weight hashes (not pilot checkpoints); identical recipe; HAM train and val lists **lesion-separated**, with no test or external data; disk 171 GB free |
| Training | **6/6 completed, each `TRAIN_EXIT 0`.** Sequential, in a detached launcher, with no competing GPU work, no interruptions and no resume needed. No warnings or non-finite losses |
| Fresh-process verification | **6/6 `VERIFY_EXIT 0`, `all_pass` true.** Reload max \|Δp\| = 0.0 (frozen tolerance 1e-6); identity and receipts OK |
| Protected inputs | 0 changed, 0 removed. The only additions are the six new run directories. The ResNet-50 models were not retrained |

**Why the full suite was not run:** no shared module changed. `rq3_verify.py` is a new script adapted from `seed_verify.py`.

## 1. Fresh initialisation (per run, verified)

- Each run starts from the **pinned original pretrained file**, with 572 (EfficientNet-B3) or 725 (DenseNet-121) backbone tensors verified. Runs never start from a pilot or another seed.
- Each run has a fresh seeded one-logit head. The head hashes differ by seed; seed 0 reproduces the Stage 9A pilot head.
- The optimiser and scheduler are fresh: the first `lr_used` is 3e-4, and steps run from 1 to 3472.
- Data isolation: training used only the 6938 HAM TRAIN records (0 non-train records), and validation used only HAM VAL.

## 2. Results (selection diagnostics only)

| Cell | Selected epoch | Best HAM VAL AUROC | Best SHA-256 | Time (s) | Peak CUDA memory (allocated / reserved, MiB) | Disk |
|---|---|---|---|---|---|---|
| EfficientNet-B3 s0 | 5 | 0.8964 | `5cb5097a…` | 657.6 | 2865 / 3084 | 1.04 GB |
| EfficientNet-B3 s1 | 4 | 0.8963 | `fb03d8fd…` | 654.0 | 2865 / 3084 | 1.04 GB |
| EfficientNet-B3 s2 | 6 | 0.9029 | `474e989b…` | 654.7 | 2865 / 3084 | 1.04 GB |
| DenseNet-121 s0 | 6 | 0.9124 | `5423f2eb…` | 680.2 | 2163 / 2504 | 0.68 GB |
| DenseNet-121 s1 | 5 | 0.9020 | `ca25f113…` | 677.3 | 2163 / 2504 | 0.68 GB |
| DenseNet-121 s2 | 8 | 0.9099 | `6cd274eb…` | 658.0 | 2163 / 2504 | 0.68 GB |

**Totals:** about 66.4 min of training and 5.2 GB of disk.

**GPU temperature:** 56–74 °C (60 s samples). This says nothing about long-run thermal stability.

**Selection data:** per-epoch HAM VAL probabilities (`val_epoch*.json`) and `selection_decisions.jsonl` are kept in each run.

**Best vs latest:** `best.json` is distinct from `latest.json` (epoch 8) in 5 of the 6 runs.

**Reading the AUROCs**
- These are **checkpoint-selection diagnostics** on the data used for selection. They are not evaluation results.
- No architecture or seed is chosen from them.
- **Seed-0 ResNet-50 remains the primary model.**
- This is a comparison under a **common recipe** at 224 px. It is not a comparison of individually optimised architectures, and the effect of using 224 px rather than EfficientNet-B3's recommended 288 px has **not** been quantified.

## 3. Verification (`runs/stage9b/verify_*.json`)

For every selected checkpoint:
- **Hashes:** the file, sidecar and `best.json` hashes agree.
- **Identity:** the checkpoint's architecture, `train_seed`, epoch and pretrained-weight hash match the config.
- **Fresh-process reload:** HAM VAL was re-evaluated with the same order, batch 16 and `xai_precision_v1`. IDs and labels are identical, max |Δp| = 0.0, and the AUROC is reproduced.
- **Steps:** 3472 steps, all from HAM TRAIN, all finite.
- **Receipts:** verify.

## 4. Operational notes

- **Monitoring:** the first monitor's log filter emitted nothing. This was a reporting issue, not a training failure: `status.txt` held the authoritative per-process exit codes, and a simpler status monitor replaced the first one.
- **START timestamps:** the timestamps in `status.txt` are identical because of cmd's variable expansion inside a `for` block. The actual start times and durations are in each run's receipt and logs.

## 5. RQ3 matrix inventory (`runs/stage9b/rq3_matrix_inventory.json`)

| | Standard | Heavy |
|---|---|---|
| ResNet-50 | s0, s1, s2: trained and verified (Stages 3A and 6A) | unimplemented |
| EfficientNet-B3 | s0, s1, s2: trained and verified (Stage 9B) | unimplemented |
| DenseNet-121 | s0, s1, s2: trained and verified (Stage 9B) | unimplemented |

- **9/9 standard cells are trained and verified.**
- The six new models are **not evaluated**: they have no thresholds and no external, RQ1 or RQ2 evaluation.
- The heavy cells remain unimplemented because `aug_heavy_v1` is unspecified.
- **RQ3 is incomplete.**

## 6. Files

**New code, tests and config:**
- `src/rv1/rq3_verify.py`, `tests/test_rq3_main_configs.py`;
- six configs.

**Runs:** `runs/main_train/rq3_{effb3,dn121}_s{0,1,2}_9b/`. Each holds 8 checkpoints with sidecars, `latest.json` and `best.json`, logs, per-epoch validation outputs, selection decisions and the receipt.

**`runs/stage9b/`:**
- `recipe_comparison_9cells.json`;
- training and verify stdout, `verify_*.json`;
- `status.txt` (exit codes);
- `gpu_samples.csv`;
- the inventory;
- the model cards;
- integrity files.

**Integrity:** files before and after were compared. 0 changed and 0 removed; every added file lies inside the six new `rq3_*_9b` run directories. The cache `.npy` files are unchanged.

## 7. Next (not started)

1. **Stage 9C:** per-model HAM VAL thresholds for the 6 new models, then the planned RQ3 evaluation: E1.1–E1.5 and the RQ2 estimands per model, with per-seed values and ranges and no between-architecture significance tests.
2. Specification and approval of `aug_heavy_v1` before any heavy cell is run.

**Stage 9B is complete.**
