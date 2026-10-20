# Stage 6A Report — ResNet-50 Seed Replicates 1 and 2

Date: 2026-10-03. All work in this stage used `research_v1/.venv_model`.

**Records:** `runs/stage6a/`. Model cards: `runs/stage6a/MODEL_CARDS_resnet50_s1_s2_v1.md`.

## Status

| Item | Result |
|---|---|
| Stage 5B interpretation note | `docs/STAGE5B_INTERPRETATION_NOTE_v1.md` (frozen 5B files unchanged) |
| Configs frozen before training | Addendum I; `train_main_resnet50_s1_v1.json` (`6e223df4…`) and `_s2_v1.json` (`c5aa6845…`) |
| Recipe comparison against Stage 3A | `recipe_comparison.json`: only `seeds.train_seed`, `training.sampler_seed`, `description` and the added Addendum I reference differ |
| Code drift against Stage 3A | `code_drift_vs_stage3a.json`: training-path modules unchanged. `config.py` changed only in its enum of purposes |
| Tests | New `test_seed_replicates.py` (4 tests) plus `test_best_selection` (5): all pass |
| Runs | Seed 1 and seed 2: 8/8 epochs each, exit 0, no non-finite loss, no warnings. **Run sequentially, with no interruption and no resume needed** |
| **Seed 1 selected** | **Epoch 7**, HAM VAL AUROC 0.8983, `ckpt_epoch7.pt`, SHA-256 `b70ff0e7…c781` |
| **Seed 2 selected** | **Epoch 8**, HAM VAL AUROC 0.8946, `ckpt_epoch8.pt`, SHA-256 `fecd74a2…5404` |
| Reload verification (fresh process, both seeds) | **Pass**: max \|Δp\| = 0.0 (tolerance 1e-6), AUROC reproduced, hashes and receipts OK |
| Protected inputs | 0 changed, 0 removed. The only additions are the two new run directories |

**Why the full suite was not run:** no shared code changed in this stage.

**Not done:** PAD adaptation, evaluation, thresholds and XAI.

## 1. Seed policy (Addendum I2)

| Changes with `train_seed` | Fixed by design |
|---|---|
| `seed_everything` (Python, NumPy, Torch, CUDA), which gives the **head initialisation** | `split_seed` 0: the HAM split and record lists |
| Augmentation sub-seeds `stable_seed(train_seed, epoch, image_uid)` | Pretrained backbone weights |
| Batch order through `sampler_seed` = 20260927 + `train_seed` | Validation order and batch; all hyperparameters; determinism flags |

**Discrepancy resolved before training.** Plan §8 specifies `stable_seed(train_seed, "loader")` for the loader, while Stage 3A used a fixed `sampler_seed` of 20260927.
- The rule above keeps seed 0 identical to Stage 3A and makes the sampling order seed-dependent.
- It is recorded as a documented deviation from the literal plan formula.

## 2. Verification (`post_training_verification_s1.json` and `_s2.json`)

**Identity**
- In `best.json`, the hash equals the file and its sidecar.
- The checkpoint's epoch and `train_seed` match.
- The `best.json` rule text says "HAM VAL", which is correct here.

**Initialisation**
- The backbone was built from the pretrained safetensors file, with all 318 tensors verified, not from seed 0 or the PAD checkpoint.
- `pos_weight` = 7.8608, from HAM TRAIN.
- Head-initialisation hashes differ between seeds (s0 `08ac7ef4…`, s1 `60063dbd…`, s2 `64d5d97e…`). Within a seed they are reproducible (test).

**Fresh start:** the first `lr_used` is 3e-4, the global steps run from 1 to 3472, and a fresh run refuses a directory that already holds a run (test).

**Data isolation**
- The 3472 steps used exactly the 6938 HAM TRAIN records, with 0 non-train records.
- The validation records are all HAM VAL.
- No HAM TEST, Derm7pt, PAD or DDI data were read.

**Reload:** HAM VAL was re-evaluated with the same order, batch 16 and `xai_precision_v1`. The IDs and labels are identical, and max |Δp| = 0.0 for both seeds.

## 3. Selection diagnostics (not performance)

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Selected |
|---|---|---|---|---|---|---|---|---|---|
| Seed 0 (Stage 3A) | 0.8701 | 0.8882 | 0.8815 | 0.8924 | 0.8937 | 0.8912 | 0.8948 | **0.8960** | 8 |
| Seed 1 | 0.8755 | 0.8804 | 0.8796 | 0.8887 | 0.8939 | 0.8967 | **0.8983** | 0.8942 | 7 |
| Seed 2 | 0.8656 | 0.8711 | 0.8814 | 0.8883 | 0.8780 | 0.8893 | 0.8877 | **0.8946** | 8 |

- These are HAM VAL selection diagnostics only.
- No seed is preferred, and seed 0 stays primary.
- The three runs are not pooled, and they do not establish general robustness.

## 4. Cost (measured)

| Seed | In-process time | Wall-clock | Epoch duration | Peak CUDA memory | Disk |
|---|---|---|---|---|---|
| 1 | 598.8 s | 16:38:38Z–16:48:41Z | 72–76 s | 1718 / 1922 MiB | 2.2 GB |
| 2 | 604.3 s | 16:48:41Z–16:58:49Z | 72–76 s | 1718 / 1922 MiB | 2.2 GB |

About 22 s of each epoch was step logging (input SHA-256 and fsync).

**GPU temperature:** 64 → 63 °C (seed 1) and 54 → 64 °C (seed 2), from snapshots taken only before and after each run.

**Operational note:** the background wrapper reported exit 1 because the `tail` command at the end of the wrapper used an invalid option. Both training processes logged `exit 0` and completed all epochs. Nothing was rerun.

## 5. Files

**New code, tests and config:**
- `src/rv1/seed_verify.py`, `tests/test_seed_replicates.py`;
- `configs/train_main_resnet50_s{1,2}_v1.json`.

**New documents:** `docs/PROTOCOL_ADDENDUM_2026-10-03_I.md`, `docs/STAGE5B_INTERPRETATION_NOTE_v1.md`.

**Runs:** `runs/main_train/resnet50_s1_20261003T163838Z/` and `resnet50_s2_20261003T164841Z/`. Each holds 8 checkpoints with sidecars, `latest.json` and `best.json`, step and epoch logs, `val_epoch1..8`, `selection_decisions.jsonl` and the receipt.

**`runs/stage6a/`:**
- `recipe_comparison.json`, `code_drift_vs_stage3a.json`;
- verification files;
- model cards;
- stdout, timestamps and GPU snapshots;
- integrity files.

**Integrity:** 746 files before, 816 after. **0 changed, 0 removed**; the 70 added files are all inside the two new run directories. The cache `.npy` files are unchanged, and seed 0 and every earlier artifact are untouched.

## 6. Next stage (not started)

- Per-model HAM-VAL threshold for seeds 1 and 2 with the §1.1 rule.
- Frozen-cohort inference for the planned seed-spread analyses: E1.1 and E1.2 spread on Derm7pt per the protocol, alongside seed 0.

**Stage 6A is complete.**
