# Stage 5A Report — One Protocol-Defined PAD Adaptation Run (RQ5, seed 0)

Date: 2026-10-03. All work in this stage used `research_v1/.venv_model`.

**Run:** `runs/adapt_train/resnet50_s0_pad_adapt_20261003T161843Z/`.

**Stage records:** `runs/stage5a/`.

## Status

| Item | Result |
|---|---|
| Contract frozen before training | Addendum H, data lists `pad_adapt_{train,val}_v1.csv` with sidecars, config `train_adapt_pad_s0_v1.json` (hash `1f58fe1a…`) |
| Data isolation (recomputed from the manifest) | adapt_train 805 / 482 / 17; adapt_val 343 / 204 / 9 (melanoma from 5 patients). Patient roles disjoint. **0 eval and 0 adapt_val records used in training steps** |
| Infrastructure | The **existing** trainer (`trainer.train`), with a new build function only (`adapt_train.build_resnet50_adapt`). No second trainer |
| Tests | New `test_adapt_train.py` (4 tests). Full suite **176/176** |
| Run | 5/5 epochs, exit 0, no warnings and no non-finite loss |
| **Selected** | **Epoch 4**, adapt_val AUROC 0.7152. `ckpt_epoch4.pt`, SHA-256 `a18bf969…1286` |
| Reload verification (fresh process) | **Pass**: same IDs and labels, max \|Δp\| = **0.0** (tolerance 1e-6), AUROC reproduced; hashes and receipt OK |
| Protected inputs | 0 changed, 0 removed |

**Why the full suite was run:** `config.py` changed (§6).

**Not done:** PAD eval, thresholds, DDI and XAI.

## 1. Contract (Addendum H)

**From the protocol (§7.2)**
- Initialisation from the Stage 3A best checkpoint. **All parameters and buffers** were loaded strictly and verified equal, including the BatchNorm running statistics.
- All layers trainable.
- AdamW, lr 1e-5, weight decay 1e-4.
- 5 epochs with a cosine schedule.
- `aug_std_v1`; seed 0.
- Selection by the highest adapt_val AUROC, with the earlier epoch winning ties, and the documented epoch-5 fallback.

**Retrospective implementation decisions** (not preregistered; the protocol is silent):

| Item | Decision |
|---|---|
| Loss | BCEWithLogits |
| `pos_weight` | 788/17 = **46.35**, from adapt_train only |
| Batch, workers | 16; 0 |
| Precision | FP32, no AMP |
| Sampler | Ordinary seeded shuffle (`sampler_seed` 20260927, as in Stage 3A) |
| Schedule granularity | Per-step cosine, `T_max` 255 |
| Backend | Stage 3A training backend, with validation under `xai_precision_v1` |
| BatchNorm | Normal train-mode behaviour |

None of these was derived from PAD eval.

**Optimiser and scheduler:** a **new** optimiser and scheduler were used, and none of the base checkpoint's optimiser or scheduler state was loaded. The step log confirms this: the first `lr_used` is 1e-5, and the global steps run from 1 to 255.

**Fallback:** the epoch-5 fallback was **not** triggered, because the AUROC was defined at every epoch.

## 2. Training results (selection diagnostics, not performance)

| Epoch | Mean training loss (weighted BCE) | adapt_val AUROC | Decision |
|---|---|---|---|
| 1 | 1.970 | 0.6830 | best |
| 2 | 1.662 | 0.7019 | best |
| 3 | 1.715 | 0.6989 | — |
| 4 | 1.688 | **0.7152** | **best (selected)** |
| 5 | 1.479 | 0.7112 | — |

**Caveats**
- adapt_val has only **9 melanoma images from 5 patients**, so the selection may be unstable. The AUROCs are selection diagnostics, not performance.
- The base model's adapt_val AUROC was not computed, and no improvement is claimed.
- **Text-label issue:** the `rule` field in `best.json` contains the trainer's fixed text "HAM VAL", which is a hard-coded string. Here the selection data were **PAD adapt_val**, as `probe_file` in the config and the `val_epoch*.json` record IDs confirm. The trainer was not changed.

**Measured cost**
- Wall-clock: 55.1 s in-process (16:18:43Z–16:19:41Z).
- Epochs took 9.8–11.8 s each, of which 2.6–3.2 s was step logging; validation took about 1.1 s.
- Peak CUDA memory: 1718 MiB allocated / 1922 MiB reserved.
- Disk: 1.4 GB for 5 checkpoints.

## 3. Verification (`runs/stage5a/post_training_verification.json`)

**Checkpoint identity:** the hash in `best.json` equals the file and its sidecar, and the checkpoint records epoch 4.

**Fresh-process reload**
- adapt_val was re-evaluated with the same order, batch 16 and `xai_precision_v1`.
- The IDs and labels are identical, max |Δp| = 0.0, and the AUROC equals the saved value.

**Step log**
- 255 steps, as expected (5 × ⌈805/16⌉).
- All 805 adapt_train records were used.
- 0 eval records and 0 adapt_val records were used.

**Receipt:** verifies.

## 4. Tests (`tests/test_adapt_train.py`, 4 tests; real frozen inputs, CPU)

| Area | What is checked |
|---|---|
| Base weights and buffers | Equal to the checkpoint (conv, fc and BN running statistics, including `num_batches_tracked`). The base file contains an optimiser state that the build never uses |
| Loaders | Contain only adapt_train (train mode) and adapt_val (eval mode) records, 805 and 343, with 9 validation melanomas; no eval records |
| `pos_weight` | 788/17, from adapt_train only |
| Split check | Rejects a list containing an eval record |

The selection, tie, best-versus-latest and resume behaviour of the unchanged trainer is covered by the existing `test_best_selection.py`.

## 5. Notes

- New: `docs/STAGE4C_INTERPRETATION_NOTE_v1.md`.
  - PAD is reported with detection counts (31/52 and 16/26), not "moderate performance".
  - The DDI AUROC CI [0.341, 0.571] includes 0.5, which **does not establish equivalence to chance**.
- Model card: `runs/stage5a/MODEL_CARD_resnet50_s0_pad_adapt_v1.md`.

## 6. Files

**Code, tests and config**
- New: `src/rv1/adapt_train.py`, `src/rv1/adapt_verify.py`, `tests/test_adapt_train.py`, `configs/train_adapt_pad_s0_v1.json`, `configs/pad_adapt_train_v1.csv` and `configs/pad_adapt_val_v1.csv` (with sidecars).
- Changed: `src/rv1/config.py`, which adds the purpose `train_adapt`. The prior copy is `runs/stage5a/config_before.py.txt`.

**Documents:** `docs/PROTOCOL_ADDENDUM_2026-10-03_H.md`, `docs/STAGE4C_INTERPRETATION_NOTE_v1.md`.

**Run and records**
- The run directory: 5 checkpoints with sidecars, `latest.json` and `best.json`, step and epoch logs, `val_epoch1..5_*.json` (per-epoch adapt_val predictions), `selection_decisions.jsonl` and the receipt.
- `runs/stage5a/`: verification, model card, stdout, timestamps and integrity.

**Integrity:** 670 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged. No `__pycache__` was created.

## 7. Next-stage requirements (not started)

1. **Primary RQ5 contrast:** AUROC(adapted) − AUROC(base) on the **identical PAD eval patients** (1150 images / 687 patients), with a 95% CI from a **patient-cluster paired bootstrap** (both models scored on each replicate). No image-level McNemar.
2. **Secondary A:** the adapted model's threshold, recomputed on **HAM VAL** with the §1.1 rule. Sensitivity and specificity differences with paired patient-cluster CIs.
3. **Secondary B:** **NOT_ESTIMABLE**, because adapt_val has 9 melanoma images, fewer than 20.
4. **Forgetting checks:** HAM TEST AUROC and Derm7pt clinical AUROC, before and after adaptation.
5. Freeze the evaluation specification before any PAD eval inference with the adapted model.

**No claim is made that adaptation improved performance.**

**Stage 5A is complete.**
