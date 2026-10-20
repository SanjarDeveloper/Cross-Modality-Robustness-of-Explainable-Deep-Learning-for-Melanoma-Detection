# Stage 3A Report — Frozen Primary Recipe and First Real v1 Model (ResNet-50, seed 0)

Date: 2026-09-27. All work in this stage used `research_v1/.venv_model`.

## Status

| Item | Result |
|---|---|
| Control fixes (composite mismatch keys; strict control-score validation) | Implemented and tested |
| Best-checkpoint machinery | Implemented and tested |
| Tests before training | **124/124 pass**; re-run after the final config edit: 124/124 |
| Recipe frozen before execution | Addendum C and config `train_main_resnet50_s0_v1.json` (hash `87351832…`); pre-launch fingerprint recorded |
| Single seed-0 training run | **Completed**: 8/8 epochs, exit 0, no non-finite loss, no OOM, no warnings |
| Selected checkpoint | **Epoch 8**, HAM VAL AUROC 0.8960 (the highest; no tie) |
| Post-training verification | Pass (§5) |
| Test or external data used | **None** |
| Protected inputs | 0 changed, 0 removed |

Not started: seeds 1–2, other architectures, any RQ evaluation, test or external inference, and the real-model XAI pilot.

## 1. Control fixes (`xai_controls.py`)

**A. Composite mismatch keys**
- Each assignment is keyed by `record_uid|modality|explainer|condition` (`assignment_key`).
- Each entry stores its full identity plus `donor_key`.
- Duplicate composite keys raise an error.
- The deterministic donor rule and the invalid-donor policy are unchanged.
- `resolve_mismatch` now takes the composite key.

**B. Strict score validation**
- The random control requires exactly 5 finite AUCs for **both** metrics. If any value is missing, None, NaN or Inf, the comparison is unavailable; the mean is never taken over only the successful seeds.
- A missing or non-finite AUC for the explainer or for a single-map control also makes the comparison unavailable.

`xai_control_diag.py` was updated to the new keys. Earlier outputs are unchanged.

## 2. Best-checkpoint machinery (`trainer.py`, `config.py`)

**New optional training fields**
- `train_backend` (`pytorch_default_tf32conv`): applied explicitly at the start and recorded.
- `eval_precision` (`xai_precision_v1`).
- `best_model_selection` gains the value `val_auroc_max_earliest_tie`.
- Existing configs are unaffected.

**Behaviour after each epoch**
- `evaluate_selection` runs the full validation pass under `xai_precision_v1`. It checks that the training backend settings were restored, and raises `ValidationUndefined` if the AUROC is undefined (a single class) or non-finite. In that case the epoch is **not** committed.
- The best-so-far state `{val_auroc, epoch, checkpoint}` is stored in every checkpoint and is restored on resume, including from an adopted orphan checkpoint.
- `best.json` is a separate, atomically written pointer to an immutable epoch checkpoint and its SHA-256. `latest.json` remains the recovery pointer.
- `val_epoch<k>_<attempt>.json` stores the record IDs, labels, per-image probabilities and AUROC.
- `selection_decisions.jsonl` records every decision.

**Tests** (`tests/test_best_selection.py`, 5 tests):
- best differs from latest, and a tie keeps the earlier epoch (scripted AUROCs 0.60/0.70/0.70 → epoch 2), with the SHA-256 in `best.json` verified;
- the best state survives a clean stop and resume, including a later tie, and a crash after the checkpoint but before the pointer update (restored from the adopted checkpoint);
- a NaN validation AUROC stops training without committing that epoch;
- the real `evaluate_selection` rejects single-class and NaN outputs;
- validation runs with TF32 off, and the training settings are restored afterwards.

## 3. The frozen recipe (Addendum C)

| Item | Value |
|---|---|
| Model | timm `resnet50.a1_in1k`; verified file SHA-256 `773525d5…41bb`; new 1-logit head under seed 0 |
| Train | Full HAM TRAIN, 6938 images, `aug_std_v1` |
| Validation | Full HAM VAL, 1529 images, `prep_v1_cached256` |
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4 |
| Loss | BCE, `pos_weight` 7.8608 from TRAIN |
| Batch / workers | 16 / 0 |
| Precision | FP32, no AMP |
| Epochs | 8, no early stopping |
| Scheduler | Per-step cosine, `T_max` 3472 |
| Training backend | cuDNN TF32 on, matmul TF32 off, "highest" (recorded) |
| Validation precision | `xai_precision_v1` |

Eight epochs is a bounded, prespecified recipe, not a claim of optimality. Nothing was changed in response to pilot, probe or smoke scores.

## 4. The training run

**Run:** `runs/main_train/resnet50_s0_20260927T155924Z/`. It holds the checkpoints for epochs 1–8 with their sidecars, `latest.json`, `best.json`, the logs, per-epoch validation outputs, the selection decisions and the receipt. A model card is at `runs/stage3a/MODEL_CARD_resnet50_s0_v1.md`.

**Time and memory (measured)**
- Wall-clock: **565.6 s** (9.4 min). In-process time: 561.7 s.
- Epochs took 67.1–71.6 s each, rising slightly over the run.
- Each epoch comprised:
  - training loop: 61.5–65.6 s, of which step logging (input SHA-256 + fsync) took about 20 s;
  - full validation pass: 4.6–5.0 s under highest FP32;
  - checkpoint commit: about 1 s.
- Throughput: 106–113 samples/s.
- Peak CUDA memory: 1718 MiB allocated, 1922 MiB reserved.
- Build (model and datasets): 1.3 s.
- Disk: 2.2 GB (8 checkpoints).
- GPU temperature: 52 °C in an idle snapshot before the run, 64 °C after. No samples were taken during the run.

**Checkpoint selection (HAM VAL AUROC per epoch)**
- 0.8701, 0.8882, 0.8815, 0.8924, 0.8937, 0.8912, 0.8948, **0.8960**.
- Best epoch: **8**, the highest value, with no tie.
- These are **checkpoint-selection results on the selection data, not independent test performance.**
- No threshold was computed.

## 5. Post-training verification (`runs/stage3a/post_training_verification.json`)

- The receipt verifies.
- The run's fingerprint equals the pre-launch fingerprint.
- The SHA-256 in `best.json` equals both the checkpoint file and its sidecar.
- The canonical step count is 3472 = 8 × 434. All step losses were finite.
- **In a fresh process**, the selected checkpoint was reloaded and the full HAM VAL was re-evaluated. The record IDs are identical, the **maximum per-image probability difference is 0.0**, and the AUROC is identical (0.8960).
- **Data isolation:**
  - every training step used only HAM `train` records;
  - validation used only HAM `val` records (1529, of which 160 melanoma);
  - no HAM TEST, Derm7pt, PAD or DDI data were read by the training or validation code.

## 6. Files and integrity

**Changed**
- `xai_controls.py`, `xai_control_diag.py`, `trainer.py`, `config.py`.
- `test_xai_controls.py`, updated for the composite keys and with new strictness tests.
- Pre-change copies are in `runs/stage3a/`.

**New**
- Tests: `tests/test_best_selection.py`.
- Protocol and configs: Addendum C, `configs/train_main_resnet50_s0_v1.json`, `configs/main_ham_val_full_v1.csv`.
- Run records: the run directory, and in `runs/stage3a/` the model card, pre-launch fingerprint, post-training verification, stdout, system snapshots, wall-clock timestamps and integrity evidence.

**Integrity:** 431 files before, 434 after.
- **0 changed, 0 removed.**
- The only additions are the addendum, the config and the validation list.
- The earlier runs, checkpoints and caches are unchanged. For the cache `.npy` files, size and modification time were compared.
- No `__pycache__` was created.

## 7. Next (not started)

1. The real-model XAI pilot on the Derm7pt XAI sample, including the production IG m decision.
2. The v1 threshold (02_rev2 §1.1) on HAM VAL, and the RQ1 inference.
3. Seeds 1–2 and other architectures, each needing separate authorisation.

**Stage 3A is complete.**
