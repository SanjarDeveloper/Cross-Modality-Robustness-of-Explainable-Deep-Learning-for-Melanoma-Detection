# Stage 2C2 Report — Bounded Training Smoke Test and Epoch-Boundary Resume

Date: 2026-09-27. All work in this stage used `research_v1/.venv_model`.

## Status (reported separately)

| Track | Result |
|---|---|
| **CPU resume** (toy CNN with BatchNorm and Dropout; fresh subprocess per arm) | **PASS**. Parameters, buffers and probe probabilities are within 1e-6. Step sequence, scheduler and RNG digest are identical. Negative controls behave as expected (§5) |
| **GPU FP32 resume** (ResNet-50, 256-image subset, E=2) | **PASS**. The two arms are **bit-identical**: all 320 parameter and buffer tensors, optimizer state, scheduler, RNG digest and probe probabilities differ by 0.0 |
| **AMP** | **Not implemented and not tested.** Nothing about AMP is claimed |
| Full suite in `.venv_model` | **74 / 74 pass** (69 earlier tests + 5 new) |
| Protected inputs | 0 changed, 0 removed |
| Main training / RQ experiments | Not started, and not authorised |

The loss falling across a small smoke run is **not** a scientific result. The probe AUROC (0.673) is recorded only as a quantity to compare between arms. It was not used for any decision.

## 1. The frozen contract (fixed before any training run)

`configs/train_smoke_v1.json` (config hash `be0f0c6f…`), with `configs/smoke_train_subset_v1.csv` and `configs/smoke_val_probe_v1.csv`. These lists were created by `rv1/smoke_setup.py`, and neither was changed after the runs.

**Train subset**
- HAM `split == train`.
- For each class, records are ordered by `(stable_seed("smoke_subset_v1", record_uid), record_uid)`.
- Selected: **128 melanoma + 128 non-melanoma = 256**.

**Probe**
- HAM `split == val`.
- Same ordering rule, using `"smoke_probe_v1"`.
- Selected: **32 + 32 = 64**.
- Used only to compare the two resume arms. No lesion is shared with the subset (asserted).
- HAM test, Derm7pt, PAD and DDI were not used.

**Settings.** Each setting is labelled as legacy (inherited from legacy `code/train.py`) or as a **smoke-only** choice. None of these is fixed as the main-training recipe.

| Setting | Value | Source |
|---|---|---|
| Model | timm `resnet50`, 1 logit | Legacy |
| Optimizer | AdamW, lr 3e-4, weight decay 1e-4 | Legacy |
| Scheduler | CosineAnnealingLR stepped per batch, `T_max` = E × steps per epoch = 2 × 16 = 32 | Legacy form |
| Loss | `BCEWithLogitsLoss`; `pos_weight` = n_neg/n_pos over the subset = 1.0; float targets of shape (B,) | Legacy |
| Masks | Not loaded; not part of the input or the loss | — |
| Epochs | E = 2 | Smoke |
| Batch size | 16 (conservative for 8 GB) | Smoke |
| Precision | FP32 | Smoke; legacy used AMP |
| Workers | 0 | Smoke; the worker paths were verified in Stage 2C1 |
| Sampler seed | 20260927 | Smoke |
| Training seed | 0 | Smoke |
| Augmentation | `aug_std_v1` | — |
| Probe preprocessing | Evaluation path | — |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8`, set before torch was imported | — |
| Best-model selection | **Disabled** (only the resumable "latest" exists) | Smoke |

**Environment changes (inside `.venv_model` only)**
- Installed `timm==1.0.28`, the recorded version.
- It pulled `huggingface_hub 2.0.0`. That was then **pinned back to 0.36.2**, the version recorded globally, to avoid unrecorded drift.
- The pinned preparation stack was unchanged: additions only (see the freezes before and after in `runs/stage2c2/`).
- `pip check` passes. The global environment and `.venv_prep` were not modified.

## 2. Pretrained weights and their provenance

Full record: `runs/stage2c2/weights/weight_provenance.json`.

**Identity and source**
- timm 1.0.28 resolves `resnet50` to the default tag **`resnet50.a1_in1k`**.
- Hugging Face repository: `timm/resnet50.a1_in1k`, commit **`767268603ca0cb0bfe326fa87277f19c419566ef`**.
- File: `model.safetensors`, downloaded with `huggingface_hub` from `https://huggingface.co/timm/resnet50.a1_in1k/resolve/<commit>/model.safetensors`.
- Stored in a project-local cache under `runs/stage2c2/weights/`.
- The legacy GitHub URL listed in the timm config was **not** used.

**Integrity**
- SHA-256 `773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb`. This equals the LFS SHA-256 reported by the Hub.
- The config records the hash, and `build_resnet50` refuses a file whose hash differs.

**Loading and verification**
- The model is loaded with `pretrained=True` and `pretrained_cfg_overlay={"file": <that exact file>}`.
- All **318 backbone tensors** were checked to be exactly equal to the file. If the weights had not loaded, this check would raise an error; there is no silent fallback to random initialisation.
- Licence: apache-2.0 (from the timm config).

**Classifier head**
- The ImageNet `fc` layer is dropped because `num_classes=1`.
- The new head uses the timm/torch `nn.Linear` default initialisation, drawn after `seed_everything(train_seed)`. Its hash is recorded in `run_start.json`.

**Label convention.** A sigmoid of the logit gives P(melanoma). `label_mel = 1` means melanoma; `0` means non-melanoma.

**Note.** The pretrained config lists bicubic interpolation and `crop_pct` 0.95. The v1 inputs instead use `prep_v1_cached256` (INTER_AREA → bilinear, full frame, no crop), as the legacy pipeline did. This is a documented difference, not a change made here.

The old research checkpoint was **not** used for initialisation.

## 3. Training loop and checkpoints (`rv1/trainer.py`, `rv1/smoke_train.py`)

**Training step**
- `model.train()` during training; `model.eval()` with `torch.no_grad()` for the probe.
- Every step logs: epoch, global step, sample count, loss, learning rate used, a finite-loss flag, the record order, and a SHA-256 of the batch input tensor.
- A non-finite loss raises an error. None occurred.

**Checkpoints**
- Written only at **epoch boundaries**, to immutable `ckpt_epoch<k>.pt` files (temporary file, then `os.replace`). After each checkpoint, `latest.json` is rewritten atomically.
- Each checkpoint contains:
  - model **parameters and buffers**;
  - optimizer and scheduler state;
  - `epoch_completed` and `global_step`;
  - the actual Python, NumPy, Torch-CPU and Torch-CUDA RNG states, plus their digest;
  - the sampler policy and the next epoch's generator state;
  - the best-selection setting (disabled);
  - the full config;
  - a **resume fingerprint**: config hash, source hashes, and the hashes of the subset, probe, weights, environment freeze, manifests and cache inputs.
- Operational arguments are **not** part of the fingerprint: `--run-dir`, `--stop-after-epoch` and `--resume`.

**Resume**
1. `seed_everything`.
2. Build the model, optimizer and scheduler.
3. Load their saved states.
4. **Then restore the saved RNG states.**
5. Check that the restored RNG digest equals the saved one, and that the recreated sampler state equals the stored one.

A fingerprint mismatch raises `ResumeMismatch` and lists the differing parts. Starting a fresh run in a directory that already has checkpoints is refused.

**Mid-epoch interruption.** Only resume at an **epoch boundary** is supported. If the laptop stops in the middle of an epoch:
- the work of that partial epoch is lost;
- resume restarts from the last completed epoch's checkpoint and repeats the interrupted epoch from its beginning, with the same seeds, so the result is the same as an uninterrupted run;
- the loss is at most one epoch of compute. Exact recovery from the middle of an epoch is not provided.

Every invocation writes its own receipt. All 3 GPU receipts pass `verify_receipt`.

## 4. GPU FP32 resume verification

Both arms used the same config, initialisation, total E = 2 and `T_max` = 32.

- **Run U** (`runs/train_smoke/U_20260927T140236Z`): 2 epochs, uninterrupted.
- **Run R** (`runs/train_smoke/R_20260927T140236Z`): stopped after epoch 1 and **exited the process**; then resumed **in a fresh process** from `ckpt_epoch1.pt` and ran epoch 2.

Results (`runs/stage2c2/resume_comparison_gpu_fp32.json`):

| Check | Result |
|---|---|
| Record order, augmentation-input digests and learning rate at each of the 32 steps | Identical |
| Loss difference | 0.0 |
| Epoch-1 checkpoints: model, optimizer, scheduler, RNG digest, global step | Identical (0.0; 16 = 16) |
| RNG digest restored in the fresh process vs. saved digest | Equal |
| Final global step / final learning rate | 32 / 32; 0.0 / 0.0 |
| Final optimizer state / scheduler / RNG digest | Identical |
| Parameters and buffers (320 tensors) | Maximum absolute difference **0.0**; no tensor differs; none missing |
| Probe probabilities | Maximum absolute difference **0.0** (tolerance 1e-3) |
| Probe AUROC | 0.6728515625 vs 0.6728515625, difference 0.0 (tolerance 1e-3); computed with scipy `mannwhitneyu` |
| Fingerprints | Equal |
| Captured nondeterminism warnings (`use_deterministic_algorithms(True, warn_only=True)`, all warnings recorded) | **None** in either arm |

**Measured (smoke-only; not to be extrapolated):**
- U: 11.2 s in total (epochs took 4.4 s and 3.0 s).
- R: 8.2 s up to the stop, then 8.1 s for the resume.
- Peak CUDA allocation: about 1718 MiB.
- No out-of-memory events and no non-finite values. The 30-minute limit was never approached.
- Disk: about 1.1 GB for the four checkpoints, plus 98 MB of weights.

## 5. CPU toy tests (`tests/test_trainer.py`, 5 tests)

The toy model is Conv → BatchNorm → ReLU → Dropout(0.5) → Linear, trained on synthetic data with augmentation noise keyed by seed and epoch. It uses the same checkpoint machinery, and every arm runs in a fresh subprocess.

1. The uninterrupted run and the resumed run agree: parameters **and** BatchNorm buffers within 1e-6, probe probabilities within 1e-6, and identical step records, scheduler and RNG digest.
2. The epoch-1 checkpoint holds `epoch_completed` 1, `global_step` 5 and scheduler `last_epoch` 5, together with every required field. The restored RNG digest equals the saved one.
3. **Negative controls:**
   - (a) skipping the RNG restore is detected: "RNG state not restored exactly";
   - (b) with both the restore and the check disabled (test-only patch), the final parameters **diverge**. So Dropout really consumes RNG, and restoring the RNG state matters.
4. A resume is rejected after a changed input file (`input:m/subset.csv`) and after a changed scientific config (`config_sha256`). A fresh start into a directory that already has checkpoints is refused.
5. The fingerprint does not depend on operational arguments.

## 6. Tests

```bash
research_v1\.venv_model\Scripts\python.exe -B -W ignore -W error::ResourceWarning -m unittest discover -s research_v1/tests -p "test_*.py"
```

→ **Ran 74 tests in 83.6 s — OK** (single environment).

## 7. Files and integrity

**New code:** `src/rv1/trainer.py`, `smoke_train.py`, `smoke_setup.py` and `resume_compare.py`. `config.py` gains the purpose `train_smoke` and a validated `training` block; earlier configs are unaffected (tested).

**New tests:** `tests/test_trainer.py`.

**New configs:** `train_smoke_v1.json` and the subset and probe lists.

**New runs:**
- `runs/train_smoke/{U,R}_20260927T140236Z/`: checkpoints, `steps.jsonl`, `epochs.jsonl`, `warnings.jsonl`, `run_start.json`, `resume_events.jsonl` (R only) and receipts;
- `runs/stage2c2/`: weights, freezes, the comparison and the integrity evidence.

**Integrity:** 246 files before, 249 after. **0 changed, 0 removed.** The 3 added files are the expected new configs and lists. The size and modification time of the cache `.npy` files are unchanged. `.venv_prep` and the global environment are unchanged. No `__pycache__` was created. Earlier receipts were not rewritten; they may show source differences for `config.py`.

## 8. Not yet tested

1. **AMP** (autocast + GradScaler): save/restore and tolerances. Not implemented.
2. Resume with `num_workers > 0` inside training. Workers were verified for data in Stage 2C1, but not in combination with resume.
3. Full-dataset throughput and the compute pilot for the 3 architectures (EfficientNet-B3 and DenseNet-121 are not built yet). **No runtime is extrapolated from this run.**
4. Whether determinism holds for other architectures and cuDNN algorithm choices, at larger batch sizes, or across driver and library versions.
5. Pretrained-weight provenance for EfficientNet-B3 and DenseNet-121.
6. The final main-training recipe (epochs, batch size, precision, checkpoint selection on HAM val) needs a decision. The smoke choices here are not that decision.
7. Carried forward:
   - the primary model is a newly trained v1 seed 0, and the legacy checkpoint stays separate;
   - `aug_heavy_v1` is deferred until its intended interpretation relative to RQ2 is specified;
   - the Secondary B endpoint for adapt_val is not estimable;
   - the global OpenCV conflict remains.

**Stage 2C2 is complete. No main training was started.**
