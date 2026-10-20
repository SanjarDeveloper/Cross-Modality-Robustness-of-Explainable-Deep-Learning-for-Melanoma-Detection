# Stage 9E Report — Heavy-Augmentation Smoke/Resume and Timing Pilots

Date: 2026-10-04. Environment: `research_v1/.venv_model`.

**Scope:** heavy-augmentation (`aug_heavy_v1`) smoke/resume checks and 2-epoch timing pilots only. No main heavy training and no external or test evaluation. The `aug_heavy_v1` recipe is unchanged; `augment.py` SHA-256 still starts `df6474a3…`.

**Session note:** this stage ran across two sessions because the first was stopped by an agent usage limit. The smoke launcher ran to completion on its own (`ALLDONE`, all exit codes 0). Nothing was rerun or duplicated.

## Status

| Item | Result |
|---|---|
| Clarification of Stage 9D | `STAGE9D_CLARIFICATION_NOTE_v2.md` supersedes v1. v1 is kept byte-identical (SHA-256 `23feb2c6…`, recorded in `runs/stage9e/clarification_note_v1_sha256.txt`). v1 wrongly labelled 158 images/s as a standard figure; it is the heavy recipe's second repeat |
| Configs | 6 frozen configs: `configs/rq3h_{smoke,pilot}_{r50,effb3,dn121}_v1.json`. Their differences from the standard configs are only `aug_version`, `description`, the supporting-input references and, for the ResNet pilot, `purpose` / `epochs = 2` (`runs/stage9e/recipe_comparison_std_vs_heavy.json`) |
| Tolerances | `configs/stage9e_tolerances_v1.json`, frozen before execution. These are the Stage 2C2 values, unchanged |
| Tests | New `tests/test_stage9e_heavy_configs.py`, **4/4 pass**. No shared module changed, so the full suite was not rerun |
| Smoke/resume | **PASS for all 3 architectures, with every difference exactly 0** |
| Timing pilots | **3/3 exit code 0.** Receipts verify, all 868 step losses are finite, the first-step learning rate is 3e-4, and the pinned backbone was verified |
| Integrity | 0 changed and 0 removed (§6) |

## 1. Configurations

**Recipe**
- `aug_heavy_v1` is the only strategy change.
- The split, preprocessing, loss, pos_weight, learning rate 3e-4, weight decay 1e-4, batch size 16, workers 0, precision, seed 0 and sampler seed are all identical to the standard configs.

**Initialisation**
- Each pilot starts from the original pinned pretrained weights (backbone tensors verified: ResNet-50 318, EfficientNet-B3 572, DenseNet-121 725).
- Each pilot gets a fresh head, a fresh AdamW optimiser and a fresh per-step cosine scheduler over 2 epochs.
- No smoke checkpoint was used.

**No silent fallback.** A test showed that `arch_build.build_timm` passes `recipe = "aug_heavy_v1"` to the training dataset; the probe dataset stays in eval mode. Each `run_start.json` records `aug_version = aug_heavy_v1`.

## 2. Smoke/resume (`runs/stage9e/resume_compare_{r50,effb3,dn121}.json`)

**Design:** the frozen smoke subset, 2 epochs and 16 steps per epoch.
- **U** ran both epochs without interruption.
- **R** ran epoch 1, then resumed in a fresh process (`--stop-after-epoch 1`, then `--resume`).

| Arch | Steps (record order, x_digest, lr) | Epoch-1 checkpoint: optimiser / scheduler / RNG | Final params and buffers | Probe probability max diff | Probe AUROC U = R | Status |
|---|---|---|---|---|---|---|
| ResNet-50 | 32 = 32, identical | 0.0 / equal / equal | 0.0 (0 of 320 tensors differ) | 0.0 | 0.7354 | PASS |
| EfficientNet-B3 | 32 = 32, identical | 0.0 / equal / equal | 0.0 (0 of 574) | 0.0 | 0.6865 | PASS |
| DenseNet-121 | 32 = 32, identical | 0.0 / equal / equal | 0.0 (0 of 727) | 0.0 | 0.8223 | PASS |

**Other checks:** fingerprints are equal, the restored RNG digest equals the saved one, there were no warnings, and there were no failed attempts.

**Augmentation inputs on resume:** the heavy augmentation outputs (`x_digest`) after resume are identical to the uninterrupted run at every step.

## 3. Timing pilots (`runs/rq3h_pilot/{r50,effb3,dn121}/`)

**Setup**
- 2 epochs on full HAM TRAIN (6,938 images, 434 steps per epoch), with a full HAM VAL pass after each epoch.
- Run sequentially on the GPU (17:49:42–17:59:52 local time), with no other GPU work running.
- 42 GPU samples at 15-second intervals.

| Arch | Wall-clock (2 epochs) | Setup* | Epoch s (ep1 / ep2) | Train loop s, incl. logging | Step logging s | Val s | Checkpoint s | Samples/s | Peak alloc / reserved MiB | Run dir |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 188.6 s | about 3.9 | 93.9 / 90.8 | 87.5 / 85.2 | 23.4 / 23.2 | 5.3 / 4.7 | 0.9 / 0.9 | 79–81 | 1718 / 1922 | 540 MB (each checkpoint 283 MB) |
| EfficientNet-B3 | 202.6 s | about 3.8 | 99.7 / 99.2 | 93.6 / 93.4 | 19.2 / 20.1 | 5.4 / 5.1 | 0.5 / 0.6 | 74 | 2865 / 3084 | 247 MB (129 MB) |
| DenseNet-121 | 207.6 s | about 3.6 | 102.0 / 102.0 | 95.7 / 96.4 | 17.0 / 18.2 | 5.8 / 5.1 | 0.4 / 0.4 | 72 | 2163 / 2504 | 162 MB (84 MB) |

\*Setup = wall-clock minus the summed epoch times: model build, weight verification and dataset construction.

**GPU temperature:** 50–73 °C (`runs/stage9e/gpu_samples.csv`).

**Validation AUROC** (diagnostic only; not used for any decision; ep1 / ep2)

| Arch | AUROC |
|---|---|
| ResNet-50 | 0.817 / 0.833 |
| EfficientNet-B3 | 0.754 / 0.767 |
| DenseNet-121 | 0.813 / 0.833 |

**Measured cost against the standard recipe**

These are same-hardware per-epoch figures from earlier runs. The standard ResNet-50 reference is the first 2 epochs of the main run, which used an 8-epoch scheduler, so that comparison is approximate.

| Arch | Standard epoch s | Heavy epoch s | Ratio | Extra train-loop s per epoch |
|---|---|---|---|---|
| ResNet-50 | about 67.5 | 92.3 | **1.37** | about +24.6 |
| EfficientNet-B3 | 84.2 | 99.4 | **1.18** | about +15.1 |
| DenseNet-121 | 85.7 | 102.0 | **1.19** | about +16.0 |

The Stage 9D projection was about +15–20 s per epoch. That held for EfficientNet-B3 and DenseNet-121; ResNet-50 is higher at +25 s.

**Memory:** peak GPU memory is identical to the standard runs, because augmentation runs on the CPU.

## 4. Projection for the 9 eight-epoch heavy cells (ESTIMATE, not measured)

**Basis:** 8 × the measured heavy epoch time plus about 4 s of setup. The cosine schedule length does not change the per-step cost.

| Arch | Per run (estimated) | 3 seeds |
|---|---|---|
| ResNet-50 | about 742 s (12.4 min) | about 37 min |
| EfficientNet-B3 | about 799 s (13.3 min) | about 40 min |
| DenseNet-121 | about 820 s (13.7 min) | about 41 min |
| **Total training** | | **about 2.0 h**, plus a fresh-process verification for each run (a few minutes in total) |

**Disk:** 8 checkpoints per run. Per seed: ResNet-50 about 2.26 GB, EfficientNet-B3 about 1.03 GB, DenseNet-121 about 0.68 GB. Total about **12 GB** for 9 runs. Free space is 165 GB.

**Caveat:** this estimate assumes the same machine, no competing load and similar thermal conditions.

## 5. Provenance correction (from Stage 9D)

Verified against `review_and_timing.json`:

| Recipe | Images/s (repeat 1, repeat 2) |
|---|---|
| Standard | 212.9, 301.3 |
| Heavy | 154.3, 158.2 |

- Mean-time ratio: **1.597**.
- Second-repeat ratio: 301.3 / 158.2 ≈ **1.91**.
- Neither ratio is a training slowdown. The measured slowdown is 1.18–1.37× per epoch (§3).
- The "10–30× stronger" GaussNoise claim is withdrawn.
- API keyword compatibility does not establish the historical environment.

## 6. Integrity (`runs/stage9e/integrity_comparison.json`)

**Limit of the baseline:** `integrity_before.json` (2510 files) was taken **after** the six `rq3h_*` configs and `STAGE9D_CLARIFICATION_NOTE_v1.md` had been created. It shows changes since that point only. It cannot prove the state before those files were created.

**Since that snapshot**
- 2512 files after; **0 changed and 0 removed**.
- Added (both intentional): `configs/stage9e_tolerances_v1.json` and `docs/STAGE9D_CLARIFICATION_NOTE_v2.md`.
- The cache `.npy` files are unchanged.

**Older protected artifacts**
- All **2489** files in the Stage 9D after-snapshot are hash-identical now, with 0 changed and 0 missing.
- The files new since Stage 9D are:
  - the Stage 9E configs and notes;
  - Stage 9D artifacts that were outside that snapshot's scope, or written after it (`STAGE9D_REPORT.md`, `runs/stage9d/*`, `aug_heavy_review_9d/*`).
- The Stage 9D review receipt verifies.

**Intentional edits:** none to existing files. The v1 → v2 correction was made by adding a new file; v1 was not edited.

**Outside the snapshot scope:** the new run dirs `runs/rq3h_smoke/`, `runs/rq3h_pilot/` and `runs/stage9e/`, and the new test file.

## 7. Files

**Configs:** `configs/rq3h_*_v1.json` (6) and `configs/stage9e_tolerances_v1.json`.

**Documents:** `docs/STAGE9D_CLARIFICATION_NOTE_v1.md` (superseded, kept) and `docs/STAGE9D_CLARIFICATION_NOTE_v2.md`.

**Test:** `tests/test_stage9e_heavy_configs.py`.

**Runs:** `runs/rq3h_smoke/{r50,effb3,dn121}/{U,R}` and `runs/rq3h_pilot/{r50,effb3,dn121}`, with their receipts.

**`runs/stage9e/`:**
- launchers `run_smoke.bat` and `run_pilots.bat`;
- status files `smoke_status.txt` and `pilots_status.txt`;
- per-process logs;
- `resume_compare_*.json`;
- `gpu_samples.csv`;
- `recipe_comparison_std_vs_heavy.json`;
- the integrity files.

## 8. Remaining before the 9 main heavy runs (not started)

1. Explicit authorisation from you.
2. Create the 9 main configs (`train_main_*_s{0,1,2}` with only the `aug_heavy_v1` change), the launcher and the fresh-process verification for each run.
3. Then evaluation, as in Stages 9C1 and 9C2, which is separately gated.

**Stage 9E is complete.**
