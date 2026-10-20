# Stage 2C3 Report — Bounded ResNet-50 Throughput Pilot

Date: 2026-09-27. All work in this stage used `research_v1/.venv_model`.

**Scope.** This is a training-time pilot for **ResNet-50 only**. It is **not** the complete research compute pilot: XAI and the other architectures were not measured. No main training, RQ analysis, heavy augmentation, AMP, test-set evaluation or XAI was run. The pilot-trained weights are **not** a research model.

## Status

| Item | Result |
|---|---|
| GPU crash/recovery check (256-image smoke contract) | **PASS** — bit-identical to the uninterrupted reference |
| Full-split timing pilot (2 epochs, 6938 images) | Completed: exit 0, no OOM, no non-finite losses, no nondeterminism warnings; 140.8 s wall-clock |
| Separate full HAM-VAL pass timing | 2.82 s for 1529 images; no scores stored, no metrics computed |
| Full test suite in `.venv_model` | **84 / 84 pass** |
| Protected inputs | 0 changed, 0 removed |

## 1. Code change (instrumentation only)

`rv1/trainer.py` now also records the following. The pre-change copy is kept at `runs/stage2c3/trainer_before_2c3.py.txt`.

- model initialisation, weight loading and dataset construction time (`build_seconds…` in `attempts.jsonl`);
- for each epoch:
  - training-loop time;
  - the part of it spent on step logging (per-step input SHA-256 plus fsync);
  - samples per second;
  - probe time;
  - checkpoint serialize + hash + commit time;
  - peak CUDA allocated **and reserved** memory.

GPU sections are timed with `torch.cuda.synchronize()`.

Training behaviour, provenance and durability are unchanged. Per-step hashing and fsync stay **enabled**, and the reported training times **include** them.

`config.py` gains the purpose `train_pilot`, which requires the training block. The trainer tests were re-run after this change (15/15) before any GPU work.

## 2. GPU crash/recovery check

Contract: `train_smoke_v1.json` (256 images, E=2, FP32, batch 16).

**Runs**
- **U3** `runs/train_smoke/U3_20260927T143239Z`: fresh reference run, uninterrupted.
- **C3** `runs/train_smoke/C3_20260927T143239Z`: `RV1_CRASH_AT=step:19`, which kills the process 3 steps into epoch 2. It exited with **code 97**, leaving only the committed `ckpt_epoch1.pt`.
- The run was then resumed in a fresh process as attempt `a2`, mode `resume_from_checkpoint`.

**Comparison** (`runs/stage2c3/gpu_crash_recovery_comparison.json`, using the canonical steps):
- The canonical steps (32) are identical in order, augmentation digests, learning rates and losses.
- The raw log keeps all **35** step lines, including the 3 steps of the crashed attempt.
- Model and buffers: maximum difference **0.0**. Optimizer and scheduler: identical.
- RNG digests are equal, and the restored digest equals the saved one.
- Probe probabilities: difference **0.0**. AUROC difference: 0.0.
- Fingerprints are equal.
- **Status: PASS.**

**Caveats**
- This simulates a process crash. It does not simulate power loss.
- The crashed attempt writes no receipt, because `os._exit` bypasses it. That attempt is recorded in `attempts.jsonl` and `steps.jsonl`.
- The receipts of U3 and of the resume invocation pass `verify_receipt`.

## 3. Timing contract (frozen before starting)

Configuration: `configs/train_pilot_resnet50_v1.json`, config hash `3422f52651ec…`.

| Setting | Value |
|---|---|
| Weights | `resnet50.a1_in1k` (SHA-256 `773525d5…41bb`, verified); newly initialised 1-logit head |
| Precision | FP32 |
| Batch size | 16 |
| `num_workers` | 0 |
| Augmentation | `aug_std_v1` |
| Epochs | 2 |
| Seed | 0 |
| Sampler seed | 20260927 |
| Optimizer | AdamW, lr 3e-4, weight decay 1e-4 |
| Scheduler | Per-step cosine, `T_max` = 868 |
| Training data | **The complete frozen HAM TRAIN split**: `configs/pilot_ham_train_full_v1.csv`, 6938 records in manifest order, **783 melanomas** |
| `pos_weight` | **7.8608** (6155 / 783), computed from this split. The smoke subset's balanced value of 1 was not reused |
| Probe | The trainer's 64-image HAM-VAL probe, required by the trainer; its values were **not** used for anything |

These are pilot settings, not the main-training recipe. No setting or checkpoint was chosen on validation data.

**Runtime settings**
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- `cudnn.deterministic=True`, `cudnn.benchmark=False`.
- Deterministic algorithms on, with `warn_only=True`. No warnings were captured.

**Laptop state, recorded before and after**
- On AC power ("power_plugged=True", battery at 79%).
- GPU idle snapshots before and after the run: 54 °C / P0 before, 63 °C / P4 after, driver 591.91.
- Clock and temperature were **not** sampled during the run.
- The `powercfg` query failed ("Invalid Parameters"), so the Windows power plan was **not recorded**.
- Thermal stability cannot be judged from two epochs.

**Run:** `runs/train_pilot/P1_20260927T143355Z`. Its receipt verifies. The run used about 540 MB of disk for two checkpoints.

## 4. Measured timings

| Quantity | Epoch 1 | Epoch 2 |
|---|---|---|
| Training loop, including step logging (434 steps) | **65.31 s** | **65.19 s** |
| … of which per-step input SHA-256 + fsync | 19.57 s | 20.02 s |
| Samples per second, including logging | 106.2 | 106.4 |
| Probe (64 images) | 0.18 s | 0.18 s |
| Checkpoint serialize + hash + commit | 0.67 s | 0.66 s |
| Whole epoch (training + probe + commit) | **66.22 s** (1.10 min) | **66.09 s** (1.10 min) |
| Peak CUDA memory, allocated / reserved | 1717.8 / 1922.0 MiB | 1717.8 / 1922.0 MiB |

**Other measurements**
- Model initialisation, weight loading and dataset construction: **1.24 s**.
- Run elapsed time, measured inside the process: 136.7 s. Total wall-clock, including interpreter start-up: **140.8 s**.
- Full HAM-VAL pass (1529 images, batch 16, evaluation path): **2.82 s** (542 images/s). Peak memory 292 / 330 MiB (allocated / reserved).

**Observation:** step logging (input hashing + fsync) takes about **30%** of each training epoch. It was left on. Whether to keep per-step input hashing in main training is a later decision about the trade-off between provenance and speed; it was not changed here.

## 5. Arithmetic projection (illustrative scenarios; not guarantees)

**Assumptions**
- The measured per-epoch time holds for later epochs: 66.09–66.22 s.
- A full HAM-VAL pass is run every epoch: +2.82 s.
- Checkpoint time is already included.
- About 9 s per run for start-up and build (wall-clock 140.8 s minus 132.3 s of epochs).
- Same laptop, same settings (FP32, batch 16, `num_workers` 0, step logging on), no other load.
- Eight epochs is an **illustrative** number. It is not a frozen scientific decision.

| Scenario | Arithmetic | Result |
|---|---|---|
| One ResNet-50 run, 8 epochs | 8 × (66.2 + 2.8) s + 9 s ≈ 561 s | **≈ 9.4 min** |
| Three such seeds, run one after another | 3 × 561 s | **≈ 28 min** |
| Sensitivity: if epochs were 25% slower (hypothetical, e.g. thermal throttling or background load) | 8 × (82.8 + 2.8) + 9 ≈ 694 s per run | ≈ 11.6 min per run; ≈ 35 min for 3 seeds |

**Uncertainty**
- There are only two epoch observations, and they are nearly identical (a difference of 0.13 s). They show short-term consistency only.
- Nothing is known about thermal throttling over longer runs, OS or background activity, or disk-cache effects.
- The ±25% row is a what-if, **not** a statistical interval.

**Not covered by these numbers**
- EfficientNet-B3 and DenseNet-121.
- XAI (Grad-CAM, IG, controls and curves).
- Perturbation sweeps, and evaluation on Derm7pt, PAD or DDI.
- AMP, other batch sizes and `num_workers > 0`.
- Cloud GPUs.
- Thesis work and the project as a whole.

The earlier "166 h" estimate is **not** a measured value and is not supported by this pilot.

## 6. Recovery caveat for future cloud (Linux / Vast) use

All recovery tests so far ran on **Windows**. There, `os.rename` refuses to replace an existing file, and the commit protocol relies on that.

On POSIX systems `os.rename` **silently replaces** an existing target. Before any Linux or Vast execution, the following must be re-verified on that platform, together with the recovery tests:
- the checkpoint publication semantics (for example `os.link` + `unlink`, or an explicit existence check);
- the fsync and directory-sync behaviour.

**Portability is not claimed.**

## 7. Files and integrity

**Changed:** `src/rv1/trainer.py` (timing only) and `src/rv1/config.py` (purpose `train_pilot`).

**New:**
- `src/rv1/val_timing.py`;
- `configs/train_pilot_resnet50_v1.json` and `configs/pilot_ham_train_full_v1.csv`;
- run directories `train_smoke/U3_…`, `train_smoke/C3_…` and `train_pilot/P1_…`;
- in `runs/stage2c3/`: comparisons, the timing JSON, stdout, system snapshots, wall-clock timestamps, the integrity snapshots and comparison, and the pre-change trainer copy.

**Integrity:** 320 files before, 349 after.
- **0 changed, 0 removed.**
- Added: the 2 new config files and 27 files inside the two new `train_smoke` run directories.
- The pilot directory lies outside the snapshot scope.
- The cache `.npy` files have the same size and modification time.
- No `__pycache__` was created.
- Earlier receipts and runs were not modified. They may report source differences for `trainer.py` and `config.py`.

## 8. Remaining work (not measured or decided)

- XAI cost.
- The other two architectures and their weight provenance.
- AMP.
- The main recipe: epochs, batch size, precision, logging overhead, and checkpoint selection on HAM val.
- Longer-run thermal behaviour.
- Linux/cloud recovery semantics.
- `aug_heavy_v1`.
- The Secondary B endpoint for adapt_val (not estimable).

**Stage 2C3 is complete. Main training was not launched.**
