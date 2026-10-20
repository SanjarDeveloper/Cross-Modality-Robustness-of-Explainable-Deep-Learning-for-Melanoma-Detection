# Stage 2C1 Report — Unified Model Environment and Torch DataLoader Integration

Date: 2026-09-27

## Status

| Aspect | Result |
|---|---|
| New environment `research_v1/.venv_model` (torch + exactly one OpenCV) | Established. `pip check` passes |
| Full test suite in **one** environment (`.venv_model`) | **69 / 69 pass**: 62 earlier tests + 7 new DataLoader tests |
| Preparation outputs, `.venv_prep` vs `.venv_model` | **161 / 161 digests identical**. The repeated Stage 2B2C sample check gives identical records |
| DataLoader worker/epoch determinism, real HAM train subset | **PASS** (12/12 checks) |
| CUDA smoke check | A tiny int64 GPU operation ran and **matched the CPU result exactly** on the RTX 5060 Laptop GPU |
| GPU readiness for training | **Not established.** Only a single tensor operation ran: no network, no mixed precision, no resume determinism |
| Protected inputs | 0 changed, 0 removed |

No model was instantiated, no weights were downloaded, and there was no inference or training.

## 1. Environment

**Inspection before creating anything**
- 199 GB free on C:.
- The pip cache holds 4.4 GB.
- The only existing project environment was `.venv_prep`, which has no torch. It was kept unchanged: its freeze is identical before and after.
- The global environment was not modified. It still has 4 OpenCV distributions, plus torch 2.11.0+cu128.

**New environment: `research_v1/.venv_model`**
- Created with `python -m venv` from the recorded interpreter (Python 3.10.11).
- `include-system-site-packages = false`. No global `site-packages` path is on `sys.path` (checked).

**Installation**, from official sources only:

1. **pip upgraded inside `.venv_model` only**, from 23.0.1 to 26.2.1. The bundled pip 23.0.1 failed on the PyTorch index. It rejected wheels because of package-name normalisation mismatches (`jinja2` vs `Jinja2`, `typing-extensions` vs `typing_extensions`). The error was recorded and the fix applied only in this environment.
2. `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` from `https://download.pytorch.org/whl/cu128`. This is the same combination the global environment records.
3. The preparation packages, pinned exactly to the `.venv_prep` freeze: `opencv-python-headless 4.11.0.86`, `numpy 2.2.6`, `pillow 12.3.0`, `albumentations 2.0.8`, `albucore 0.0.24` and their dependencies.

Installation took **267 s** of wall-clock time, including downloads.

**Records** (in `research_v1/runs/stage2c1/`)
- `venv_model_freeze.txt` and `venv_prep_freeze_before.txt`.
- `pip check`: "No broken requirements found".
- OpenCV distributions in `.venv_model`: exactly **`opencv-python-headless 4.11.0.86`**.
- `cv2`, numpy, PIL, albumentations, torch and torchvision all import from `.venv_model\lib\site-packages`.
- No demo or XAI packages were installed.

## 2. Unified regression check

**Command, run with `.venv_model` only:**

```bash
research_v1\.venv_model\Scripts\python.exe -B -W ignore -W error::ResourceWarning -m unittest discover -s research_v1/tests -p "test_*.py"
```

→ **`Ran 69 tests — OK`** (42.4 s). This covers the foundation, decode, masks, preprocess, augment and torch_data tests. For the first time, the whole suite ran in a single environment.

For reference, `test_torch_data.py` in `.venv_prep` reports 7 tests **skipped**, because torch is not installed there. They were not run there, so they are not counted as passing there.

**Equivalence of preparation outputs** (`rv1/equiv_digest.py`)
- The same code and the same inputs were run in both environments. The inputs:
  - the 29 Stage 2B2C sample records (evaluation input, plus the 224 mask for HAM);
  - 20 HAM-train records under `aug_std_v1` at (0,0), (0,1) and (2,5), with masks.
- **161 of 161 SHA-256 digests are identical** (`digest_comparison.json`).
- Both environments use cv2 4.11.0, numpy 2.2.6 and albumentations 2.0.8.

**Repeated Stage 2B2C sample check** in `.venv_model`
- New run: `runs/prep_check/20260927T133023Z_942f5ccf_9a9a3698/`. `verify_receipt` → OK.
- The per-record results are **identical** to the final `.venv_prep` run `…130514Z_942f5ccf_0403c63c`, including exact cache reconstruction for 29/29.

## 3. Torch integration (`rv1/torch_data.py`)

**`ManifestDataset`** is a map-style dataset built on the frozen manifest rows.
- **Validation first:** it validates the **full** cache mapping (via `EvalReader`), and only then selects the requested `record_uid` subset. The subset does not need to cover the cache. Unknown or duplicate IDs raise an error.
- **Modes:**
  - `eval` → `aug_none`, i.e. evaluation preprocessing;
  - `train` → `aug_std_v1`, keyed by (`train_seed`, `epoch`, `image_uid`).
- **Items:**
  - `x`: `float32` tensor (3, 224, 224);
  - `label`: `int64`, taken from the record's own `label_mel`;
  - `record`: an unchanged copy;
  - optional `mask`: a bool tensor, used for diagnostics only and not as model input;
  - `mask_flags`.

  `collate` stacks `x` and the labels, and keeps records, masks and flags as lists in batch order.
- **Windows spawn:** the read-only `np.load(mmap_mode="r")` map is opened lazily inside whichever process reads. `__getstate__` removes it, so workers receive only paths, row dicts and scalars. The pickled dataset is under 20 kB (tested), and no cache is loaded into RAM.
- **Epoch handling:** `make_epoch_loader()` does the following:
  - sets `dataset.epoch`;
  - builds a **new** `DataLoader` with `persistent_workers=False`, so fresh workers pickle the updated epoch;
  - uses an explicit `torch.Generator` seeded with `stable_seed(sampler_seed, "sampler", epoch)`;
  - uses `drop_last=False`.

  No shared mutable state is used.

## 4. Worker and epoch results

**Synthetic tests** (`test_torch_data.py`, 7 tests, `.venv_model`)
- `num_workers` 0 and 2 give an identical ordered sequence and identical per-record tensors.
- No record is missing or duplicated when 11 records are split into batches of 4.
- Shuffling is reproducible for a given sampler seed and changes with a different seed.
- Changing the epoch reaches the workers: epoch 1 with workers equals epoch 1 without workers, and differs from epoch 0.
- Returning to epoch 0 reproduces its outputs.
- Per-record content does not depend on the sampler seed.
- `eval` mode does not depend on the seed, epoch or number of workers, and equals `preprocess.to_model_input`.
- Subset selection and pickling behave correctly.
- Types and validation behave correctly.

**Real HAM-train subset** (`rv1/loader_check.py`)
- 16 records ordered by `stable_seed("loader_check_v1", uid)`; batch 5; sampler seed 20260927; `train_seed` 0; masks on.
- Run: `runs/loader_check/20260927T133120Z_7d270eb4_cc7801d6/`. `verify_receipt` → OK.

All checks returned true:

| Check | Result |
|---|---|
| Same records and labels, `num_workers` 0 vs 2 | true |
| Labels stay attached to their own record | true |
| Identical per-record outputs, `num_workers` 0 vs 2, epochs 0 and 1 | true |
| Identical ordered sequence, `num_workers` 0 vs 2 | true |
| Shuffle reproducible with the same sampler seed; changes with a different seed | true |
| Per-record outputs independent of the sampler seed | true |
| Epoch reaches the workers | 16 of 16 records changed between epoch 0 and epoch 1 |
| Returning to epoch 0 reproduces its outputs | true |
| No missing or duplicate records | true |
| Evaluation mode independent of seed, epoch and workers | true |

Sampler order and augmentation content were checked separately: order was compared as a sequence, and content was compared by `record_uid`.

**Timings** (measured machine time)

| Configuration | Time for 16 records |
|---|---|
| `num_workers=0` | 0.09–1.19 s (the first call includes cache warm-up) |
| `num_workers=2` | about 3.0 s, including spawning the worker processes |
| Whole loader check | 17.8 s |

No performance metric was computed, and no validation or test data was used.

## 5. CUDA smoke check

- `torch.cuda.is_available()` = True.
- Device: **NVIDIA GeForce RTX 5060 Laptop GPU**, capability (12, 0), torch CUDA 12.8, cuDNN 91900.
- Operation: the sum of squares of `arange(1e6)` as `int64` on the GPU. Result 333332833333500000, **identical** to the CPU result. No errors.

This confirms only that the installation can execute a CUDA kernel. It does **not** establish readiness for training, performance, AMP behaviour, or determinism of GPU training and resume.

## 6. Files and integrity

**New files**
- Code:
  - `src/rv1/torch_data.py`
  - `src/rv1/equiv_digest.py`
  - `src/rv1/loader_check.py`
- Tests: `tests/test_torch_data.py`.
- Configuration: `configs/loader_check_v1.json`.
- `config.py` gains the purpose `loader_check`. Earlier configurations are unaffected (tested).
- Records in `runs/stage2c1/`:
  - freezes;
  - digest files and `digest_comparison.json`;
  - integrity snapshots and comparison;
  - install timestamps.

**Integrity** (226 files before, 231 after)
- **0 changed, 0 removed.**
- Added, all expected: `configs/loader_check_v1.json` and the new repeated `prep_check` run directory (4 files).
- The size and modification time of the cache `.npy` files are unchanged.
- `.venv_prep` is unchanged, and the global environment is unchanged.
- No `__pycache__` directories were created.
- Earlier receipts were not rewritten. They may show source differences for `config.py`, which was legitimately extended.

## 7. Carried forward and unresolved

- The primary model is a **newly trained v1 seed 0**. The legacy checkpoint stays in the separate legacy-reproduction analysis and is not pooled with new seeds (Addendum A).
- **`aug_heavy_v1` remains deferred.** Overlap between the training corruptions and the RQ2 perturbations is **not inherently invalid**. However, its intended interpretation (for example, "robustness from exposure to similar corruptions" versus an independent generalisation test) must be specified before the RQ3 augmentation experiment.
- **Not yet tested:**
  - training-loop determinism;
  - checkpoint/resume with optimizer, scheduler, scaler and RNG state (03_rev2 §12);
  - AMP and GPU numeric tolerances;
  - model instantiation and pretrained-weight provenance (download source and hash);
  - throughput for the full matrix. The earlier "~5 h" estimate is still unverified; the pilot has not been run.
- `pin_memory` is off, and `num_workers > 2` was not tested. Worker start-up on Windows costs about 2–3 s per epoch loader.
- Still open from earlier stages:
  - the Secondary B endpoint for adapt_val is not estimable;
  - full-cache reconstruction has not been verified (only a sample);
  - the global OpenCV conflict remains. It is avoided by using the project environments.

**Stage 2C1 is complete. No training was started.**
