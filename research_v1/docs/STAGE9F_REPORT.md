# Stage 9F Report — Nine Main Heavy-Augmentation Training Runs

Date: 2026-10-04. Environment: `research_v1/.venv_model`.

**Scope:** training and fresh-process verification only. No threshold selection, no RQ1/RQ2/external evaluation and no XAI. The `aug_heavy_v1` recipe and `augment.py` are unchanged. No source module changed.

## Status

| Item | Result |
|---|---|
| Configs | 9 frozen configs, `configs/train_main_heavy_{r50,effb3,dn121}_s{0,1,2}_v1.json`. Compared with the matching `train_main_*_v1.json`, the only differences are `versions.aug_version` (std → heavy), `description` and two supporting references (Addendum K, heavy spec) (`runs/stage9f/recipe_comparison_std_vs_heavy_main.json`). Membership, weights, seeds, sampler seed 20260927 + s, preprocessing, loss, pos_weight, lr, weight decay, batch, workers, FP32, 8 epochs, cosine-per-step scheduler, backend, eval precision and selection rule are all identical |
| Preflight | `runs/stage9f/preflight.json`, ALL_OK (§1) |
| Training | **9/9 TRAIN_EXIT 0**: 1 attempt each, 3472 steps, all losses finite, latest epoch 8 |
| Verification | **9/9 VERIFY_EXIT 0, all_pass.** Fresh-process reload max \|Δp\| = 0.0 (tolerance 1e-6, frozen in `configs/stage9f_verification_v1.json`) |
| Wall-clock | 18:10:52 → 20:14:44 local, **2 h 04 min** for all 9 runs including verification. The projection was about 2.0 h |
| Integrity | 2680 → 3005 files: **0 changed and 0 removed.** The cache `.npy` files are unchanged (§5) |

## 1. Preflight (CPU, before launch)

- **Stage 9E resume comparisons:** PASS for r50, effb3 and dn121.
- **GPU:** idle; no other Python processes were running.
- **Disk:** 165 GB free.
- **Bindings, per cell:**
  - architecture, seed and weight SHA-256 match;
  - the training dataset recipe is `aug_heavy_v1`;
  - all backbone tensors verified (318 / 572 / 725).
- **Head initialisation:**
  - The initial head SHA-256 after `seed_everything(seed)` equals the corresponding standard run's recorded head in **9/9** cells.
  - For ResNet-50, the standard runs were built with `smoke_train.build_resnet50` and these with `arch_build.build_timm`. The heads are still identical, which was checked rather than assumed.
- **pos_weight:** identical to the standard run in 9/9 cells.
- **Data:**
  - TRAIN 6938 / VAL 1529, with splits `train` / `val` only.
  - Record overlap 0; lesion-cluster overlap 0; non-HAM records 0.
- **Expected steps:** 3472 (434 × 8), equal to the standard runs' logged steps.
- **Hashes:** the config, source and input hashes and the fingerprint digest are recorded for each cell.

## 2. Results per cell (`runs/stage9f/stage9f_summary.json`, `model_cards_heavy.md`)

HAM VAL AUROC is used **only for checkpoint selection**.

| Cell | Selected epoch | VAL AUROC (selected) | Checkpoint | Wall s | Mean epoch s | Peak alloc MiB | Run-dir disk |
|---|---|---|---|---|---|---|---|
| ResNet-50 s0 | 4 | 0.8652 | `e49f9da6…` | 720 | 89.5 | 1718 | 2.26 GB |
| ResNet-50 s1 | 8 | 0.8637 | see summary | 756 | 94.0 | 1718 | 2.26 GB |
| ResNet-50 s2 | 8 | 0.8698 | see summary | 718 | 89.3 | 1718 | 2.26 GB |
| EfficientNet-B3 s0 | 6 | 0.8640 | see summary | 854 | 106.1 | 2865 | 1.04 GB |
| EfficientNet-B3 s1 | 8 | 0.8605 | see summary | 866 | 107.8 | 2865 | 1.04 GB |
| EfficientNet-B3 s2 | 8 | 0.8630 | see summary | 904 | 112.4 | 2865 | 1.04 GB |
| DenseNet-121 s0 | 8 | 0.8666 | see summary | 858 | 106.8 | 2163 | 0.68 GB |
| DenseNet-121 s1 | 7 | 0.8639 | see summary | 822 | 102.2 | 2163 | 0.68 GB |
| DenseNet-121 s2 | 8 | 0.8758 | see summary | 796 | 99.1 | 2163 | 0.68 GB |

**DenseNet-121 s1 selection:** epoch 7 has AUROC 0.863934 and epoch 5 has 0.863870. Epoch 7 is strictly higher, so the tie rule did not apply.

**Cost against the projection**
- ResNet-50 and DenseNet-121 are within ±5% of the projection.
- EfficientNet-B3 took 7–13% longer than projected: 854–904 s against about 799 s.
- Total disk: about 11.9 GB; free space afterwards is 153 GB.
- GPU temperature: 56–76 °C (249 samples).

**Per-epoch evidence:** validation predictions (`val_epoch*.json`), `selection_decisions.jsonl` and all 8 checkpoints are kept in every run dir.

## 3. Verification (`runs/stage9f/verify_*.json` and the aggregate)

All of the following checks passed in all 9 cells:
- the file, sidecar and `best.json` hashes are equal;
- architecture, seed and selected epoch match;
- the checkpoint's embedded config says `aug_version = aug_heavy_v1`;
- the latest pointer is epoch 8, and the selected checkpoint was compared with it;
- the HAM VAL record order and labels are identical to the saved ones;
- reloaded probabilities differ by 0.0, and the recomputed AUROC equals the saved one;
- all step records are HAM TRAIN;
- the first-step lr is 3e-4 (fresh start);
- the receipts verify;
- the head at run start equals the standard head.

**No continuation from another checkpoint:**
- Each run has one attempt, which started at epoch 0 from the pinned weights.
- None continued from a standard, smoke, pilot or other-seed checkpoint.

## 4. Updated 18-cell RQ3 inventory (`runs/stage9f/rq3_inventory_18cell.json`)

| Strategy | ResNet-50 | EfficientNet-B3 | DenseNet-121 |
|---|---|---|---|
| `aug_std_v1` (s0–s2) | trained, verified and evaluated | trained, verified and evaluated | trained, verified and evaluated |
| `aug_heavy_v1` (s0–s2) | **trained and verified, not yet externally evaluated** | same | same |

**No winning strategy is declared.** The seed-0 standard ResNet-50 remains the primary model.

## 5. Integrity (`runs/stage9f/integrity_comparison.json`)

**Snapshot timing and scope**
- The before snapshot (2680 files) was taken at the start of Stage 9F, before the 9 configs, the tolerance file and the runs were created.
- Its scope now also includes `rq3h_smoke`, `rq3h_pilot` and `stage9e`.

**Result**
- 3005 files after: **0 changed and 0 removed.**
- Added, all expected:
  - 9 configs and `stage9f_verification_v1.json`;
  - 9 × 35 run files in `runs/main_train/rq3h_*_9f`.
- `runs/stage9f/` itself is outside the scope.

## 6. Files

**Configs:** `configs/train_main_heavy_*_v1.json` (9) and `configs/stage9f_verification_v1.json`.

**Runs:** `runs/main_train/rq3h_{arch}_s{seed}_9f/` (9).

**`runs/stage9f/`:**
- launcher `run_main_heavy.bat` and `status.txt`;
- `train_*.txt`, `verify_*.txt` and `verify_*.json`;
- `preflight.py` and `preflight.json`;
- `aggregate.py`, `stage9f_summary.json`, `model_cards_heavy.md` and `rq3_inventory_18cell.json`;
- `gpu_samples.csv`;
- the recipe comparison and the integrity files.

## 7. Not done (requires separate authorisation)

Thresholds for the heavy models; HAM TEST, PAD and DDI evaluation; the RQ2 perturbation sweep; and XAI.

**No cells are unfinished.**

**Stage 9F is complete.**
