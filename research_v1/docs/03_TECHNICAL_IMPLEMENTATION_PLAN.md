# 03 — Technical Implementation Plan

Status: plan only. Nothing in this document is implemented yet. Terminology, cohorts and metrics follow `02_RESEARCH_PROTOCOL_V1_DRAFT.md`.

## 1. Principles

1. **Legacy files are read-only.** research_v1 never writes under `code/`, `results/`, `checkpoints/`, `cache/`, `audit/`, `protocol/`, `runs/` or `DATASETS/`.
2. **Per-image outputs come first.** Every aggregate is recomputed from persisted per-image rows by a separate analysis layer.
3. **Everything is explicit and recorded:** preprocessing, metric version, seeds, cohort manifest hash and code hash.
4. **Outputs are immutable.** Each run writes to a new directory. A run is complete only when it has a receipt with output hashes.
5. **Dependencies stay minimal.** Use what is already installed wherever possible.

## 2. Directory layout (proposed)

```
research_v1/
  docs/                      Stage 1 documents (this package)
  src/rv1/
    config.py                frozen dataclass configs + hashing
    seeds.py                 stable_seed(), seed_everything(), RNG state capture
    manifests.py             build/validate cohort manifests
    data.py                  Dataset reading cache/raw per prep version; joint image-mask transforms
    perturb.py               wrappers over legacy simulator ops with explicit rng + geometry
    models.py                timm build, target layers
    train.py                 training loop with full checkpoint/resume
    infer.py                 per-image prediction writer
    xai/explainers.py        Grad-CAM, IG, control maps
    xai/curves.py            insertion/deletion v1 with full coverage
    stats/                   paired tests, cluster bootstrap, Holm, DeLong
    analysis/                rq1.py rq2.py rq3.py rq4.py rq5.py cmrs.py
    report/                  tables + figures from analysis outputs only
    provenance.py            env capture, receipts, output hashing
  tests/                     unit + golden tests
  configs/                   YAML/JSON run configs (versioned)
  manifests/                 cohort manifests v1 (CSV + sha256)
  runs/<run_id>/             immutable run outputs
  demo/app.py                CMEX viewer (reads runs/, report/)
```

## 3. Components, interfaces and dependency order

| Order | Component | Input → Output | Depends on |
|---|---|---|---|
| 1 | `seeds`, `config`, `provenance` | config → `config_hash`; environment → `env.json` | — |
| 2 | `manifests` | legacy `audit/manifests/*_v1.csv` + raw metadata → `manifests/{ham_split0,derm7pt_primary,derm7pt_all,pad,pad_rq5_partition,ddi,xai_derm7pt,xai_ham}_v1.csv` | 1 |
| 3 | `data` | manifest rows + `prep_version` → tensors (+ masks) | 2 |
| 4 | `perturb` | (image uint8, mask or None, factor, severity, rng) → (image, mask) | 1 |
| 5 | `models`, `train` | config → checkpoint series + `train_log.jsonl` | 3 |
| 6 | `infer` | checkpoint + manifest + condition → `predictions.parquet` (or CSV) | 3, 4, 5 |
| 7 | `xai` | checkpoint + XAI manifest → `xai_images.parquet` + `curves.npz` | 3, 4, 5 |
| 8 | `stats` | arrays → estimates / CIs / p-values | — (pure functions) |
| 9 | `analysis` | per-image files → `analysis_<rq>.json` | 6, 7, 8 |
| 10 | `report` | analysis JSON → tables/figures | 9 |
| 11 | `demo` | report + stored per-image rows → UI | 10 |

## 4. Reuse vs rewrite

| Legacy element | Decision |
|---|---|
| `labels.py` mapping functions | Import directly (read-only) |
| Legacy manifests | Input to the v1 manifest builder |
| Cache `.npy` + metadata | Read-only input under `prep_v1_cached256`. Validate its order against the v1 manifest |
| `msim` blur/compression/illumination/nuisance ops | Call them with an explicit `rng` argument. Never use `ModalityShift._rng` |
| `msim.scale` | Re-implement as `scale_v1`, which returns the affine parameters and applies the same geometry to the mask. The mask surround is filled with 0 (non-lesion) |
| `eval/metrics` sens/spec/AUROC/DeLong | Port into `stats`. Unit-test DeLong against known values (e.g. comparison with the pROC reference example, or permutation agreement) |
| `bootstrap_ci` | Replace with `cluster_bootstrap(fn, df, cluster_col, B=2000, seed)` |
| `eval/xai.GradCAM` | Port, adding a zero-map guard and counter |
| `deletion_insertion` | Replace (§9) |
| `campaign.py` | Replace with a small `run.py` that keeps the attempt/receipt pattern |
| `cmex_app.py` | Replace |
| `protocol/power_analysis.py` | Call read-only for planning tables. It has no role in the results |

## 5. Data manifest schema (v1)

One row per image (Derm7pt has one row per image, with `pair_id` linking the pairs).

| Field | Type | Notes |
|---|---|---|
| `dataset` | str | ham10000 / derm7pt / pad_ufes_20 / ddi |
| `image_uid` | str | `<dataset>:<native image id>` |
| `rel_path` | str | relative to `DATASETS/` |
| `sha256` | str | source file hash |
| `label_mel` | int | 1 = melanoma |
| `dx_raw` | str | native diagnosis string |
| `modality` | str | dermoscopy / clinical / smartphone |
| `cluster_id` | str | lesion_id (HAM), case_id (Derm7pt), patient_id (PAD), DDI_ID |
| `pair_id` | str/null | Derm7pt case |
| `same_file_pair` | bool | Derm7pt flag |
| `split` | str | train/val/test (HAM); adapt_train/adapt_val/eval (PAD RQ5) |
| `split_seed` | int | 0 |
| `mask_path` | str/null | HAM only |
| `skin_tone` | str/null | DDI group or PAD Fitzpatrick |
| `cache_name`, `cache_index` | str, int | for `prep_v1_cached256` |
| `cohort_flags` | str | e.g. `derm7pt_primary;xai_derm7pt` |

Each manifest also gets a `.sha256` file and a `manifest_card.json` containing counts per label and split. These counts are asserted against Stage 1 figures: HAM 6938/1529/1548; Derm7pt 1002/249 and 1011/252; PAD 2298/52/36 patients.

## 6. Per-image output schemas

**predictions (one row per image × condition × model):**
`run_id, model_id, arch, strategy, train_seed, split_seed, ckpt_sha256, prep_version, image_uid, pair_id, cluster_id, dataset, modality, label_mel, factor, severity, perturb_seed, logit, prob, threshold, pred_mel, metric_version`

**xai_images (one row per image × condition × explainer/control):**
`… identifiers as above …, explainer (gradcam|ig|ctrl_random_k|ctrl_centre|ctrl_sobel|ctrl_mismatch), target=mel_logit, baseline=blur_k, steps, p_full, p_base, ins_auc_v1, del_auc_v1, ins_mean_v0, del_mean_v0, ins_norm, del_norm, norm_defined, map_constant(bool), map_max, ig_completeness_err, iou, pointing_hit, mask_transformed(bool), curve_ref`

**curves.npz:** `curve_ref → (ins[steps+1], del[steps+1], frac[steps+1])`, stored as float32.

**Map storage:** only saliency maps for the demo subset (≤ 50 images) are stored, as float16 `.npz`. All other maps are regenerable from the checkpoint.

## 7. Explicit versions

| Key | Value |
|---|---|
| `prep_version` | `prep_v1_cached256` (primary). `prep_v1_raw224` exists only for a consistency test on 50 images |
| `aug_version` | `aug_std_v1` (the legacy `train_aug` recipe), `aug_heavy_v1` (the legacy `_heavy_tf` recipe), `aug_none` |
| `perturb_version` | `msim_v1` (legacy parameters + stable seeding + `scale_v1`) |
| `metric_version` | `metrics_v1` (trapezoid, normalised, cluster bootstrap) |
| `xai_version` | `xai_v1` (targets, baseline, steps, controls) |
| `cmrs_version` | `cmrs_v1` |

Every output row and every analysis JSON carries all of these keys.

## 8. Seed policy

| Seed | Controls | Rule |
|---|---|---|
| `split_seed` = 0 | HAM lesion split; PAD RQ5 patient partition | Fixed for the whole study |
| `train_seed` ∈ {0,1,2} | Weight-init head, data order, dropout, augmentation | `seed_everything(train_seed)` sets Python `random`, NumPy, Torch and CUDA. Albumentations `Compose(..., seed=stable_seed(train_seed,"aug"))` plus per-worker reseeding in `worker_init_fn`. `num_workers=0` by default (measured faster on this machine; UNV) |
| `perturb_seed` | Stochastic perturbations | `stable_seed(split_seed, image_uid, factor)` via SHA-256 → uint64. It is independent of severity, process and worker, and **never uses built-in `hash()`** |
| `control_seed` | Random control maps | `stable_seed("ctrl", k, image_uid)`, k=0..4 |
| `sample_seed` | XAI manifests, bootstrap | Fixed constants recorded in config |
| Determinism | — | `torch.backends.cudnn.deterministic=True`, `benchmark=False`; `torch.use_deterministic_algorithms(True, warn_only=True)`. Warnings are logged. Bitwise GPU determinism is not claimed. Reproducibility is verified through a tolerance test (§12) |
| Environment | — | The runner sets `PYTHONHASHSEED=0` for child processes (defence in depth), even though no code path depends on it |

## 9. Insertion/deletion v1 specification

- Input: the normalised image x, map s (H×W), baseline b = Gaussian blur of x (fixed k = 2·⌊min(H,W)/8⌋+1, σ = k/6, reflect padding; kept identical to the legacy method so that v0 values are comparable).
- Order: `argsort` over (−s, pixel_index), which is **stable and deterministic**.
- Steps: S = 32. Boundaries `e_j = round(j·N/S)` for j=0..S, so that **all N pixels are covered** and chunk sizes differ by at most 1.
- Curves: `ins[0]=f(b)`, `ins[S]=f(x)`; `del[0]=f(x)`, `del[S]=f(b)`, where f = sigmoid melanoma probability in fp32.
- `ins_auc_v1 = trapz(ins, frac)` with `frac = e_j/N`; `del_auc_v1` likewise. The v0 mean is also stored.
- Constant map (max−min < 1e-12): the flag is set, the image is **excluded from explainer statistics**, and it is counted in the report.
- Assertions: `abs(ins[S]-f(x))<1e-4`, `abs(del[S]-f(b))<1e-4`, `len==S+1`.

## 10. Integrated Gradients specification

- Baseline = the same blurred image b. Target = the melanoma logit. Steps = 32 (Riemann midpoint; pilot tests 16/32/64). Batched and in fp32.
- Map = sum over channels of |attr|, then a 5×5 box smoothing (recorded). This produces a spatial ranking for the curves.
- Completeness error = |Σattr − (logit(x) − logit(b))| / |logit(x) − logit(b)|, stored per image. The pilot threshold is a median below 5%.
- Implementation: Captum if it is already installed (to be checked read-only in T-01); otherwise about 30 lines of own code with a unit test against a linear model, where IG is exact.

## 11. Image–mask geometry

- `perturb.apply(img, mask, factor, sev, rng)` returns `(img2, mask2, geom)`.
- Photometric factors (blur, illumination, compression, nuisance): `mask2 = mask`. Glare and hair do not change lesion geometry. **Occlusion is not masked out; this is documented.**
- `scale_v1`: apply the same resize and offset to the mask with INTER_NEAREST and pad with 0. Unit test: the mask centroid moves with the lesion, and the mask area scales by 1/z².
- Any future geometric factor must implement a mask transform, or declare `mask_supported=False`. In that case IoU and pointing are written as NaN with the reason `geometry_unsupported`.

## 12. Checkpoint and resume

A checkpoint is saved at the end of every epoch (atomic write to `.tmp`, then rename) and contains:

- `model_state`, `optimizer_state`, `scheduler_state`, `scaler_state`;
- `epoch_completed`, `global_step`, `best_val_auroc`, `best_epoch`;
- RNG states: Python `random.getstate()`, NumPy `get_state()`, `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, the Albumentations Compose RNG state (or its seed plus the number of draws), and the DataLoader generator state;
- `config`, `config_hash`, `code_sha`, `manifest_sha`, `env_hash`.

Additional rules:

- The best model is saved separately as `best.pt`, together with its threshold.
- Resume is allowed only if `config_hash`, `manifest_sha` and `code_sha` match. Otherwise the run fails with an error.
- Resume happens at epoch granularity. A partially completed epoch is discarded and re-run from the start of that epoch.
- Test: train 2 epochs; separately, train 1 epoch, kill, resume and train 1 epoch. The final weights must match within tolerance (max abs diff < 1e-4 on CPU; on GPU a looser tolerance plus identical val AUROC to 3 decimals). The result is recorded.

## 13. Provenance and immutability

- Run ID = `<utc timestamp>_<config_hash[:8]>`. The run directory is created exclusively (failure if it exists).
- `run.json` holds the config, the git-free code snapshot hash (SHA-256 of all `src/` files), the manifest hashes, `env.json` (Python, torch, CUDA, GPU, and all package versions), the command line, and start/end times.
- `receipt.json` is written last and holds the SHA-256 of every output. A run without a receipt is incomplete and is never used.
- Analysis reads only runs that have receipts, and records the run IDs it consumed.
- Outputs are never overwritten. A re-run creates a new run ID.

## 14. Statistical and XAI correctness checks (tests)

| Check | Test |
|---|---|
| McNemar melanoma-only | Synthetic table with known b and c; also asserts that non-melanoma rows are ignored |
| Cluster bootstrap | A cluster duplicated k times must not narrow the CI compared with a single copy |
| DeLong | Equals the Mann–Whitney AUC; variance agrees with a bootstrap within 15% on a synthetic set |
| Holm | Textbook example |
| Trapezoid | Linear curve → 0.5·(a+b); endpoint identity versus v0 |
| Coverage | For N ∈ {224², 300², 7×7} and S=32, every pixel is replaced exactly once |
| Constant map | Flag is set, and the image is excluded from explainer statistics |
| Grad-CAM | Non-zero map for a random image. fp32 gives the same result as eval mode |
| IG | Exact on a linear model |
| Controls | Random map: mean ins ≈ mean del across seeds (sanity check) |
| Perturbation | Severity 0 is bit-identical. Same output in two separate processes (subprocess test). Monotone statistics (legacy `verify`) |
| Mask geometry | `scale_v1` centroid and area test |
| Preprocessing | Cached vs raw difference recorded. Model input dtype and shape checked |
| Legacy reproduction | The v1 pipeline on the legacy checkpoint reproduces legacy Derm7pt-all sensitivity (0.948/0.802) and AUROC (0.817/0.688) to within ±0.005. Divergence stops the work (see backlog) |

## 15. Demo / report consistency

- The demo **does not compute study metrics.** It reads `analysis_*.json` and the per-image rows.
- A live "try an image" mode may compute a prediction and a Grad-CAM map using **the same `rv1.data` preprocessing and `rv1.perturb` functions**. It labels these values "single-image illustration, not a study metric", and it never displays a CMRS for a single image.
- The CMRS shown equals `analysis_cmrs.json` together with its axis signature.
- Negatives are labelled "non-melanoma".
- A consistency test loads 5 stored predictions and recomputes them through the demo code path. They must match to 1e-4.

## 16. Dependency strategy

- Use the installed environment: torch 2.11.0+cu128, timm 1.0.28, albumentations 2.0.8, scikit-learn 1.7.2, scipy 1.15.3 [FACT from `requirements_observed.txt`]. Pandas and matplotlib must be confirmed read-only in T-01.
- **No MLflow, Quantus, Docker or pytorch-grad-cam.** The own implementations are small and tested. This departs from the original plan and must be stated in the thesis.
- Captum is optional (see §10).
- **Streamlit is not installed.** Installing it (a single package, pinned) requires the researcher's approval at the demo stage (T-15). Fallback: a static HTML report with pre-rendered example panels.
- Pin `requirements_rv1.txt` from the actual environment. Nothing is installed during Stage 1.
