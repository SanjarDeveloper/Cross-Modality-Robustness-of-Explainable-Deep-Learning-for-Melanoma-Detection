# Stage 9C2 Report — RQ2-Style Perturbation Evaluation Across the Nine RQ3 Standard Models

Date: 2026-10-04. All work in this stage used `research_v1/.venv_model`.

**Run:** `runs/rq3_rq2_9c2/20261004T103829Z_73f9ae39_a3207e31/`. The spec and analyze receipts verify.

**Stage records:** `runs/stage9c2/`.

## Status

| Item | Result |
|---|---|
| Spec frozen | `spec.json` (hashed). It records the 9 models (checkpoint hash, epoch, threshold from the Stage 9C1 registry, own unperturbed HAM TEST reference and hash), the simulator (`perturb_msim_v1`, `855cf57b…`), grid, pipeline, manifest, source hashes, baseline tolerance (1e-6) and bootstrap settings |
| Coverage | ResNet-50 s0 **reuses** the Stage 8B sweep (verified by receipt). **8 new model sweeps × 45 units = 360 unit directories, all verified.** Logical rows: 8 × 69,660 = **557,280 new**, **626,940** including the reuse |
| Identical perturbations | Each unit is transformed **once** from the original cached images (seeds depend only on `image_uid` and factor) and scored by all 8 models. The audit digests (first 4 records of every unit) **equal Stage 8B's** transformed images |
| Baseline checks | Severity 0 against each model's **own** unperturbed HAM TEST predictions (Stage 9C1 or 4A): **40/40 checks with max \|Δp\| = 0.0**. Identities, labels, clusters and counts all agree |
| Reproduction of Stage 8C (ResNet-50 s0) | With the same bootstrap seed and draws, the point estimates **and** CIs reproduce Stage 8C exactly (max \|Δ\| 0.0). The published 8C results are unchanged |
| Exit codes | sweep 0, analyze 0 |
| Runtime | Sweep **24.4 min** (10:38:41–11:03:05Z; estimated about 25 min), about 9 s transformation and 27 s inference per unit for 8 models. Analysis 80 s. Output 236 MB |
| Tests | New `test_rq3_rq2.py` (6 tests). `test_rq2_stats` and `test_rq2_predict` pass |
| Protected inputs | 0 changed, 0 removed |

**Why the full suite was not run:** no shared module changed.

## 1. Baselines (severity 0; each model's own threshold)

| Model | Sens | Spec | AUROC |
|---|---|---|---|
| ResNet-50 s0 / s1 / s2 | 0.924 / 0.971 / 0.971 | 0.661 / 0.610 / 0.582 | 0.913 / 0.901 / 0.920 |
| EfficientNet-B3 s0 / s1 / s2 | 0.912 / 0.900 / 0.924 | 0.577 / 0.661 / 0.638 | 0.884 / 0.886 / 0.897 |
| DenseNet-121 s0 / s1 / s2 | 0.935 / 0.941 / 0.941 | 0.671 / 0.694 / 0.706 | 0.913 / 0.914 / 0.921 |

## 2. Worst-case loss over the tested grid (`worst_case_table.csv`; sensitivity pp / AUROC)

| Model | Blur | Illumination | Compression | Scale | Nuisance |
|---|---|---|---|---|---|
| ResNet-50 s0 | 72.9 / 0.066 | 34.7 / 0.272 | 0.0 / 0.052 | 22.4 / 0.078 | 7.1 / 0.008 |
| ResNet-50 s1 | 63.5 / 0.109 | 30.6 / 0.271 | 1.8 / 0.041 | 20.6 / 0.087 | 11.2 / 0.009 |
| ResNet-50 s2 | 42.4 / 0.138 | 35.3 / 0.271 | 0.0 / 0.027 | 10.0 / 0.059 | 7.6 / 0.019 |
| EfficientNet-B3 s0 | 41.2 / 0.170 | 22.9 / 0.182 | 0.0 / 0.036 | 8.2 / 0.114 | 7.1 / 0.000 |
| EfficientNet-B3 s1 | 31.2 / 0.363 | 30.6 / 0.175 | 17.1 / 0.121 | 1.8 / 0.288 | 17.1 / 0.012 |
| EfficientNet-B3 s2 | 55.9 / 0.118 | 25.9 / 0.170 | 10.6 / 0.080 | 47.1 / 0.154 | 8.2 / 0.015 |
| DenseNet-121 s0 | 60.6 / 0.105 | 43.5 / 0.291 | 1.2 / 0.039 | 24.7 / 0.076 | 18.8 / 0.022 |
| DenseNet-121 s1 | 93.5 / 0.057 | 44.7 / 0.349 | 1.2 / 0.012 | 52.4 / 0.080 | 14.7 / 0.020 |
| DenseNet-121 s2 | 73.5 / 0.086 | 45.9 / 0.284 | 1.8 / 0.025 | 40.6 / 0.051 | 10.6 / 0.011 |

CIs are in the CSV files; 0 undefined replicates. Seed ranges per architecture and factor are in `seed_ranges.csv`.

**Specificity:** specificity at every level is in `per_level_table.csv`. It moves in the opposite direction for blur and scale, which raise specificity, and with illumination and compression, which lower it, consistent with shifts of the operating point.

## 3. Crossing markers and rankings (`crossing_table.csv`, `ranking_table.csv`)

**Observed 10-pp sensitivity crossings by factor**

| Factor | Crossing / share of replicates with no crossing |
|---|---|
| Blur | 0.16–0.56; never censored |
| Illumination | 0.30–0.61; never censored |
| Scale | Varies by model: ResNet-50 s2 = 1.0 (50% no crossing); EfficientNet-B3 s0/s1 = ">1.0" (66% and 99% no crossing); others 0.20–0.56 |
| Nuisance | Mostly ">1.0", or late (0.61–0.96) |
| Compression | ">1.0" except EfficientNet-B3 s1 (0.90) and s2 (0.99, 40% no crossing) |

**Within-model rankings** (factor with the largest worst-case loss, share of replicates)

| Metric | Result |
|---|---|
| Sensitivity | Blur in all nine models (shares 0.58–1.00) |
| AUROC | Illumination in eight models (shares 0.63–1.00); blur for EfficientNet-B3 s1 (0.98) |

## 4. Interpretation (descriptive)

**Patterns that repeated across all architectures and seeds**
- Blur gives the largest sensitivity loss.
- Illumination gives the largest AUROC loss (8 of 9 models).
- Nuisance has small AUROC effects.
- Compression rarely reduces sensitivity.

**Patterns that varied within architectures, sometimes more than between them**
- Scale losses: EfficientNet-B3 1.8–47.1 pp; DenseNet-121 24.7–52.4 pp.
- Blur AUROC loss: EfficientNet-B3 0.118–0.363.
- Compression sensitivity loss: EfficientNet-B3 0–17.1 pp.

**Read losses together with baselines.** For example, DenseNet-121 has the highest baselines and also the largest illumination losses. EfficientNet-B3 has smaller illumination sensitivity losses but lower baselines and lower AUROC. A small loss from a lower baseline is not evidence of robustness.

**No universally best architecture is established.** The patterns depend on the factor and metric, and between-seed variation is large.

**Not done:** between-architecture significance tests, combined p-values, ensembles or seed pooling. The seed-0 ResNet-50 remains the primary model.

**Scope:** this is a **common-recipe pretrained-model comparison** at 224 px. It does not isolate a causal architecture effect, and pretraining is confounded with architecture. Rankings hold only within the chosen parameter ranges, and the perturbations are synthetic and bundled.

## 5. Files

**New code, tests and config:** `src/rv1/rq3_rq2.py`, `tests/test_rq3_rq2.py`, `configs/rq3_rq2_9c2_v1.json`.

**Run outputs:**
- `spec.json` with its sidecar;
- 360 `u_<model>__<factor>__<sev>/` directories (predictions plus a hashed manifest with the audit digests and baseline difference);
- `progress.jsonl`;
- `results.json` (including the Stage 8C reproduction);
- `per_level_table.csv`, `worst_case_table.csv`, `crossing_table.csv`, `ranking_table.csv`, `seed_ranges.csv`;
- 10 plots (`plot_<factor>_{d_sens_pp,d_auroc}.png`, one line per model);
- `thesis_results_rq3_rq2.md`, which has its own sidecar;
- the receipts.

Every row carries the model ID, checkpoint hash, threshold, simulator hash and perturbation seed and parameters.

**Integrity:** 1730 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged. No `__pycache__` was created.

## 6. Remaining (the RQ3 matrix is **not** complete)

- The `aug_heavy_v1` specification and the 9 heavy cells, with their evaluation.
- Deferred RQ4 components: secondary IG variants, E4.6 and the simulated XAI arm.
- DDI for the new models, if required.

**Stage 9C2 is complete.**
