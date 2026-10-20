# Stage 9C1 Report — Unperturbed RQ3 Evaluation of the Nine Standard Cells

Date: 2026-10-04. All work in this stage used `research_v1/.venv_model`.

**Run:** `runs/rq3_eval_9c1/20261004T102700Z_031b3a03_940d15b8/`. The four receipts (spec, threshold, infer, analyze) verify.

**Stage records:** `runs/stage9c1/`.

## Status

| Item | Result |
|---|---|
| Spec frozen | `spec.json` (hashed). It records the 9 selected checkpoints, each checked against `best.json`, its sidecar and the checkpoint contents (epoch, architecture, seed). It also records manifests, cohorts, inference settings, threshold rule, analyses and bootstrap seeds. No latest checkpoint was substituted |
| Threshold phase (separate, completed first) | `threshold_registry.json` (hashed). **6 new thresholds**, each with HAM VAL reproduction against Stage 9B max \|Δp\| = 0.0 and 152/160 sensitivity. The 3 ResNet thresholds are reused unchanged. Each threshold is bound to its checkpoint hash |
| Inference | **22 new prediction sets**, all exit 0: 148.6 s of inference plus model loading |
| New prediction sets in detail | ResNet s1/s2: HAM TEST and PAD all. The 6 new models: HAM TEST, Derm7pt ALL and PAD all |
| Analyses | All 9 models; **0 undefined replicates** anywhere. Analysis runtime 47.9 s |
| Phase exit codes | threshold 0, infer 0, analyze 0 (`runs/stage9c1/status.txt`) |
| Tests | New `test_rq3_eval.py` (4 tests). `test_rq3_main_configs` (resource-warning fix only), `rq1_stats`, `desc_stats` and `auroc_paired` pass |
| Protected inputs | 0 changed, 0 removed. All reused artifacts were verified by hash and left unchanged |

**Why the full suite was not run:** no shared module changed.

**Out of scope:** DDI. It is not required for E1.1–E1.5 and was not run in this bounded stage.

## 1. Coverage and reuse inventory

| Model | Threshold | HAM TEST | Derm7pt ALL | PAD all |
|---|---|---|---|---|
| ResNet-50 s0 (primary) | reused (4A) | reused (4A) | reused (4A) | reused (4C) |
| ResNet-50 s1 | reused (6B) | **new** | reused (6B) | **new** |
| ResNet-50 s2 | reused (6B) | **new** | reused (6B) | **new** |
| EfficientNet-B3 s0, s1, s2 | **new** | **new** | **new** | **new** |
| DenseNet-121 s0, s1, s2 | **new** | **new** | **new** | **new** |

**New predictions:** each file keeps the full schema (IDs, pair and cluster, labels, roles and flags, logit, probability, prediction, threshold and checkpoint identity, prep and precision).

**Checks on every model and cohort:** the checkpoint identity is verified; Derm7pt pairing goes through `rq1_stats.build_pairs`, which checks labels, completeness, identities and pred == (p ≥ t).

**Bootstrap draws:** the RQ3-comparison analyses are recomputed for **all nine** models with **common bootstrap seeds**, so every model on a cohort uses the same draws:

| Cohort / analysis | Seed |
|---|---|
| HAM TEST | 20261060 |
| Derm7pt sensitivity, PRIMARY | +1 |
| Derm7pt sensitivity, ALL | +2 |
| Derm7pt AUROC | +3 |
| Derm7pt specificity | +4 |
| PAD | +5 |

The published Stage 4B, 4C and 6B intervals, which used other seeds, are **unchanged**. For example, the primary ResNet-50 s0 Δ is 43.4 pp with CI [36.5, 50.2] (published) vs [36.1, 50.6] (RQ3 recomputation).

## 2. Thresholds (HAM VAL; v1 rule)

| Model | Epoch | Threshold | VAL specificity (TN/FP) |
|---|---|---|---|
| ResNet-50 s0 / s1 / s2 | 8 / 7 / 8 | 0.1757 / 0.1179 / 0.0938 | reused |
| EfficientNet-B3 s0 / s1 / s2 | 5 / 4 / 6 | 0.0781 / 0.1632 / 0.0704 | 787/582, 869/500, 868/501 |
| DenseNet-121 s0 / s1 / s2 | 6 / 5 / 8 | 0.3366 / 0.4017 / 0.2854 | 942/427, 918/451, 928/441 |

## 3. Per-model results (`comparison_table.csv`, `per_model_results.json`)

| Model | HAM AUROC [CI] | HAM sens / spec | Derm7pt Δ sens, pp [CI]; b/c | Derm7pt AUROC, derm / clin (Δ) | Correctness accuracy, derm / clin | ECE, derm / clin | PAD AUROC [CI] | PAD sens / spec |
|---|---|---|---|---|---|---|---|---|
| ResNet-50 s0 | 0.913 | 0.924 / 0.661 | 43.4 [36.1, 50.6]; 117/9 | 0.777 / 0.668 (0.109) | 0.594 / 0.693 | 0.123 / 0.134 | 0.684 | 0.596 / 0.721 |
| ResNet-50 s1 | 0.901 | 0.971 / 0.610 | 11.2 [6.0, 16.5]; 40/12 | 0.781 / 0.699 (0.082) | 0.596 / 0.587 | 0.085 / 0.081 | 0.722 | 0.808 / 0.428 |
| ResNet-50 s2 | 0.920 | 0.971 / 0.582 | 24.1 [18.1, 29.7]; 64/4 | 0.775 / 0.668 (0.107) | 0.482 / 0.583 | 0.176 / 0.124 | 0.760 | 0.865 / 0.439 |
| EfficientNet-B3 s0 | 0.884 | 0.912 / 0.577 | 24.5 [16.9, 32.1]; 83/22 | 0.797 / 0.618 (0.179) | 0.610 / 0.573 | 0.096 / 0.176 | 0.541 | 0.712 / 0.385 |
| EfficientNet-B3 s1 | 0.886 | 0.900 / 0.661 | 36.9 [29.7, 43.8]; 104/12 | 0.820 / 0.610 (0.210) | 0.632 / 0.621 | 0.087 / 0.138 | 0.632 | 0.577 / 0.624 |
| EfficientNet-B3 s2 | 0.897 | 0.924 / 0.638 | 39.0 [32.1, 45.4]; 104/7 | 0.832 / 0.638 (0.194) | 0.607 / 0.614 | 0.102 / 0.154 | 0.615 | 0.519 / 0.676 |
| DenseNet-121 s0 | 0.913 | 0.935 / 0.671 | 26.9 [18.9, 34.9]; 91/24 | 0.770 / 0.681 (0.090) | 0.675 / 0.709 | 0.118 / 0.088 | 0.753 | 0.423 / 0.876 |
| DenseNet-121 s1 | 0.914 | 0.941 / 0.694 | 32.5 [25.7, 39.8]; 90/9 | 0.780 / 0.701 (0.079) | 0.607 / 0.705 | 0.198 / 0.078 | 0.753 | 0.615 / 0.761 |
| DenseNet-121 s2 | 0.921 | 0.941 / 0.706 | 30.9 [24.1, 37.8]; 91/14 | 0.824 / 0.711 (0.113) | 0.743 / 0.732 | 0.061 / 0.073 | 0.645 | 0.327 / 0.829 |

**Where the full details are:** CIs for HAM, PAD and the Derm7pt AUROC difference and specificity difference; the descriptive McNemar, DeLong and correctness-McNemar p-values; and the Derm7pt ALL sensitivity analysis.

**Status of these p-values:** they are **descriptive** in RQ3. There are no between-architecture tests and no combined p-values.

## 4. Seed ranges (`seed_ranges.csv`; not confidence intervals)

| Architecture | HAM AUROC | Derm7pt Δ sens (pp) | Derm7pt Δ AUROC | Derm7pt clinical AUROC | PAD AUROC | PAD sens |
|---|---|---|---|---|---|---|
| ResNet-50 | 0.901–0.920 | 11.2–43.4 | 0.082–0.109 | 0.668–0.699 | 0.684–0.760 | 0.596–0.865 |
| EfficientNet-B3 | 0.884–0.897 | 24.5–39.0 | 0.179–0.210 | 0.610–0.638 | 0.541–0.632 | 0.519–0.712 |
| DenseNet-121 | 0.913–0.921 | 26.9–32.5 | 0.079–0.113 | 0.681–0.711 | 0.645–0.753 | 0.327–0.615 |

**Observations (descriptive)**
- Every model has a positive dermoscopy − clinical gap in both sensitivity and AUROC.
- Within-architecture seed variation is large, for example the ResNet-50 Δ sensitivity of 11–43 pp, and often comparable to between-architecture differences.
- EfficientNet-B3 variants have the largest Derm7pt AUROC gaps and the lowest clinical and PAD AUROCs. That pattern may involve the 224-px input deviation, which has not been quantified.
- Thresholds transfer inconsistently between models: PAD sensitivity ranges from 0.33 to 0.87.

**Scope:** these compare **common-recipe pretrained variants at 224 px**. They are not pure architecture effects (architecture and pretraining are not disentangled) and not individually optimised models. No model is declared best, seeds are not pooled, there is no ensemble, and seed-0 ResNet-50 remains the primary model.

## 5. Files

**Code and tests**
- New: `src/rv1/rq3_eval.py`, `tests/test_rq3_eval.py`, `configs/rq3_eval_9c1_v1.json`.
- Changed: `tests/test_rq3_main_configs.py`, which now uses a context manager for the provenance file. There is no computational change; the prior copy is in `runs/stage9c1/`.

**Run outputs:**
- `spec.json` and `threshold_registry.json`, each with a sidecar;
- 22 `pred_<model>_<cohort>.csv` files;
- `inference_log.json` (reuse status and timing);
- `per_model_results.json`;
- `comparison_table.csv`, `seed_ranges.csv`;
- `thesis_results_rq3_unperturbed.md`, which has its own sidecar;
- the receipts.

**Integrity:** 1678 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged. No `__pycache__` was created.

## 6. Remaining (RQ3 is **not** complete)

- RQ2 perturbation comparisons across the standard models.
- The `aug_heavy_v1` specification and the 9 heavy cells.
- DDI for the new models, if later required.
- The deferred RQ4 components: secondary IG variants, E4.6 and the simulated XAI arm.

**Stage 9C1 is complete.**
