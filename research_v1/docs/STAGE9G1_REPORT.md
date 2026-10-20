# Stage 9G1 Report — Unperturbed Evaluation of the Nine Heavy Models and Matched Standard-vs-Heavy Contrasts

Date: 2026-10-04. Environment: `research_v1/.venv_model`.

**Run:** `runs/rq3_heavy_eval_9g1/20261004T152428Z_2cea4004_64df5fc0/` (19 MB).

**Stage records:** `runs/stage9g1/`.

**Scope:** thresholds on HAM VAL, then HAM TEST, Derm7pt ALL/PRIMARY and PAD all, for the 9 selected heavy checkpoints. No training, no reselection, no RQ2, DDI or XAI.

## Status

| Item | Result |
|---|---|
| Code | New `src/rv1/rq3_heavy_eval.py`. Its spec and compare phases are new. The threshold, infer and analyze phases are the **unchanged, tested Stage 9C1 functions** (`rq3_eval.phase_*`), driven by a 9C1-format spec. No shared module changed |
| Tests | New `tests/test_rq3_heavy_eval.py`, **6/6 pass**: binding, matching, difference direction, shared resampling, known difference-of-gaps, join completeness. `test_rq3_eval.py` 4/4 pass. The full suite was not rerun, because no shared module changed |
| Spec | `spec.json` (hashed). For each model it was verified against the file, sidecar, `best.json`, the embedded architecture/seed/`aug_version = aug_heavy_v1` and the Stage 9F card. Selected (not latest) checkpoints. Standard references come from Stage 9C1 |
| Phase exits | threshold 0, infer 0, analyze 0, compare 0. All receipts verify. Total time about 6.5 min |
| Thresholds | `threshold_registry.json`, frozen before inference: 9 new thresholds bound to each checkpoint SHA-256. HAM VAL reproduced with max \|Δp\| = 0.0. No standard threshold was reused |
| Predictions | **27 sets**: 9 × HAM TEST (1548), 9 × Derm7pt ALL (2022, PRIMARY flag kept), 9 × PAD all (2298, patient ID and rq5_role kept). All probabilities finite, identifier joins complete, and checkpoint and threshold identities on every row |
| Bootstrap | B = 2000. Per-model intervals use the Stage 9C1 seeds, so standard and heavy models get identical cluster draws. Contrast seeds were frozen in the spec. **0 undefined replicates everywhere** |
| Integrity | 3047 → 3048 files: **0 changed and 0 removed.** The only added file is the new config. Cache unchanged (§6) |

**Note on the Stage 9C1 receipts.** Their only verification difference is the documented, intentional Stage 9D change to `augment.py` (prior copy `runs/stage9d/augment_before.py.txt`). The spec phase tolerates exactly that one mismatch, re-checks every 9C1 output hash, and records the result in `spec.json`.

**PAD all:** this is an external evaluation of HAM-trained models. None of them is the PAD-adapted RQ5 model.

## 1. Thresholds and per-model results

Files: `per_model_results.json`, `comparison_table.csv`, `seed_ranges.csv`.

**Heavy thresholds**

| Arch | s0 / s1 / s2 |
|---|---|
| ResNet-50 | 0.442 / 0.446 / 0.265 |
| EfficientNet-B3 | 0.305 / 0.180 / 0.215 |
| DenseNet-121 | 0.344 / 0.446 / 0.422 |

For comparison, the standard thresholds were 0.070–0.402.

**Per-model analyses** were computed for all 9 heavy models with the Stage 9C1 definitions:
- HAM TEST with lesion-bootstrap intervals;
- Derm7pt E1.1 (descriptive McNemar), E1.2, E1.3 (DeLong p and pair bootstrap), E1.4 (correctness and ECE);
- PAD all with patient-bootstrap intervals;
- the Derm7pt ALL sensitivity analysis. ALL and PRIMARY sensitivity differences agree within 0.4 pp in every model.

**Derm7pt PRIMARY sensitivity difference (dermoscopy minus clinical, pp)**

| Arch | Heavy s0 / s1 / s2 | Standard s0 / s1 / s2 |
|---|---|---|
| ResNet-50 | 27.7 / 30.9 / 11.2 | 43.4 / 11.2 / 24.1 |
| EfficientNet-B3 | 24.1 / 1.2 / 11.6 | 24.5 / 36.9 / 39.0 |
| DenseNet-121 | 25.3 / 30.9 / 23.7 | 26.9 / 32.5 / 30.9 |

**Dermoscopic ECE:** heavy 0.19–0.36, against standard 0.06–0.20. Higher in every heavy model.

## 2. Matched contrasts

Files: `contrast_results.json`, `matched_table.csv`, `matched_seed_ranges.csv`.

**Design**
- Pairs are matched on architecture and seed.
- Differences are heavy minus standard, with each model at its own threshold.
- Within each replicate, both models are evaluated on the same resampled lesions, patients or pairs.
- For Derm7pt, both modalities and both models stay together. The difference of gaps is computed inside each replicate, never by subtracting CI endpoints.

The full tables, with both models' values and all intervals, are in `runs/stage9g1/thesis_results_rq3_heavy_unperturbed.md` (Tables H1 and H2).

**Consistency across seeds** (number of matched pairs with a positive difference, out of 3 per architecture)

| Measure | ResNet-50 | EfficientNet-B3 | DenseNet-121 |
|---|---|---|---|
| HAM sensitivity ↑ | 1 | 3 | 2 |
| HAM specificity ↑ | **0** | **0** | **0** |
| HAM AUROC ↑ | 0 | 1 (plus 1 at 0.000) | 0 |
| PAD sensitivity ↑ | 1 | 3 | 3 |
| PAD specificity ↑ | 2 | 1 | 0 |
| PAD AUROC ↑ | 2 | **3** (all intervals > 0) | 2 |
| Derm7pt dermoscopic sensitivity ↑ | 3 | 3 | 3 |
| Derm7pt clinical sensitivity ↑ | 2 | 3 | 3 |
| Derm7pt dermoscopic specificity ↑ | 0 | 0 | 0 |
| Derm7pt clinical AUROC ↑ | 2 | **3** | 0 |
| Sensitivity gap narrower | 2 | 3 | 3 |
| AUROC gap narrower | 1 | **3** | **0** |

## 3. Interpretation (exploratory)

**Operating points, not discrimination.** The dominant effect of heavy augmentation was a shift in operating points: higher sensitivity, lower specificity, higher thresholds and worse calibration. There was no uniform gain in discrimination, and HAM TEST AUROC fell slightly in 7 of 9 models.

**EfficientNet-B3 on shifted cohorts.** The only discrimination gains that were consistent across seeds were for EfficientNet-B3 on the shifted cohorts: PAD AUROC +0.12 to +0.19, and Derm7pt clinical AUROC +0.07 to +0.10, with a narrower AUROC modality gap.

**Narrower sensitivity gaps came with specificity losses.** For example, EfficientNet-B3 s1's gap of 0.37 → 0.01 came with clinical specificity falling from 0.66 to 0.22.

**Heavy is not called better.** That conclusion cannot rest on sensitivity alone. No strategy is declared a winner, and the primary model (seed-0 standard ResNet-50) is unchanged.

## 4. Files

**Code, test and config:** `src/rv1/rq3_heavy_eval.py`, `tests/test_rq3_heavy_eval.py`, `configs/rq3_heavy_eval_9g1_v1.json`.

**Run dir:**
- `spec.json`;
- `threshold_registry.json`;
- 27 `pred_*_heavy_s*_{ham_test,derm7pt,pad_all}.csv` files;
- `inference_log.json`;
- `per_model_results.json`, `comparison_table.csv`, `seed_ranges.csv`;
- `contrast_results.json`, `matched_table.csv`, `matched_seed_ranges.csv`;
- the receipts for all five phases.

**`runs/stage9g1/`:**
- `run_phases.bat`, `status.txt` and the phase logs;
- `rq3_inventory_18cell_v2.json`;
- `thesis_results_rq3_heavy_unperturbed.md`;
- the integrity files.

## 5. Updated 18-cell inventory (`runs/stage9g1/rq3_inventory_18cell_v2.json`)

| Strategy | Training | Evaluation |
|---|---|---|
| `aug_std_v1` × 9 | trained and verified | unperturbed (9C1) and RQ2 sweep (9C2) |
| `aug_heavy_v1` × 9 | trained and verified (9F) | unperturbed (9G1). **RQ2 sweep, DDI and XAI not done** |

## 6. Integrity (`runs/stage9g1/integrity_comparison.json`)

- The before snapshot (3047 files) was taken at the start of the stage, before the config, module and run were created.
- The protected scope includes `runs/main_train`, all reused standard prediction dirs (`predict_v1`, `ext_eval_4c`, `seed_spread_6b`, `rq3_eval_9c1`), configs and docs.
- Result: 0 changed, 0 removed, 1 added (the config). The cache is unchanged.
- The new run dir, the module and the test file are outside this scope.

## 7. Remaining

- RQ2 perturbation sweep for the heavy models.
- DDI (if required).
- Deferred RQ4 work.

**Stage 9G1 is complete.**
