# Stage 5B Report — RQ5 Base vs PAD-Adapted Evaluation

Date: 2026-10-03. All work in this stage used `research_v1/.venv_model`.

**Run:** `runs/rq5_eval_5b/20261003T162616Z_ff14137e_715bb2b4/`. All four receipts verify (spec, threshold, infer, analyze).

**Stage records:** `runs/stage5b/`.

## Status

| Item | Result |
|---|---|
| Adapted checkpoint | `ckpt_epoch4.pt`, SHA-256 `a18bf969…1286`, enforced as epoch 4 (`check_selected`) and verified against the file, sidecar and `best.json` |
| Base checkpoint | `303d21e7…03dd` |
| Selection clarification | `selection_clarification.json` (hashed). Actual selection data: **PAD adapt_val** (343 images, 9 melanoma), list SHA-256, epoch 4, AUROC 0.7152, earlier-epoch tie rule, per-epoch evidence and the config, `val_epoch4` and `best.json` hashes. The incorrect "HAM VAL" text in `best.json` is not used, and frozen files are not rewritten |
| Evaluation specification | `eval_spec.json`, hashed in the spec phase before any adapted inference |
| **Secondary A threshold (adapted)** | **t = 0.040640708059072495** on full HAM VAL (152/160 = 0.950; specificity 797/1369 = 0.582; HAM VAL AUROC 0.874). Saved and hashed (`f1c34337…`) in a **separate completed phase** before PAD eval, HAM TEST and Derm7pt inference. The base threshold of 0.1757 is unchanged |
| **Secondary B** | **NOT_ESTIMABLE**: adapt_val has 9 melanoma images, fewer than 20. Not fitted |
| Joins and checks | PAD eval 1150 images, 687 patients, 26 melanoma images from 18 patients ✓. Complete one-to-one joins with the base predictions. Every thresholded prediction was re-verified, each with **its own** model's threshold (enforced by `ModelThreshold`) |
| Undefined bootstrap replicates | 0 for every metric |
| Runtime (measured) | Threshold phase including model load and HAM VAL inference: 7.9 s. Inference: PAD eval 3.8 s, HAM TEST 5.6 s, Derm7pt clinical 3.7 s. Analysis 19.4 s. **59 s** wall-clock for all four phases |
| Tests | New `test_rq5_stats.py` (6 tests). Relevant suites pass |
| Protected inputs | 0 changed, 0 removed |

**Why the full suite was not run:** no shared module changed.

**Not done:** training, reselection, recipe changes and calibration.

## 1. Results — PAD eval (paired patient-cluster bootstrap, B = 2000, seed 20261011)

| Measure | Base | Adapted | Adapted − base [95% CI] |
|---|---|---|---|
| **AUROC (primary RQ5 contrast, descriptive)** | 0.7201 [0.598, 0.834] | 0.6410 [0.524, 0.750] | **−0.0791 [−0.1573, −0.0122]** |
| Sensitivity (Secondary A) | 16/26 = 0.615 | 20/26 = 0.769 | +0.154 [+0.036, +0.290] |
| Specificity (Secondary A) | 834/1124 = 0.742 | 316/1124 = 0.281 | −0.461 [−0.494, −0.429] |

TP/FN/TN/FP: base 16/10/834/290; adapted 20/6/316/808.

**What the paired bootstrap does:** within every replicate, both models are scored on the **same** sampled patients, with all of each patient's images and the multiplicity kept. The difference CIs are percentiles of the paired differences, not subtractions of the separate CI endpoints.

**Interpretation**
- **Ranking:** for this recipe and partition, the adapted model's ranking (AUROC) on PAD eval was **lower**, and the paired CI excludes 0.
- **Detection at the threshold:** at its own HAM-VAL threshold, which is much lower (0.041 vs 0.176), the adapted model flagged far more images as melanoma. Sensitivity rose (+4 melanoma images) and specificity fell sharply. This is an operating-point shift, not an improvement in ranking.
- **Small numbers:** there are only 26 melanoma images from 18 patients, so all estimates are imprecise.

## 2. Forgetting checks (AUROC, adapted − base; paired bootstrap)

| Cohort | n (mel) | Clusters (unit) | Base | Adapted | Difference [95% CI] |
|---|---|---|---|---|---|
| HAM TEST | 1548 (170) | 1155 (lesion) | 0.9133 | 0.8972 | −0.0162 [−0.0359, +0.0020] |
| Derm7pt PRIMARY clinical | 1002 (249) | 1002 (case) | 0.6677 | 0.6874 | +0.0197 [−0.0157, +0.0551] |
| Derm7pt ALL clinical (sensitivity analysis) | 1011 (252) | 1011 (case) | 0.6650 | 0.6841 | +0.0192 [−0.0160, +0.0552] |

Both intervals include 0, which **does not establish** that performance was preserved: the intervals also include changes of a few AUROC points. No cross-dataset causal interpretation is made.

## 3. Disclosures and limitations

- **Prior inspection:** the PAD eval **baseline** results were inspected in Stage 4C. The adaptation recipe (Addendum H) was frozen without using them, and nothing was tuned after this evaluation.
- **Scope:** one recipe, one seed and one partition. Selection used only 9 melanoma images from 5 patients.
- **Conditionality:** the CIs are conditional on the fixed models and thresholds. They exclude training and selection uncertainty.
- **Threshold choice:** each model uses its own model-specific HAM-VAL threshold. They are not one shared value.
- **Not done:** no image-level McNemar and no confirmatory test.
- **Generalisation:** the results apply to this recipe and partition only. They do not show that adaptation in general helps or harms.

## 4. Tests (`tests/test_rq5_stats.py`, 6 tests)

| Area | What is checked |
|---|---|
| Thresholds | A model-specific threshold applied to the other model's rows raises an error |
| Secondary B | NOT_ESTIMABLE at 9 positives, with no threshold |
| Joins | Missing, duplicate, label-mismatched, wrong-model and invalid-probability records are rejected |
| Paired bootstrap | Identical draws for both models, with all images and multiplicity kept. Families bootstrapped with the same seed get the same draws. Δ equals the adapted value minus the base value per replicate |
| Swapping models | Reverses the AUROC and sensitivity differences |
| Selected-checkpoint identity | An epoch-5 or wrong-hash checkpoint is rejected |

## 5. Files

**New code, tests and config:** `src/rv1/rq5_stats.py`, `src/rv1/rq5_eval.py`, `tests/test_rq5_stats.py`, `configs/rq5_eval_5b_v1.json`.

**Run outputs:**
- `selection_clarification.json` and `eval_spec.json`, each with a sidecar;
- `threshold_adapted.json` with its sidecar;
- `pred_adapted_{ham_val,pad_eval,ham_test,derm7pt_clinical}.csv`;
- `inference_log.json`;
- `results.json`, `paired_table.csv`;
- `bootstrap_replicates.npz`;
- `thesis_results_rq5.md`, which has its own sidecar;
- four receipts.

**Integrity:** 710 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged. No `__pycache__` was created.

## 6. Remaining work (not started)

- Seeds 1–2 and other architectures.
- RQ2 and RQ3.
- Full XAI (RQ4).

**Stage 5B is complete.**
