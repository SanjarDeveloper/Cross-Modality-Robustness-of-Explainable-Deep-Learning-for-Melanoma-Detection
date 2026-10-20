# Stage 4C Report — External Baseline Evaluation on PAD-UFES-20 and DDI

Date: 2026-10-03. All work in this stage used `research_v1/.venv_model`.

**Run:** `runs/ext_eval_4c/20261003T160420Z_c5ef9dda_9190c7cf/`. The spec, infer and analyze receipts all verify.

**Stage records:** `runs/stage4c/`.

## Status

| Item | Result |
|---|---|
| Specification frozen before inference | `eval_spec.json` (hashed; separate process). It records the checkpoint and threshold hashes, manifest and cache hashes, prep and precision, batch, order, cohorts and roles, bootstrap units, seeds, B, quantiles, DDI tone groups and the missing-data rule |
| Model and threshold | Seed-0 ResNet-50 epoch 8 (`303d21e7…`, verified); Stage 4A threshold t = 0.17569470405578613 (`469a4d45…`), unchanged |
| Inference | Same path as Stage 4A (`predict_v1.setup` and `predict_v1.infer`): `prep_v1_cached256`, `xai_precision_v1`, batch 16, manifest order |
| Inference coverage | PAD 2298 rows and DDI 656 rows. Exact one-to-one manifest join, no duplicates, finite scores in [0, 1], pred == (p ≥ t) for every row |
| PAD eval counts | 1150 images, 687 patients, 26 melanoma images, 18 melanoma patients, as expected. Patient roles are disjoint |
| DDI counts | 656 images, 21 melanoma; 7 melanoma in each of the 3 tone groups, as expected |
| Undefined bootstrap replicates | PAD: 0. DDI: 2/2000 in FST I–II and in FST III–IV (0.1%, not flagged) |
| Runtime (measured) | Model setup 2.3 s; PAD inference 8.6 s; DDI inference 2.3 s; analysis 11.9 s. 45 s wall-clock for all three phases |
| Tests | New `test_ext_stats.py` (6 tests). Relevant suites pass |
| Protected inputs | 0 changed, 0 removed |

**Why only the relevant suites were run:** no shared module changed. The modules `predict_v1`, `desc_4b3`, `rq1_stats`, `auroc_paired` and `config.py` were reused unchanged.

**Output safety:** outputs were published through a temporary file and a non-overwriting rename, after JSON validation.

**Correction note:** `docs/STAGE4B3_SUMMARY_CORRECTION_v1.md`. Clinical melanoma sensitivity was 106/249 = 42.6%, not "almost none".

## 1. PAD cohort roles (source documented)

**Applicable sources**
- 02_rev2 §1.2: protocol cohort "PAD all".
- §3 E1.5: PAD descriptive, with patient-cluster bootstrap.
- §7.1: the frozen `rq5_role` partition.
- §7.4: PAD eval Fitzpatrick counts and metrics.

No later addendum restricts PAD.

**Roles used**

| Cohort | Role in this stage |
|---|---|
| **PAD all** (2298 images, 1373 patients) | **E1.5 descriptive external baseline** of the unadapted model. Every row keeps its `rq5_role` |
| **PAD eval** (1150 / 687) | Baseline on the frozen RQ5 evaluation partition, for future comparison |
| adapt_train (805 / 482; 17 melanoma images / 13 patients) | **Counts only.** No metrics were used for any choice |
| adapt_val (343 / 204; 9 / 5) | **Counts only** |

**Limits on use**
- PAD-all results are **not** held-out performance of any adapted model.
- The partition was not changed.
- Nothing here chooses an adaptation recipe.

**Mixed-label patients:** 11 (eval 7, adapt_train 3, adapt_val 1). Image labels are retained.

## 2. PAD results (image-level; patient-cluster percentile bootstrap, B = 2000)

| Cohort | Images (mel) | Patients (mel) | TP/FN/TN/FP | Sensitivity [95% CI] | Specificity [95% CI] | AUROC [95% CI] |
|---|---|---|---|---|---|---|
| PAD all | 2298 (52) | 1373 (36) | 31/21/1619/627 | 31/52 = 0.596 [0.460, 0.731] | 1619/2246 = 0.721 [0.701, 0.740] | 0.684 [0.599, 0.765] |
| PAD eval | 1150 (26) | 687 (18) | 16/10/834/290 | 16/26 = 0.615 [0.435, 0.800] | 834/1124 = 0.742 [0.713, 0.771] | 0.720 [0.612, 0.828] |

**Bootstrap details**
- Seeds: 20261005 for PAD all and 20261006 for PAD eval.
- All images of each sampled patient are included, with multiplicity kept.

**Not done:** image-level McNemar, and any paired comparison with Derm7pt or HAM.

## 3. PAD eval Fitzpatrick groups (§7.4)

**Counts per group** (images / patients / melanoma images):

| Group | Images | Patients | Melanoma images |
|---|---|---|---|
| FST 1 | 81 | 40 | 3 |
| FST 2 | 429 | 230 | 19 |
| FST 3 | 205 | 107 | 4 |
| FST 4 | 24 | 16 | 0 |
| FST 5 | 4 | 3 | 0 |
| FST 6 | 0 | 0 | 0 |
| **Missing** | **407** | **291** | **0** |

**Metrics only where there are at least 5 melanoma images.** Only **FST 2** qualifies:
- sensitivity 10/19 = 0.526;
- specificity 289/410 = 0.705;
- AUROC 0.674.

**What was not done:** no missing values were imputed. Subgroup CIs, tests and ranking were not computed, and none are reported.

## 4. DDI results (image-level; no patient identifiers)

| Group | Images (mel) | TP/FN/TN/FP | Sensitivity [Wilson 95% CI] | Specificity | AUROC [image bootstrap 95% CI] | Undefined replicates |
|---|---|---|---|---|---|---|
| Overall | 656 (21) | 6/15/382/253 | 6/21 = 0.286 [0.138, 0.500] | 0.602 | 0.460 [0.341, 0.571] | 0 |
| FST I–II | 208 (7) | 3/4/137/64 | 3/7 = 0.429 [0.158, 0.750] | 0.682 | 0.478 [0.216, 0.751] | 2 |
| FST III–IV | 241 (7) | 0/7/135/99 | 0/7 = 0.000 [0.000, 0.354] | 0.577 | 0.363 [0.238, 0.492] | 2 |
| FST V–VI | 207 (7) | 3/4/110/90 | 3/7 = 0.429 [0.158, 0.750] | 0.550 | 0.548 [0.319, 0.774] | 0 |

**Interval settings**
- Wilson score intervals without continuity correction, checked against the Newcombe (1998) reference values.
- Image bootstrap seeds: 20261007 (overall) and 20261008–10 (tone groups).

**How to read the results**
- The overall DDI AUROC (0.460) is not distinguishable from chance on these data.
- Each tone group has only **7 melanomas**, so the estimates are very imprecise.
- **No fairness, bias, equity or causal conclusion is drawn, and the groups are not tested or ranked.**

## 5. Tests (`tests/test_ext_stats.py`, 6 tests)

| Area | What is checked |
|---|---|
| Wilson intervals | Reference examples: 0/10, 5/10 and 81/263 (Newcombe 1998), plus n = 0 and 7/7 |
| Role isolation | A patient in two roles, or an unknown role, is rejected |
| Tone mapping | PAD Fitzpatrick values map to groups and missing values are kept as "missing"; invalid values are rejected; the DDI mapping is checked |
| Subgroup boundary | 4 melanoma images → no metrics; 5 → metrics |
| Patient resampling | A patient with both a melanoma and a nevus keeps its per-image labels; all images are included with the correct multiplicity; undefined AUROC replicates are counted and flagged |
| Prediction join | Missing rows, duplicate rows and label mismatches are rejected |

## 6. Limitations

- One model (seed 0) and one threshold derived from HAM VAL.
- No adaptation.
- PAD and DDI differ from HAM and Derm7pt in site, device, population and label definitions, so **no cross-dataset comparison or causal claim** is made.
- PAD patients contribute repeated images.
- DDI intervals assume image independence, because patient identifiers are unavailable.
- PAD has 407 eval images with missing skin type.
- The PAD-all role mixture includes the future adaptation partitions, so this is not an adapted-model result.

## 7. Files

**New code and config:** `src/rv1/ext_stats.py`, `src/rv1/ext_eval.py`, `tests/test_ext_stats.py`, `configs/ext_eval_4c_v1.json`.

**New document:** `docs/STAGE4B3_SUMMARY_CORRECTION_v1.md`.

**Run outputs:**
- `eval_spec.json` with its sidecar;
- `predictions_pad_ufes_20.csv`, `predictions_ddi.csv`;
- `inference_log.json`;
- `results.json`;
- `pad_role_counts.csv`, `pad_eval_tone_table.csv`, `ddi_table.csv`;
- `bootstrap_replicates.npz`;
- `thesis_results_4c.md`, which has its own sidecar;
- three receipts.

**Integrity:** 638 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged. No `__pycache__` was created.

## 8. Remaining work (not started)

- RQ5 adaptation on the frozen PAD partition.
- Seeds 1–2 and the other architectures.
- Full XAI (RQ4).
- RQ2 and RQ3 as planned.

**Stage 4C is complete.**
