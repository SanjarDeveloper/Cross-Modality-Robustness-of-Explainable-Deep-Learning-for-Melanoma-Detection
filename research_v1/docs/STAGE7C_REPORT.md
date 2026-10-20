# Stage 7C Report — RQ4 Secondary Normalization, Exploratory E4.3, and Alternative-Control Uncertainty

Date: 2026-10-04. This stage was a CPU analysis of the verified Stage 7A and 7B artifacts only.

**Run:** `runs/rq4_secondary_7c/20261004T073730Z_cb20b3f8_c3f30e65/`. The spec and analyze receipts verify.

**Stage records:** `runs/stage7c/`.

## Status

| Item | Result |
|---|---|
| Stage 7B interpretation note | `docs/STAGE7B_INTERPRETATION_NOTE_v1.md`. Frozen 7B files unchanged |
| Specification frozen before results | `analysis_spec.json` (hashed). It records the 7A and 7B receipt output hashes, normalization rules, E4.3 model, interval methods, populations, seeds and the missing-data rule. Retrospective implementation choices are labelled as such |
| Normalization check | normalized contrast = raw contrast / d. Maximum \|diff\| 1.8e-15 |
| Primary populations reused | Grad-CAM 275 and IG 373 matched pairs (checked against Stage 7B) |
| Undefined bootstrap replicates | 0 (median/HL) |
| Rank-deficient E4.3 replicates | 0 |
| Tests | New `test_rq4_secondary.py` (7 tests); `test_rq4_stats` passes |
| Runtime | 121 s |
| Protected inputs | 0 changed, 0 removed |
| Stage 7B results | The primary results and 12-test Holm family are **unchanged**. Nothing here is added to that family |

**Why the full suite was not run:** no shared module changed.

## 1. Retrospective implementation decisions (not preregistered)

| Item | Decision |
|---|---|
| E4.3 population | Stage 7B **raw** primary pairs. The protocol does not restrict E4.3 to the normalization subset |
| E4.3 fit | OLS with uncentred covariates. Pair-bootstrap percentile CIs; rank-deficient replicates counted |
| Diagnostics | Rank, condition number, residual SD, R², residual quantiles, covariate ranges and correlation |
| Normalization bootstrap | Pairs shared within each block, so raw and normalized values come from the same draws |
| Seeds | From 20261020 |
| Alternative controls | Median and HL with pair-bootstrap CIs. For mismatch, random-control contrasts are also computed on the identical subset |

## 2. Normalized scores (02_rev2 §6.5)

**Method**
- d = p_full − p_base, from each image's stored curve endpoints.
- No clipping, no replacement of the denominator, and negative d is never normalized.
- The random-control mean AUCs use the **same** image's endpoints and d.
- Map eligibility and both modalities are required.

**Categories at δ = 0.05** (identical for both explainers, because d depends only on the model and image):

| Group | Defined | Baseline more melanoma-like (d ≤ −δ) | Near-zero denominator |
|---|---|---|---|
| Dermoscopy, melanoma | 232 | 2 | 15 |
| Dermoscopy, non-melanoma | 169 | 9 | 71 |
| Clinical, melanoma | 159 | 10 | 80 |
| Clinical, non-melanoma | 89 | 21 | 139 |

Full tables by explainer and δ are in `normalization_categories.csv`.

**Matched pairs** (primary-eligible pairs that are also normalization-defined in both modalities):

| Explainer | δ = 0.02 | δ = 0.05 | δ = 0.10 |
|---|---|---|---|
| Grad-CAM | 221 (144/77) | **190 (131/59)** | 153 (111/42) |
| IG | 251 (158/93) | **206 (141/65)** | 155 (111/44) |

**Raw vs normalized on the identical subset** (δ = 0.05; HL; `normalized_vs_raw_table.csv`):

| | Raw D | Normalized D [95% CI] |
|---|---|---|
| Grad-CAM insertion | −0.062 | −0.068 [−0.183, 0.038] |
| Grad-CAM deletion | −0.023 | +0.015 [−0.080, 0.104] |
| IG insertion | −0.024 | **+0.140** [0.069, 0.215] |
| IG deletion | −0.101 | +0.034 [−0.020, 0.088] |

For comparison, the original raw-population D values (Stage 7B) were −0.086, −0.027, −0.033 and −0.089.

**Reading**
- **Population change:** the subset is smaller and more melanoma-heavy, while the raw D values stay negative and similar.
- **Rescaling:** dividing by d moves the shift contrasts towards zero, and reverses the sign for IG insertion.
- The clinical penalty in the raw contrasts is therefore partly associated with clinical images' smaller probability range. Normalization **does not remove confidence confounding** (02_rev2 §6.5).
- **Sensitivity thresholds:** δ = 0.02 and 0.10 show the same pattern, except that Grad-CAM insertion D at δ = 0.02 is −0.134 [−0.266, −0.026].
- These results are descriptive, with no tests and no Holm adjustment.

## 3. Exploratory E4.3 (`e43_table.csv`)

**Model:** D = b0 + b1·x1 + b2·x2, where x1 = Δp_full and x2 = Δd (clinical − dermoscopy), fitted on the raw primary pairs.

| | n | Mean D | Median / HL D | b0 [CI] | b1 [CI] | b2 [CI] | R² | Condition number |
|---|---|---|---|---|---|---|---|---|
| Grad-CAM insertion | 275 | −0.086 | −0.072 / −0.086 | −0.043 [−0.075, −0.010] | −1.01 [−1.55, −0.61] | 1.21 [0.80, 1.75] | 0.14 | 29.1 |
| Grad-CAM deletion | 275 | −0.030 | −0.019 / −0.027 | 0.025 [−0.004, 0.054] | 0.62 [0.26, 1.11] | −0.42 [−0.90, −0.05] | 0.15 | 29.1 |
| IG insertion | 373 | −0.036 | −0.029 / −0.033 | 0.001 [−0.015, 0.017] | −0.49 [−0.76, −0.16] | 0.63 [0.30, 0.91] | 0.19 | 31.9 |
| IG deletion | 373 | −0.098 | −0.073 / −0.089 | −0.018 [−0.036, 0.000] | 0.76 [0.34, 1.23] | −0.44 [−0.91, −0.04] | 0.46 | 31.9 |

**Covariates**
- Ranges: x1 from −0.91 to 0.79, x2 from −0.96 to 0.78.
- **corr(x1, x2) = 0.989 for Grad-CAM and 0.991 for IG** (variance inflation about 46 and 56).
- The two slopes have opposite signs and are **not separately interpretable**.
- Pairs with |x1| and |x2| both below 0.05: 26 (Grad-CAM) and 52 (IG).

**Intercept**
- b0 is the predicted D when x1 = x2 = 0. That reference point lies inside the observed ranges and is sparsely supported.
- It is **not** a universally adjusted effect, and it is **not** comparable to HL as the same estimand.

**What this shows and does not show:** the regression shows an association between the shift contrasts and modality-related changes in confidence. It does not show causation, and it does not remove confounding.

## 4. Stronger controls (`alternative_controls_table.csv`; HL [95% CI])

| Explainer | Control | Pairs | Insertion, derm / clin / D | Deletion, derm / clin / D |
|---|---|---|---|---|
| Grad-CAM | centre | 275 | 0.183 [0.162, 0.205] / 0.069 [0.057, 0.082] / −0.096 | **−0.008 [−0.020, 0.002] / −0.020 [−0.026, −0.014]** / −0.015 |
| Grad-CAM | Sobel | 275 | 0.266 / 0.117 / −0.130 (all CIs exclude 0) | **0.005 [−0.012, 0.023] / 0.004 [−0.005, 0.013]** / 0.002 |
| Grad-CAM | mismatch | **146** | 0.107 [0.082, 0.135] / 0.056 [0.031, 0.083] / −0.048 | 0.071 [0.048, 0.097] / 0.033 [0.018, 0.052] / −0.025 [−0.050, 0.002] |
| IG | centre | 373 | 0.107 / 0.078 / −0.010 [−0.032, 0.010] | 0.109 / 0.031 / −0.068 |
| IG | Sobel | 373 | 0.182 / 0.107 / −0.053 | 0.124 / 0.051 / −0.060 |
| IG | mismatch | **280** | 0.119 / 0.069 / −0.035 | 0.149 / 0.071 / −0.056 |

**Random-control context on the identical mismatch subset:**

| Explainer | Insertion, derm / clin | Deletion, derm / clin |
|---|---|---|
| Grad-CAM (146 pairs) | 0.181 / 0.063 | 0.067 / 0.035 |
| IG (280 pairs) | 0.117 / 0.075 | 0.188 / 0.085 |

**Reading**
- **IG** outperformed every stronger control, on both curves and in both modalities: all modality CIs are above 0.
- **Grad-CAM** outperformed every control on **insertion**. On **deletion**, it did **not** outperform the centre prior (clinical HL is negative, with a CI below 0) or Sobel (CIs include 0).
- Outperforming random does not imply outperforming simple structural priors.

**Mismatch caveat:** donor maps are reused across recipients, and the recipient-case bootstrap treats the realized donor assignment as fixed. These intervals are therefore likely too narrow and must not be over-interpreted.

## 5. Standing limitations

- Evidence-dependent exclusions: Grad-CAM all-zero maps in 285 images; IG completeness failures in 138 (13.9%).
- Normalization adds further evidence-dependent missingness.
- The cohort is class-enriched.
- 24 pilot pairs were previously inspected.
- Only seed 0 is analysed.
- The study was developed retrospectively.
- The metrics describe behaviour under the pixel-replacement procedure. They do not describe clinical correctness or causal explanation.

## 6. Files

**New code, tests and config:** `src/rv1/rq4_secondary.py`, `tests/test_rq4_secondary.py`, `configs/rq4_secondary_7c_v1.json`.

**New document:** `docs/STAGE7B_INTERPRETATION_NOTE_v1.md`.

**Run outputs:**
- `analysis_spec.json` with its sidecar;
- `results.json`;
- `per_image_inputs.csv` (categories, d and contrasts per image and explainer);
- `normalization_categories.csv`, `normalized_vs_raw_table.csv`;
- `e43_table.csv`;
- `alternative_controls_table.csv`;
- `bootstrap_replicates.npz`;
- `thesis_results_rq4_secondary.md`, which has its own sidecar;
- the receipts.

**Integrity:** 1081 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged.

## 7. Remaining components (RQ4 is **not** complete)

**RQ4**
- Secondary IG variants: |attr| maps and positive maps with σ = 2 smoothing. These need new curve computation.
- E4.4: melanoma vs non-melanoma subgroup contrast.
- E4.5: Grad-CAM–IG agreement (Spearman correlation, top-10% overlap).
- E4.6: error prediction from clinical-image features.
- The simulated HAM arm (§6.7: 170 + 170 images, severities {0, 0.5, 1}), which depends on the RQ2 perturbation simulator.
- Seeds 1–2 on a 100-pair XAI subset (optional per protocol).

**Other research questions:** RQ2 and RQ3.

**Stage 7C is complete.**
