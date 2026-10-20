# Stage 11A Report — E4.6 Data Audit and Frozen Evaluation Design

Date: 2026-10-10. Environment: `research_v1/.venv_model`.

**Run:** `runs/e46_design_11a/<run_id>/`.

**Stage records:** `runs/stage11a/`.

**Not done:** feature values, model fitting and cross-validation. **This stage stops before fitting.**

## 1. Authoritative text and what is new

**Protocol 02_rev2 §6.7, E4.6:** does a model built on deployment-available features computed from the clinical image alone improve error prediction over a probability-only baseline? It is to be evaluated with 5-fold cross-validation grouped by case. Features that need the twin image are mechanistic only.

**§9 row 8:** v1 replaces the v0 in-sample "drift" model, which was built from twin-image explanation differences.

**Everything below is a retrospective addition,** frozen in `e46_spec.json` (hashed) before any feature value or model:
- the cohorts and outcome coding;
- the features and models;
- the fold algorithm;
- the metrics and the uncertainty method.

## 2. Question, model and outcome

- **Model:** standard ResNet-50 seed 0 (`303d21e7…`). Frozen threshold **0.17569** (Stage 4A, `ESTIMATED`, bound to the checkpoint).
- **Predictions:** Stage 4A `predictions_derm7pt_all.csv`, clinical rows. The checkpoint and threshold identities and the rule `pred = 1[p ≥ t]` were verified for all 1,011 rows.
- **Outcome:** **error = 1** when the clinical-image classification disagrees with the melanoma label (FN for melanoma, FP for non-melanoma).
- **Use of the label:** it defines the outcome and the fold strata only; it never enters a predictor.

## 3. Audited cohorts (`e46_spec.json → cohorts`; hashed CSVs)

| Cohort | Cases (= images) | Mel / non-mel | Errors (FN / FP) | Role |
|---|---|---|---|---|
| Full clinical, Derm7pt PRIMARY | 1,002 | 249 / 753 | **308** (143 / 165) = 30.7% | Error prevalence in the primary cohort; probability-only context (M0 vs linear) |
| **XAI clinical** (Stage 7A, enriched) | **498** | 249 / 249 | **198** (143 / 55) = 39.8% | **Primary E4.6 comparison** |
| ↳ IG-valid subset (completeness pass) | 394 | 207 / 187 | 158 (107 / 51) | Sensitivity |
| ↳ Grad-CAM non-zero subset | 304 | 180 / 124 | 127 (77 / 50) | Descriptive only |

**Joins and identity**
- Exactly one clinical image per case: `case_id` = `pair_id` = `cluster_id`.
- No duplicate cases.
- Every XAI record joins to its Stage 4A prediction. The 7A and 4A logits differ by at most 1.5e-5, from batch context; **the 4A probability is authoritative**.

**Not required, not reused**
- No dermoscopic-image eligibility is required.
- The paired modality cohorts (373 IG / 275 Grad-CAM) are **not** reused.

**Missingness**
- IG completeness failure: 104/498.
- Grad-CAM zero map: 194/498.
- Every reference has an IG curve: a primary curve, or the stored as-computed curve for an invalid map.

**Enrichment.** The XAI cohort contains **all** FN of the primary cohort but only 55 of its 165 FP. Its error mix (FN-heavy) and its prevalence therefore differ from the full cohort, and from any deployment population.

## 4. Predictors (`feature_dictionary.json`)

**Probability (all models), `prob_basis`:** restricted cubic spline of logit(p) with 4 knots at the 5/35/65/95th percentiles of the **training-fold** logits, giving 3 columns. This nonlinear baseline lets the probability-only model capture errors at both low p (FN) and high p (FP), so that XAI features cannot appear useful merely by adding nonlinearity.

**XAI features (M1), from saved clinical-image artefacts on the CPU; no new GPU computation**

| Feature | Definition |
|---|---|
| `ig_delta_ins` | IG AUC_ins − mean of the 5 random AUC_ins (primary or as-computed curve) |
| `ig_delta_del` | Mean of the 5 random AUC_del − IG AUC_del (same curve rule) |
| `ig_completeness_failure` | Label-free numerical diagnostic (0/1) |
| `ig_conc_top10` | Saliency concentration: sum of the top 10% (5,018 / 50,176) pixel values of the IG positive map ÷ total positive mass; missing if the map is all zero |
| `gradcam_zero` | 1 if the clinical Grad-CAM positive map is all-zero or constant |
| `gradcam_conc_top10` | As `ig_conc_top10`, for Grad-CAM; missing when `gradcam_zero` = 1 |

**Secondary model only (M2):** `d_baseline` = p_full − p_base. This is a model response to the blurred baseline, labelled explicitly as **not** an explanation.

**Excluded, with reasons:**
- the true label, correctness and the prediction-vs-label comparison;
- the dermoscopic twin and any modality difference or drift (mechanistic only);
- masks and localisation;
- **mismatch controls** (their donors are chosen by label);
- **entropy** (a deterministic function of p, not XAI information);
- centre and Sobel controls (to keep the feature set small).

**Leakage guard.** `leakage_check` rejects predictor names containing forbidden fields, and a whitelist lists every field a feature builder may read.

## 5. Models and cross-validation (frozen)

**Model family:** L2-penalised logistic regression (lbfgs, **C = 1.0 fixed**, max_iter 5000, no class weights). **No tuning.**

**Models**

| Model | Predictors | Role |
|---|---|---|
| M0 | `prob_basis` | Probability-only comparator |
| M0-linear | logit(p) only | Reference only, not a comparator |
| M1 | M0 + the 6 XAI features | Primary comparison with M0 |
| M2 | M1 + `d_baseline` | Secondary |

**In-fold processing:** spline knots, median imputation of missing concentration values and standardisation are all fitted on the training folds only. The zero-map indicator carries the missingness.

**Folds**
- k = 5, grouped by case.
- Stratified by label × error.
- Deterministic `stable_seed('e46_folds_v1', case)` order, dealt round-robin.
- Sufficiency check: at least 10 errors and 10 non-errors per fold.

**Fold counts**

| Cohort | Per fold |
|---|---|
| XAI cohort | 99–100 cases, 39–40 errors |
| Full cohort | 200–201 cases, 61–62 errors |
| IG-valid subset (parent folds restricted) | 28–36 errors |
| Grad-CAM non-zero subset (parent folds restricted) | 22–28 errors |

**The same folds and the same cases are used for every matched comparison.**

**Metrics**
- **Primary:** out-of-fold (OOF) AUROC for error.
- **Secondary:** OOF Brier score, OOF log-loss, and calibration intercept/slope (descriptive). Per-fold AUROCs are descriptive only and are **not** five independent experiments.

**Comparison:** ΔAUROC = M1 − M0 (and ΔBrier) on identical OOF predictions.

**Uncertainty:** paired case bootstrap of the fixed OOF predictions (B = 2,000, seed 20261300, percentile 95% CI). This **does not capture refitting or fold-assignment uncertainty.**

**No hypothesis tests.**

**Sensitivity analyses:** the IG-valid subset; M2 vs M0; M0 vs M0-linear on the full cohort.

**Interpretation rule (frozen):** an exploratory statement that XAI features add in-cohort information is made only if the ΔAUROC interval excludes 0 on the primary cohort.

## 6. Scope

This is **exploratory, retrospective internal cross-validation** on a class-enriched, previously examined cohort.
- There is no deployment validation.
- Error prevalence in this cohort is **not** a general-population rate.
- The Derm7pt 12-test Holm family is unchanged.

## 7. Stage 10D clarification (`docs/STAGE10D_CLARIFICATION_NOTE_v1.md`)

1. **Which values are HL.** The faithfulness levels, paired changes and IG − Grad-CAM differences are Hodges–Lehmann estimates. Localisation values are medians or proportions.
2. **Near-zero is not equivalence.** "Near-zero advantage" (IG blur 1.0 Δ_ins HL −0.000 [−0.002, 0.001]) does **not** establish equivalence to random: no margin was prespecified.
3. **Two kinds of counts.** Condition-only eligible counts (Table S1; e.g. Grad-CAM blur 1.0, n = 131) differ from baseline-joint counts (Table S2; n = 128).

## 8. Verification

- **Tests:** `tests/test_e46_design.py`, 6/6 — outcome incl. the threshold boundary, leakage guard (clean set and forbidden names), deterministic, order-independent, one-fold-per-case and balanced folds, duplicate rejection, and insufficiency detection.
- **Receipt** verifies; the spec, both cohort CSVs and the feature dictionary have `.sha256` sidecars.
- **Integrity:** 4739 → 4741 files, **0 changed and 0 removed**. Added: the config and the 10D clarification note.

**Files**
- Code, test and config: `src/rv1/e46_design.py`, `tests/test_e46_design.py`, `configs/e46_design_11a_v1.json`.
- Run dir: `e46_spec.json`, `cohort_full_clinical.csv`, `cohort_xai_clinical_folds.csv`, `feature_dictionary.json`, their sidecars and the receipt.

**Next (Stage 11B, not started):** build the feature table from saved artefacts, then fit and evaluate the cross-validation as frozen here.
