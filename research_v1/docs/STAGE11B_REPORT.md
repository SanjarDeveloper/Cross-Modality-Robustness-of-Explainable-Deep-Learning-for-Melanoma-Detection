# Stage 11B Report — E4.6 Feature Table and Frozen Cross-Validation

Date: 2026-10-10. Environment: `research_v1/.venv_model`. CPU only.

**Run:** `runs/e46_cv_11b/20261010T055649Z_e04857d3_8f13c4ef/`.

**Stage records:** `runs/stage11b/`.

**Authoritative design:** `runs/e46_design_11a/20261010T054550Z_620b65e3_2926da95/` (`e46_spec.json` SHA-256 `8c4ebedc…`). It verified, together with all of its sidecars.

**Not done:** GPU work, melanoma-model training, threshold changes, tests, or changes to the Holm family.

## 1. Before fitting

**Clarification note v1 (`docs/STAGE11A_CLARIFICATION_NOTE_v1.md`)**
- **§1** binds the run path and hash.
- **§2** fixes how ΔAUROC is read:
  - interval entirely above 0 → exploratory increment relative to the spline M0;
  - entirely below 0 → worse;
  - containing 0 → inconclusive, which does **not** mean equivalence or no benefit.
- **§3** freezes the IG-valid sensitivity analysis as a **refit within the subset** on the restricted parent folds.
- **§4** fixes the curve rule and the stop-on-degeneracy rule.
- **§5 discloses a solver substitution.** scikit-learn is listed in the environment inventory, but it **is not importable in `.venv_model`**. The identical objective (½‖w‖² + C·Σ log-loss, C = 1, unpenalised intercept) is solved exactly with NumPy Newton–Raphson. Its equality with a generic scipy optimiser is tested.

**Tests:** `tests/test_e46_cv.py`, **9/9, run before any fitting**:
- **spline:** the Harrell RCS formula, linearity beyond the last knot, and degenerate knots raising an error;
- **fold-local processing:** flipping held-out labels leaves held-out predictions and fitted parameters unchanged; extreme held-out values do not change knots, medians or means;
- **out-of-fold predictions:** exactly one per case, with identical populations for all models;
- **explicit failures:** an all-missing column or a zero-variance column stops the run;
- **solver:** Newton–Raphson equals the generic optimiser;
- **metric and comparison directions:** including A − B > 0 when A is better, and a zero difference for identical models;
- **calibration:** perfect calibration recovered;
- **real feature dependencies:** on a Stage 7A record, flipping the label, changing `pair_id`, and adding mismatch or dermoscopy-twin fields leave the features **identical**. The sanitised record contains no label, pair, mismatch, donor or dermoscopy field, and changing a random-control AUC changes `ig_delta_ins` by exactly that amount.

`test_e46_design` 6/6 also passes.

## 2. Feature table (`feature_table.csv` + `.sha256`, `feature_meta.json`)

**Coverage:** 498 / 498 cases from saved artefacts only:
- Stage 4A logit and probability;
- Stage 7A clinical records and `maps.npz`, with chunk manifests hashed in the meta file.

**Provenance:** every row carries a JSON field with the source and missingness reason of each feature.

**As-computed IG curves:** 104 rows (completeness failures). They are labelled `NOT a valid explanation` and flagged by `ig_completeness_failure = 1`. None was dropped and no zero was substituted.

**Missing values:** only `gradcam_conc_top10`, for the 194 all-zero Grad-CAM maps. These were median-imputed in-fold, and `gradcam_zero` carries the missingness.

## 3. Execution history (fully disclosed)

1. **Attempt 1 stopped as frozen.** The IG-valid refit hit a zero-variance column: `ig_completeness_failure` is identically 0 in a subset defined by completeness passing.
   - Nothing was published and no metric was seen. The log is kept as `phase_cv_attempt1_failed.txt`.
2. **Clarification note v2** (`docs/STAGE11A_CLARIFICATION_NOTE_v2.md`): the structurally constant indicator is omitted from M1 **in that subset refit only**. Everything else is unchanged.
3. **Attempt 2:** exit code 0, `receipt_cv` verifies, **all fold models converged**, 0 undefined bootstrap replicates.

**Receipt note.** The features receipt now reports a source-hash mismatch for `e46_cv.py`, because the shared module was edited for the subset change after the features phase. The feature-builder functions are unchanged, and the feature table still verifies against its own sidecar.

## 4. Results (`cv_results.json`, `oof_predictions.csv`, `fold_models.json`)

**Primary: XAI clinical cohort (498 cases, 198 errors; identical folds)**

| Model | OOF AUROC [95% CI] | Brier | Log-loss | Calibration intercept / slope |
|---|---|---|---|---|
| M0 (spline) | **0.640** [0.591, 0.691] | 0.230 | 0.652 | 0.005 / 0.96 |
| M0-linear (reference) | 0.490 [0.440, 0.541] | 0.240 | 0.672 | 0.000 / −3.81 |
| M1 (+ XAI) | 0.615 [0.564, 0.664] | 0.230 | 0.653 | 0.009 / 0.79 |
| M2 (+ XAI + d_baseline) | 0.612 [0.560, 0.661] | 0.231 | 0.655 | 0.008 / 0.76 |

**Paired differences**

| Comparison | ΔAUROC [95% CI] | ΔBrier [95% CI] | Reading |
|---|---|---|---|
| **M1 − M0** | **−0.024 [−0.060, +0.012]** | +0.0005 [−0.0053, +0.0063] | Inconclusive |
| M2 − M0 | −0.028 [−0.064, +0.008] | +0.0012 [−0.0047, +0.0072] | Inconclusive |
| M0 − M0-linear | +0.150 [0.075, 0.223] | −0.0099 [−0.0171, −0.0020] | The nonlinear baseline matters |

**Sensitivity: IG-valid refit (394 cases, 158 errors)**

| Model | OOF AUROC |
|---|---|
| M0 | 0.651 |
| M1 (without the constant indicator) | 0.634 |

ΔAUROC = −0.016 [−0.060, +0.026], inconclusive.

**Full Derm7pt primary clinical cohort (1,002 cases, 308 errors; probability only)**

| Model | OOF AUROC |
|---|---|
| M0 | 0.726 |
| M0-linear | 0.726 |

ΔAUROC = −0.001 [−0.004, +0.003].

**Per-fold AUROCs** are in `cv_results.json`; they are descriptive only, not independent experiments.

**Out-of-fold predictions saved:** the primary and IG-valid cases (`oof_predictions.csv`). The full-cohort predictions are summarised in `cv_results.json` but not saved per case. They serve only as probability-only context.

## 5. Interpretation (exploratory)

**No added discrimination.** Relative to this specific spline probability baseline, the clinical-image XAI feature set did **not** improve out-of-fold error discrimination in the enriched cohort. The point estimate was slightly negative, and the interval includes zero.

**What this is not:**
- It is **not** a demonstration that XAI features contain no information beyond every function of probability.
- It is not a test of other models, feature sets or cohorts.

**Combined feature set.** The numerical diagnostics and the saliency features were evaluated as one set; no ablation attributes effects to either. M2 also adds a model-response feature, so the two cannot be separated there either.

**The flexible baseline was decisive.** In the enriched cohort, errors sit at both ends of the probability scale (FN at low p, FP at high p), so a linear baseline (AUROC 0.49) would have made almost any added flexible features look useful.

**Calibration.** The XAI models had lower calibration slopes (0.79 / 0.76 against 0.96), consistent with mild overfitting from the extra features.

**Limitations:**
- the cohort is enriched and previously examined, with an FN-heavy error mix;
- the error prevalence is not generalisable;
- there is a single model and a single seed;
- the bootstrap uses fixed out-of-fold predictions, so it excludes refitting and fold-assignment uncertainty;
- no deployment, causal or clinical claims are made.

**Thesis-ready text:** `runs/stage11b/thesis_results_e46.md` (Tables E1–E2; numbers checked against `cv_results.json`).

## 6. Integrity and files

**Integrity:** 4757 → 4759 files, **0 changed and 0 removed**. Added: `configs/e46_cv_11b_v1.json` and `docs/STAGE11A_CLARIFICATION_NOTE_v2.md`. The v1 note was written before the snapshot.

**Output hashes:** `runs/stage11b/output_hashes.txt`.

**Code and tests:** `src/rv1/e46_cv.py`, `tests/test_e46_cv.py`.

**Run dir:**
- `feature_table.csv` and `feature_meta.json`, with sidecars;
- `cv_results.json`, `fold_models.json` and `oof_predictions.csv`;
- the receipts.

**`runs/stage11b/`:**
- `phase_cv_attempt1_failed.txt` and `status_attempt1.txt`;
- `phase_cv.txt` and `status.txt`;
- the thesis text and the integrity files.

**E4.6 evaluation is complete. Stage 11B stops here.**
