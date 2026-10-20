# Stage 11A Clarification Note v1 — 2026-10-10 (written before any Stage 11B fitting)

This note binds the Stage 11B execution to the frozen Stage 11A specification. The following are unchanged:
- `docs/STAGE11A_REPORT.md`;
- `e46_spec.json`, the cohort/fold files and the feature dictionary.

## 1. Authoritative run (replaces the report's placeholder path)

**Run:** `runs/e46_design_11a/20261010T054550Z_620b65e3_2926da95/`.

**Hashes**

| File | SHA-256 |
|---|---|
| `e46_spec.json` | `8c4ebedc309b4d5136cf13810d7fb43f07cb6fc45d7f6235eddb414393f838ec` |

The design receipt and all four `.sha256` sidecars verify (checked 2026-10-10, before Stage 11B).

## 2. Reading ΔAUROC = AUROC(M1) − AUROC(M0) (and M2 − M0)

| 95% interval | Reading |
|---|---|
| Entirely **above** zero | Supports an **exploratory positive increment** relative to the specified spline M0, in this cohort |
| Entirely **below** zero | Indicates **worse** out-of-fold discrimination than M0 |
| **Contains** zero | Inconclusive. It does **not** establish equivalence or "no benefit" (no equivalence margin was specified) |

**What the intervals cover.** They come from a paired case bootstrap of fixed out-of-fold predictions. They do not include refitting or fold-assignment uncertainty.

## 3. The IG-valid sensitivity analysis: one interpretation, frozen

The spec text "ig_valid subset (completeness pass), parent folds restricted" is interpreted as follows:

**Refit within the subset.**
- M0 and M1 are re-trained using only IG-valid cases, on the parent fold assignments restricted to those cases (5 folds).
- All fold-local processing is refitted within the subset.
- Out-of-fold predictions are evaluated on the same subset.

**Not done:** restricting the primary-cohort predictions to the subset. This choice is a labelled clarification, not a result-driven choice.

## 4. Other execution details fixed now

- **IG curve per case:** the primary curve if the map is valid. Otherwise the stored as-computed curve is used, and `ig_completeness_failure` = 1 marks it as **not a valid explanation**. No case is dropped and no zero is substituted.
- **Degenerate input stops the run.** If any fold has tied spline knots, an all-missing training column or a zero-variance training column, the run stops with an error and is reported. The model is not modified, and held-out data are not inspected.
- **Convergence:** reported per fold and model (iterations against `max_iter`).

## 5. Solver substitution (disclosed before fitting)

**What the spec says:** the frozen model is "sklearn LogisticRegression (lbfgs, C = 1.0, max_iter 5000)".

**What happened:** scikit-learn is listed in `env/env_inventory_v1.json`, but it is **not importable in `.venv_model`**, the project environment used for every stage. No package is installed and no interpreter is switched.

**What is used instead:** the identical objective, solved exactly by NumPy Newton–Raphson in `e46_cv.fit_logistic`:
- the objective is ½‖w‖² + C·Σ log-loss, with an unpenalised intercept and no class weights;
- standardised inputs;
- convergence when the gradient or step falls below 1e-10;
- iterations and convergence recorded per fold.

**Why the result is the same:** the optimum is unique (the objective is strictly convex), so the solver choice does not change the model.

**Test:** `test_e46_cv.TestSolver` checks equality against a generic scipy optimiser on the same objective (tolerance 1e-5).

**Calibration slope:** the unpenalised calibration-slope fit uses the same solver with no penalty.
