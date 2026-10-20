# Stage 11B Interpretation Note v1 — 2026-10-10

This note gives the preferred wording for three Stage 11B statements. The following are unchanged:
- `docs/STAGE11B_REPORT.md`;
- `runs/stage11b/thesis_results_e46.md` (and its `.sha256`);
- all results.

## 1. Main E4.6 result

**Preferred wording:** "No improvement in out-of-fold error discrimination was demonstrated relative to the specified spline probability baseline."

- M1 − M0 ΔAUROC: −0.024 [−0.060, 0.012].
- M2 − M0: −0.028 [−0.064, 0.008].
- Both intervals contain zero, so the result is inconclusive.

**Do not write** that the clinical XAI features "add no information", or anything that implies such a general absence. The analysis compares against one specified baseline with one fixed model family and feature set.

## 2. Calibration slopes

The lower calibration slopes of M1 and M2 (0.79 and 0.76, against 0.96 for M0) are **compatible with** overfitting from the additional features. They do not establish that overfitting is the cause.

## 3. Solver test

`test_e46_cv.TestSolver` verifies that the NumPy Newton–Raphson solver reaches the same optimum as a generic scipy optimiser **on the implemented objective** (½‖w‖² + C·Σ log-loss, C = 1, unpenalised intercept), to 1e-5.

It does **not** establish that the code is equivalent to scikit-learn in every implementation detail, for example scaling conventions, stopping rules or edge-case handling. The equivalence claim is limited to the objective function.
