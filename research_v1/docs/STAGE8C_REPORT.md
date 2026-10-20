# Stage 8C Report — Descriptive RQ2 Statistics

Date: 2026-10-04. This stage was a CPU analysis of the verified Stage 8B predictions only.

**Run:** `runs/rq2_stats_8c/20261004T082234Z_cbc0aad2_2529e34b/`. The spec and analyze receipts verify.

**Stage records:** `runs/stage8c/`.

## Status

| Item | Result |
|---|---|
| Specification frozen before computation | `analysis_spec.json` (hashed). It records the Stage 8B verify-receipt outputs and the hashes of all 45 prediction files, estimands, units, directions, seed (20261050), quantile methods, censoring rule and ranking tie rule |
| Input validation (all 45 units, by identifier) | 1548 images / 170 melanoma / 1155 lesions; identical labels and clusters in identical order; checkpoint and threshold identities; pred == (p ≥ t); finite p in [0, 1]; no duplicate keys. **The 5 severity-0 copies are identical and are treated as one baseline** |
| Baseline | TP/FN/TN/FP 157/13/911/467; sensitivity 0.924; specificity 0.661; AUROC 0.913 |
| Bootstrap | Lesion-cluster, B = 2000. The **same draws** are used for the baseline, every severity and every factor. **0 undefined replicates** for every metric |
| Tests | New `test_rq2_stats.py` (8 tests). Relevant suites pass |
| Runtime | 13 s |
| Protected inputs | 0 changed, 0 removed |

**Why the full suite was not run:** no shared module changed.

**Not done:** inference and hypothesis tests.

## 1. Frozen definitions

| Quantity | Definition |
|---|---|
| Metrics | Image-level sensitivity and specificity at the frozen t (p ≥ t); AUROC from average ranks |
| Changes | Sensitivity in pp; AUROC on the 0–1 scale |
| Intervals | Pointwise 95% percentile intervals |
| Worst-case loss | loss(s) = m(0) − m(s) over the 9-level grid **including s = 0**, so the loss is ≥ 0, and improvements remain negative losses at individual levels. Tie rule: the smallest severity. **Recomputed within every replicate** |
| Crossing | The first s with sens(s) ≤ sens(0) − 0.10, with a 1e-12 tolerance for float exact hits. Linear interpolation between the first bracketing adjacent levels, with no monotone smoothing. Recomputed against each replicate's own baseline |
| Crossing categories | Finite crossing / no crossing in the grid (right-censored, ">1.0") / undefined |
| Censored quantiles | Empirical type-1 quantiles, with censored values ordered after the finite ones. A quantile landing in the censored group is reported as ">1.0" |
| Ranking | Share of valid replicates in which each factor has the largest worst-case loss. Ties split equally. Replicates with any undefined value are excluded and counted |

## 2. Per-level results (`per_level_table.csv`; plots `plot_d_sens_pp.png`, `plot_d_auroc.png`)

**Selected levels** (Δ sensitivity in pp [CI] | Δ AUROC [CI]):

| Factor | s = 0.25 | s = 0.5 | s = 0.75 | s = 1.0 |
|---|---|---|---|---|
| Blur | −5.3 [−10.6, 0.0] \| −0.011 | −27.6 [−35.5, −20.0] \| −0.033 | −52.4 \| −0.046 | **−72.9 [−80.4, −64.8] \| −0.066 [−0.099, −0.035]** |
| Illumination | −3.5 \| −0.030 | −13.5 \| −0.114 | −24.1 \| −0.202 | −34.7 [−43.6, −25.5] \| **−0.272 [−0.315, −0.229]** |
| Compression | **+4.1 [0.9, 8.1]** \| +0.001 | +4.1 \| +0.001 | +5.3 \| −0.008 | +2.4 [−2.4, 7.4] \| −0.052 [−0.079, −0.027] |
| Scale | −1.8 \| −0.002 | −8.8 \| −0.027 | −14.7 \| −0.047 | −22.4 [−29.8, −15.2] \| −0.078 [−0.119, −0.041] |
| Nuisance | +1.2 \| +0.004 | −1.8 \| −0.001 | −3.5 \| −0.005 | −7.1 [−11.9, −2.4] \| −0.008 [−0.023, 0.008] |

**Specificity at s = 1.0**

| Factor | Specificity | FP |
|---|---|---|
| Blur | 0.983 | 23 |
| Illumination | 0.582 | 576 |
| Compression | 0.565 | 599 |
| Scale | 0.831 | 233 |
| Nuisance | 0.787 | 294 |

Baseline specificity is 0.661 (467 FP).

**Shifts in the operating point.** Blur and scale lower the melanoma probabilities overall: fewer positives of both kinds. Illumination and compression raise false positives.

**Monotonicity.** Performance is **not monotone** for every factor. Compression raises sensitivity at moderate levels, and nuisance does so at s = 0.125–0.25.

## 3. Worst-case loss over the tested grid (`worst_case_table.csv`)

| Factor | Sensitivity, pp [CI] (observed worst level) | AUROC [CI] (level) |
|---|---|---|
| Blur | 72.9 [64.8, 80.4] (1.0) | 0.066 [0.035, 0.099] (1.0) |
| Illumination | 34.7 [25.5, 43.6] (1.0) | 0.272 [0.229, 0.315] (1.0) |
| Compression | **0.0 [0.0, 2.4]** (s = 0: no level below baseline) | 0.052 [0.027, 0.079] (1.0) |
| Scale | 22.4 [15.3, 29.8] (1.0) | 0.078 [0.041, 0.119] (1.0) |
| Nuisance | 7.1 [2.4, 11.9] (1.0) | 0.008 [0.000, 0.024] (0.875) |

**The worst level is not fixed across replicates.** The distribution of worst levels over replicates is stored per factor. For example, nuisance AUROC: 1.0 in 769 replicates, 0.875 in 806, and s = 0 (no loss) in 226. This is why the maximum is recomputed in every replicate.

These are worst losses **over the sampled grid**. They are not worst losses over all severities, images or acquisition conditions.

## 4. Ten-percentage-point sensitivity crossing (`crossing_table.csv`)

| Factor | Observed | Valid / undefined replicates | No crossing | Quantiles 2.5% / 50% / 97.5% |
|---|---|---|---|---|
| Blur | 0.295 | 2000 / 0 | 0% | 0.243 / 0.295 / 0.361 |
| Illumination | 0.442 | 2000 / 0 | 0% | 0.320 / 0.443 / 0.580 |
| Compression | >1.0 | 2000 / 0 | **100%** | >1.0 / >1.0 / >1.0 |
| Scale | 0.562 | 2000 / 0 | 0% | 0.372 / 0.562 / 0.830 |
| Nuisance | >1.0 | 2000 / 0 | **87.9%** | 0.913 / >1.0 / >1.0 |

**How to read this**
- ">1.0" means **not observed within the tested grid**. It is not a prediction of behaviour beyond s = 1.
- Interpolated crossings are markers between measured levels, not additional measurements.

## 5. Descriptive ranking (`ranking_table.csv`)

| Metric | Shares of the largest worst-case loss |
|---|---|
| Sensitivity | **blur 1.00**; all others 0 |
| AUROC | **illumination 1.00**; all others 0 |

For both metrics there were 2000 valid replicates, 0 excluded and 0 ties.

These are bootstrap ranking frequencies under the chosen parameter ranges. They are **not** probabilities that a real-world mechanism is the most harmful.

## 6. Interpretation and limits

- Performance does not decline monotonically for every factor, and some factors improve sensitivity at some levels.
- Rankings depend on the chosen parameter ranges: radius 7 px, JPEG quality 18, zoom 2.4, and so on.
- The factors are bundled: scale (size, position, synthetic background and seams), illumination (brightness, contrast and colour), and nuisance (glare and hair). They do not isolate individual mechanisms.
- The changes are synthetic and not calibrated to smartphone acquisition.
- Pointwise intervals are **not** simultaneous bands.
- One model (seed 0) with a fixed threshold, on in-distribution HAM TEST dermoscopic images. No causal cross-dataset claim is made.

## 7. Tests (`tests/test_rq2_stats.py`, 8 tests)

| Area | What is checked |
|---|---|
| Worst-case loss | Zero when every level equals the baseline; improvements never give a negative worst loss; tie rule; undefined input gives an undefined result |
| Worst level | Changes between replicates |
| Crossing | Interpolated; exact boundary (157/170 → 140/170 hits 0.25); nonmonotone first crossing despite later recovery; no crossing → censored; undefined input → undefined (kept distinct from censored) |
| Censored quantiles | Mixed inputs; all-censored input; empty input |
| Ranking | Tie split, exclusions, shares summing to 1 |
| Lesion draws | Multiplicity, with identical images across units |
| AUROC | Rank-based value equals the Mann–Whitney implementation; a single class gives undefined |
| Inputs | A wrong prediction file is rejected |

## 8. Files

**New code, tests and config:** `src/rv1/rq2_stats.py`, `tests/test_rq2_stats.py`, `configs/rq2_stats_8c_v1.json`.

**Run outputs:**
- `analysis_spec.json` with its sidecar;
- `results.json`;
- `per_level_table.csv`, `worst_case_table.csv`, `crossing_table.csv`, `ranking_table.csv`;
- `bootstrap_replicates.npz`;
- `plot_d_sens_pp.png`, `plot_d_auroc.png`;
- `thesis_results_rq2.md`, which has its own sidecar;
- the receipts.

**Integrity:** 1291 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged.

## 9. Remaining (not started)

- The simulated XAI arm (§6.7, HAM 170 + 170 images, severities {0, 0.5, 1}).
- Secondary IG variants and E4.6.
- RQ3.
- Optional additional-seed XAI.

**Stage 8C is complete.**
