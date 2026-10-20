# Stage 4B2 Report — RQ1 E1.3: Secondary Paired AUROC Analysis

Date: 2026-10-01. This stage was a CPU analysis of the saved Stage 4A predictions only.

**Valid run:** `runs/analysis_rq1_auroc/20261001T185601Z_0984a1d2_5092cae2/`. Spec and analyze receipts verify.

**Stage records:** `runs/stage4b2/`.

## Status

| Item | Result |
|---|---|
| Clarification of the 4B1 interpretation | `docs/STAGE4B1_CLARIFICATION_v1.md` (new; frozen 4B1 files unchanged) |
| Specification frozen before computation | `analysis_spec.json` (SHA-256 `1202db48…943d`, written in a separate process). It records the input hashes, cohorts, model, direction, bootstrap settings, and the DeLong source and implementation hash |
| Input validation | Pairing by `pair_id` (the Stage 4B1 `build_pairs`). Labels, complete pairs, finite probabilities within [0, 1], model and threshold identity, and pred == (p ≥ t) all checked. PRIMARY 1002/249; ALL 1011/252 |
| AUROC cross-check (`scipy.stats.mannwhitneyu`) | Agrees to within 1e-12. Values equal the Stage 4A descriptive AUROCs |
| **E1.3 (PRIMARY)** | **ΔAUROC = +0.109 [95% CI 0.066, 0.154]; paired DeLong z = 4.78, p = 1.74 × 10⁻⁶** (secondary, unadjusted) |
| Undefined bootstrap replicates | 0 / 2000 (both cohorts) |
| Tests | New `test_auroc_paired.py` (8 tests). Relevant suites (`auroc_paired`, `rq1_stats`, `threshold`) pass |
| Protected inputs | 0 changed, 0 removed |

**One failed attempt, preserved:**
- Run `20261001T185544Z_0984a1d2_de498fb6` crashed while writing `results.json`, because a numpy bool could not be serialised to JSON. It left a truncated `results.json` (1088 bytes) and no analyze receipt.
- It is kept unchanged as failed evidence and **must not be used**.
- After a serialisation-only fix (`default=lambda o: o.item()`), the spec was re-frozen in a new run and the analysis was rerun. The DeLong module and the inputs were unchanged.

**Why only the relevant suites were run:** no shared module changed. `config.py` and `rq1_stats.py` are untouched, so the full suite was not required.

## 1. Specification

| Item | Setting |
|---|---|
| Analysis cohort | PRIMARY: 1002 pairs, 249 melanoma |
| Sensitivity analysis | ALL: 1011 pairs |
| Excluded from filtering | Correctness, XAI eligibility and pilot membership are not used to filter anything |
| Score | The continuous probability |
| Direction | AUROC_dermoscopy − AUROC_clinical, on the 0–1 scale |
| AUROC | Mann–Whitney, with tied positive–negative pairs counted as ½ |
| **DeLong** | Correlated (paired) DeLong, DeLong & Clarke-Pearson (1988, *Biometrics* 44:837–845), computed with the midrank form of Sun & Xu (2014, *IEEE SPL* 21:1389–1393) |
| DeLong implementation | **Own code**, `src/rv1/auroc_paired.py`; no reused code or licence. Its hash is recorded in the spec and re-verified |
| DeLong test | Two-sided normal z-test; **secondary and unadjusted** |
| Zero variance | z and p are labelled undefined, never replaced |
| Bootstrap | **Ordinary case/pair cluster bootstrap over all pairs (not class-stratified)**, as 02_rev2 §2/§3 specifies, B = 2000, percentile 95% |
| Seed | 20261003 (+0 for PRIMARY, +1 for ALL) |
| RNG | `Generator(PCG64)` with `integers(0, K, K)` over the sorted `pair_id` list |
| Quantiles | `numpy.quantile(method='linear')` |
| Undefined replicates | One-class replicates counted, not redrawn, excluded from the quantiles; flagged if more than 5% |

## 2. Results

| Cohort | AUROC dermoscopy [95% CI] | AUROC clinical [95% CI] | ΔAUROC [95% CI] | DeLong SE | Correlation of placement values | z | p (two-sided) |
|---|---|---|---|---|---|---|---|
| **PRIMARY** (1002 pairs; 249 mel / 753 non-mel) | 0.7770 [0.7410, 0.8116] | 0.6677 [0.6291, 0.7064] | **+0.1093 [0.0664, 0.1541]** | 0.0229 | 0.26 | 4.78 | **1.74 × 10⁻⁶** |
| ALL (1011; sensitivity analysis) | 0.7729 [0.7401, 0.8066] | 0.6650 [0.6263, 0.7031] | +0.1079 [0.0645, 0.1541] | 0.0227 | 0.27 | 4.76 | 1.92 × 10⁻⁶ |

**The DeLong covariance was kept.** The placement-value covariance between the modalities is positive (8.9 × 10⁻⁵ in the PRIMARY cohort). An independent-samples variance would therefore have overstated the SE.

**Status of this result.** E1.3 is **secondary**. Its p-value is unadjusted and is not part of the primary family; the only primary test remains E1.1.

## 3. Interpretation

| Measure | What it describes |
|---|---|
| **Sensitivity** (E1.1/E1.2, Stage 4B1) | Detection of melanoma at the **frozen threshold** |
| **AUROC** (E1.3) | **Ranking** of melanomas above non-melanomas across all thresholds |

Both point the same way: the clinical images are worse.

**Limits:**
- Neither measure establishes **calibration**, and neither establishes a **causal modality effect**. Case composition is matched within pairs, but acquisition and framing are inseparable from modality.
- The "clinical" images are close-up photographs. They are not verified smartphone images.
- The limitations remain: a single model and seed, Derm7pt's prior exposure (legacy work and the Stage 3B XAI pilot), and unknown patient-level dependence (no patient identifiers).
- The intervals are conditional on the fixed model.
- The findings are not generalised beyond this setting.

## 4. Tests (`tests/test_auroc_paired.py`, 8 tests)

| Area | What is checked |
|---|---|
| Known AUROCs | Including ties (0.5 credit), with agreement with `mannwhitneyu` under heavy ties |
| Paired variance and covariance | Equal to an **independent O(mn) direct placement-value calculation** to within 1e-12, with and without ties; covariance is non-zero |
| Swapping modalities | Δ reverses sign and p is unchanged |
| Identical score vectors | Δ = 0 and Var = 0, with p undefined and labelled. No division error |
| One class | Gives `undefined` |
| Plan 03_rev2 §14 criterion | DeLong variance within 15% of a 5000-replicate bootstrap on synthetic data |
| Pair bootstrap | Both modalities from the same draw; one-class replicates counted (not redrawn); reproducible under a fixed seed |
| Row order | Identifier-based pairing makes the results invariant to row order |

## 5. Files

**New code and config:**
- `src/rv1/auroc_paired.py`, `src/rv1/rq1_auroc.py`, `tests/test_auroc_paired.py`;
- `configs/analysis_rq1_auroc_v1.json`;
- `docs/STAGE4B1_CLARIFICATION_v1.md`.

**Run outputs:**
- `analysis_spec.json` with its sidecar;
- `results.json`;
- `bootstrap_replicates.npz` (2000 × 3 per cohort);
- `thesis_results_rq1_e13_auroc.md`, which was written after the receipt and has its own sidecar;
- the receipts.

**Integrity:** 595 files before and after, with **0 changed, 0 removed and 0 added**. The cache `.npy` files are unchanged. No `__pycache__` was created.

## 6. Remaining RQ1 work (RQ1 is **not** complete)

- **E1.4:** the all-case correctness McNemar with its decomposition, and **ECE** by modality.
- **HAM TEST** uncertainty intervals.
- **PAD-UFES-20 (E1.5)** and **DDI**.
- **Seeds 1–2** and other architectures.
- Full XAI (RQ4) is separate.

**Stage 4B2 is complete.**
