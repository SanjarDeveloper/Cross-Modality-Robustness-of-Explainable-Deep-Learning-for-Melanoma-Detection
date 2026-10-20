# Stage 4B1 Report — Primary RQ1 Paired Analysis (E1.1, E1.2; secondary specificity)

Date: 2026-10-01. This stage was a CPU analysis of the saved Stage 4A predictions only.

**Run:** `runs/analysis_rq1/20261001T184716Z_69709444_0f8862c5/`. Both receipts (spec and analyze) verify.

**Stage records:** `runs/stage4b1/`.

## Status

| Item | Result |
|---|---|
| Specification frozen before any statistic | `analysis_spec.json`, SHA-256 `ed3251ad…4ab1`, written by a separate process. The analysis re-verified the spec hash and every input hash first |
| Inputs | Stage 4A `predictions_derm7pt_all.csv`, `threshold.json` (`469a4d45…`), `receipt_predict.json`, `verification.json`; checkpoint `303d21e7…` |
| Pair reconstruction (by `pair_id`) | 2022 rows → 1011 pairs. Each pair has exactly one dermoscopy and one clinical record, a consistent label and cohort flags, finite probabilities, and matching model and threshold hashes. **pred == (prob ≥ t) for every row** |
| Cohort counts (recomputed from rows) | PRIMARY **1002 pairs / 249 melanoma**; ALL **1011 / 252**. As expected |
| Melanoma table, PRIMARY | 97 both detected / **117 dermoscopy only** / **9 clinical only** / 26 both missed. Matches Stage 4A |
| **E1.1 exact McNemar** | **p = 4.188 × 10⁻²⁵** (log₁₀ p = −24.38). **H1₀ rejected** at α = 0.05 (single test). Direction: dermoscopy > clinical |
| **E1.2 sensitivity difference** | **+43.4 pp [95% CI 36.5, 50.2]** |
| Tests | New `test_rq1_stats.py` (10 tests). Full suite **151/151** |
| Protected inputs | 0 changed, 0 removed |

The `config.py` change was the only shared-module change, which is why the full suite was run (§6).

## 1. Specification (from protocol 02_rev2 §2, §3 and Addendum G)

| Item | Setting |
|---|---|
| Direction | Dermoscopy − clinical |
| E1.1 | Two-sided **exact McNemar**, i.e. the exact binomial test on the discordant pairs: p = min(1, 2·P[X ≤ min(b, c)]), X ~ Bin(b + c, ½). Computed in exact rational arithmetic; no chi-square. α = 0.05; family size 1 |
| E1.2 | Paired sensitivity difference in pp over PRIMARY melanoma pairs |
| Secondary (E1.4, descriptive) | Specificity per modality and the paired difference over PRIMARY non-melanoma pairs |
| ALL cohort | Prespecified **sensitivity analysis**, not a second confirmatory result |
| Bootstrap | Percentile pair (case-cluster) bootstrap, **B = 2000**, 95% |
| Seed | 20261001 (config `bootstrap_seed`), with offsets 0–3 per cohort and estimand |
| RNG | `numpy.random.Generator(PCG64)`, drawing with `integers(0, K, K)` over the sorted `pair_id` list |
| Quantiles | `numpy.quantile(method='linear')` at 0.025 and 0.975 |
| Resampling unit | `pair_id`. Both modalities move together, and repeated draws keep their multiplicity |
| Resampled sets | Sensitivity resamples melanoma pairs; specificity resamples non-melanoma pairs |
| Undefined replicates | Counted, excluded from the quantiles, never redrawn. The CI is flagged if more than 5% are undefined |

**Observed undefined replicates:** **0** in every bootstrap.

**Scope of the intervals:** they are conditional on the fixed model and threshold. They exclude training and threshold-selection uncertainty.

## 2. Results — PRIMARY cohort (1002 pairs)

| Measure | Dermoscopy | Clinical | Difference, dermoscopy − clinical (pp) [95% CI] |
|---|---|---|---|
| Sensitivity (n = 249 melanoma pairs) | 214/249 = 85.9% [81.5, 90.0] | 106/249 = 42.6% [36.5, 48.6] | **+43.4 [36.5, 50.2]** |
| Specificity (n = 753 non-melanoma pairs), *secondary* | 381/753 = 50.6% [47.0, 54.2] | 588/753 = 78.1% [75.2, 81.0] | −27.5 [−31.6, −23.2] |

**Non-melanoma pair outcomes (n = 753):**
- 328 correct under both modalities;
- 53 correct under dermoscopy only;
- 260 correct under clinical only;
- 112 false positives under both.

**Interpretation (02_rev2 §3 categories):** this is a **positive** result.
- p < 0.05, with the direction b > c.
- The whole CI lies well above 0, and above the 10 pp descriptive reference.

## 3. Sensitivity analysis — ALL cohort (1011 pairs)

| Measure | Result |
|---|---|
| Sensitivity | 215/252 vs 107/252 |
| Sensitivity difference | **+42.9 pp [35.7, 50.0]** |
| Melanoma table | 98 / 117 / 9 / 28 |
| McNemar | b and c unchanged, so p = 4.188 × 10⁻²⁵ again |
| Specificity | 383/759 vs 590/759 |
| Specificity difference | −27.3 pp [−31.2, −22.9] |

**Why the denominator changes:**
- The 9 extra pairs are the **same-file pairs**, where both modality slots hold the same image file. They are outside the primary cohort by definition (verified).
- They contain 3 melanoma and 6 non-melanoma pairs.
- Their two predictions are identical (Stage 4A), so they are always concordant. They raise the denominators and the concordant cells, but never b or c.
- They mainly test the pipeline, and are not a genuine modality comparison.

## 4. Interpretation limits

- These results show an association and a performance difference for **one model (seed 0), one threshold and one dataset**. They are **not** a causal modality effect, because acquisition, framing and case mix are confounded.
- The "clinical" images are clinical close-up photographs. **They are not verified smartphone images.**
- "Non-melanoma" includes other malignant and premalignant lesions; it does not mean benign.
- The unit is the case/pair. Derm7pt has no patient identifiers, so patient-level independence is **not** claimed.
- **Derm7pt exposure:** the dataset was used in legacy work, and 24 pairs were inspected in the Stage 3B numerical XAI pilots. No weights or thresholds were tuned on Derm7pt, but it is **not pristine**.
- **HAM VAL** was used for checkpoint selection and the threshold.
- The findings are not generalised beyond this evaluated setting.

## 5. Tests (`tests/test_rq1_stats.py`, 10 tests)

| Area | What is checked |
|---|---|
| Exact McNemar | Equals an **independent brute-force** calculation for every (b, c) in 0..12 × 0..12 (within 1e-12); agrees with `scipy.stats.binomtest` for (117, 9), (10, 3) and (0, 5); gives known values (1, 0) → 1 and (5, 0) → 1/16 |
| Zero discordant pairs | Gives p = 1 |
| Swapping modalities | The p-value is identical (exact rational) and the effect changes sign |
| Pair resampling | K draws per replicate, repeated pairs kept, and both modalities travel with each pair |
| Seed | The same seed reproduces the replicates and the CI; a different seed changes them |
| Undefined replicates | Counted, not redrawn, and the CI is flagged when more than 5% are undefined |
| Rejections | Duplicate record, duplicate modality, incomplete pair, inconsistent label, pred ≠ (p ≥ t), non-finite probability and identity mismatch are all rejected; pairing does not depend on row order |

The full suite (151 tests) was run because the shared `config.py` changed (§6).

## 6. Files

**Code and config**
- New: `src/rv1/rq1_stats.py`, `src/rv1/rq1_primary.py`, `tests/test_rq1_stats.py`, `configs/analysis_rq1_primary_v1.json`.
- Changed: `src/rv1/config.py`, which adds the purpose `analysis_v1`. The prior copy is `runs/stage4b1/config_before.py.txt`.

**Run outputs:**
- `analysis_spec.json` with its sidecar;
- `paired_table.csv` (1011 pairs, with `pair_id`, record IDs, probabilities and predictions);
- `results.json`, the machine-readable results;
- `bootstrap_replicates.npz`, holding all 2000 replicates for each bootstrap;
- `thesis_results_rq1_primary.md`, the thesis table and results paragraph. It was written after the analyze receipt, so it is not listed in that receipt; it has its own SHA-256 sidecar;
- the receipts.

**Integrity:** 575 files before and after, with **0 changed, 0 removed and 0 added**. The cache `.npy` files are unchanged. No `__pycache__` was created.

## 7. Remaining RQ1 work (RQ1 is **not** complete)

1. **E1.3** paired AUROC difference on the 1002 primary pairs (DeLong, secondary and unadjusted, with a case-bootstrap CI).
2. Other prespecified secondary and descriptive metrics: E1.4 all-case correctness McNemar with its decomposition, and ECE by modality.
3. Uncertainty for **HAM TEST** (cluster bootstrap CIs).
4. **PAD-UFES-20 (E1.5)** and **DDI** evaluation.
5. **Seeds 1–2**: E1.1 and E1.2 reported as spread.
6. Other architectures, as planned.

**Stage 4B1 is complete.**
