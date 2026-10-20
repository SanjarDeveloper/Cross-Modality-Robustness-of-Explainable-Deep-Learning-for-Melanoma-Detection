# Stage 7B Report — Primary RQ4 Statistics and As-Computed Sensitivity Analysis

Date: 2026-10-04. This stage was a CPU analysis of the verified Stage 7A v2 artifacts only.

**Run:** `runs/rq4_stats_7b/20261004T072324Z_ccc51a7d_0c493e02/`. The spec and analyze receipts verify.

**Stage records:** `runs/stage7b/`.

## Status

| Item | Result |
|---|---|
| Test-discovery discrepancy | **Resolved** (§1). `test_xai_prod.py` now runs **8** tests directly and under discovery, with 0 ResourceWarnings |
| Contract frozen before inference | `analysis_spec.json` (hashed; separate process). It records the input hashes of the v2 artifacts and the full Stage 7A audit-receipt output list, inclusion rules, estimands, Wilcoxon settings (SciPy **1.15.3**), Holm family, gate, bootstrap seeds and quantile method |
| Invalid v1 artifacts | Excluded by construction: only the listed v2 files and the receipt-verified `chunk_*` and `mismatch2_*` records are read. `invalid_v1_donor_bug/` is never opened |
| Production-wide IG stop rule | **None** exists in the protocol or addenda. 02_rev2 §6.6 requires only that the failure rate be reported, and the convergence criterion is a pilot rule. IG m = 512 stayed fixed |
| Primary populations (recomputed) | Grad-CAM **275** pairs; IG **373** pairs, matching Stage 7A. Both modalities were enforced. `scores_v2` agrees with the chunk records to 0.0 |
| **12-test family** | All 12 Holm-rejected at 0.05. All 4 gates passed |
| As-computed sensitivity | Covers **498/498** pairs for both explainers. Non-primary and descriptive |
| Undefined bootstrap replicates | 0 |
| Tests | New `test_rq4_stats.py` (9 tests). `test_xai_prod.py` has 8. Full suite **207/207**, run because the shared `xai_prod.py` was edited (§1) |
| Runtime | 86 s wall-clock (spec plus analysis) |
| Protected inputs | 0 changed, 0 removed |

## 1. Test-discovery fix (Stage 7A)

**Cause.** `TestV2Fixes` (the 2 regression tests) had been appended **after** the `if __name__ == "__main__": unittest.main()` block.
- Running the file directly executed `unittest.main()` before the class was defined, so only **6** tests ran.
- Under `python -m unittest tests.test_xai_prod`, the class *was* collected and 8 tests ran. The "8" in the 7A report was correct for discovery, but not for direct execution.

**Fix.** The `__main__` block was moved to the end of the file. Both execution modes now run 8 tests:

| Class | Tests |
|---|---|
| `TestAtomicResume` | `test_interrupted_write_quarantined`, `test_publish_and_verify` |
| `TestDonors` | `test_full_cohort_mapping` |
| `TestEligibilityGuard` | `test_invalid_maps_never_primary` |
| `TestRefsAndChunks` | `test_chunk_plan_frozen`, `test_full_coverage_unique` |
| `TestV2Fixes` | `test_donor_eligible_reads_explainer_record`, `test_result_dir_filter` |

**ResourceWarnings.** Unclosed-file warnings (14 occurrences) were fixed with context managers:
- in `xai_prod.py`, through the helpers `_read_bytes`, `_read_json` and `_read_csv_rows`;
- in the test file.

The computation semantics are unchanged; only file-reading idioms changed.

**Copies and hashes**
- Prior copies: `runs/stage7b/xai_prod_before_7b.py.txt` and `test_xai_prod_before_7b.py.txt`.
- `xai_prod.py`: `cae69f46…` → `e4d268fa…`.

**Note for future runs:** Stage 7A's `spec.json` and amendment list the old source hash. Any future Stage 7A phase run on that directory would refuse to start until a new amendment records this change. Stage 7B only reads artifacts, so it is unaffected.

## 2. Frozen statistical contract (summary)

**Primary population**
- Matched eligible pairs: the map is primary-eligible in both modalities, **and** the random control is available (5 finite AUCs) in both.
- No filtering by correctness, direction or pilot membership.
- E4.1 and E4.2 use the same population per explainer.

**Contrasts**
- Insertion Δ = AUC(explainer) − mean₅ AUC(random).
- Deletion Δ = mean₅ AUC(random) − AUC(explainer).
- E4.2: D = Δ_clinical − Δ_dermoscopy.

**Tests**
- Family: 12 tests = 8 E4.1 + 4 E4.2.
- Wilcoxon signed-rank, two-sided: zeros dropped (`wilcox`), normal approximation with tie-corrected variance, no continuity correction. **This is not an exact test.**
- An untestable input would be carried as p = 1, keeping the family at 12. There were none, and every input had n_nonzero = n.
- Holm adjustment.

**Estimates**
- The **sample median** and the **Hodges–Lehmann pseudomedian** (median of Walsh averages, i ≤ j) are reported separately.
- Each has a 95% pair-bootstrap CI: B = 2000, seeds 20261012–15, numpy linear quantiles, and pairs drawn once and shared across statistics.

**Assumption.** The signed-rank test assumes a distribution symmetric about its centre. HL estimates the pseudomedian. Where the sample median and HL differ (§3), the distribution is skewed, and the test speaks to the pseudomedian rather than to the median.

**Gate.** Positive Holm-supported dermoscopy evidence means adjusted p < 0.05 and HL > 0. A failing gate makes the E4.2 result descriptive, but the test is never removed from the family.

## 3. Primary results (`rq4_family_table.csv`)

| Test | n | Median [95% CI] | HL [95% CI] | p raw | p Holm | Interpretation |
|---|---|---|---|---|---|---|
| E4.1 Grad-CAM, insertion, dermoscopy | 275 | 0.147 [0.102, 0.203] | 0.166 [0.138, 0.192] | 2.9e-25 | 2.4e-24 | confirmatory |
| E4.1 Grad-CAM, insertion, clinical | 275 | 0.035 [0.023, 0.057] | 0.059 [0.042, 0.080] | 2.4e-14 | 1.4e-13 | confirmatory |
| E4.1 Grad-CAM, deletion, dermoscopy | 275 | 0.066 [0.030, 0.087] | 0.075 [0.052, 0.098] | 1.4e-10 | 7.2e-10 | confirmatory |
| E4.1 Grad-CAM, deletion, clinical | 275 | 0.021 [0.010, 0.032] | 0.032 [0.020, 0.046] | 6.4e-08 | 1.9e-07 | confirmatory |
| E4.2 Grad-CAM, insertion, D | 275 | −0.072 [−0.122, −0.030] | −0.086 [−0.113, −0.056] | 2.9e-08 | 1.2e-07 | confirmatory (gate passed) |
| E4.2 Grad-CAM, deletion, D | 275 | −0.019 [**−0.041, +0.002**] | −0.027 [−0.049, −0.003] | 0.029 | 0.029 | confirmatory (gate passed); weakest result |
| E4.1 IG, insertion, dermoscopy | 373 | 0.098 [0.083, 0.114] | 0.116 [0.104, 0.128] | 2.4e-58 | 2.4e-57 | confirmatory |
| E4.1 IG, insertion, clinical | 373 | 0.053 [0.046, 0.065] | 0.072 [0.062, 0.083] | 4.5e-56 | 4.1e-55 | confirmatory |
| E4.1 IG, deletion, dermoscopy | 373 | 0.163 [0.137, 0.189] | 0.190 [0.171, 0.211] | 8.6e-63 | 1.0e-61 | confirmatory |
| E4.1 IG, deletion, clinical | 373 | 0.060 [0.048, 0.070] | 0.079 [0.067, 0.094] | 2.8e-62 | 3.1e-61 | confirmatory |
| E4.2 IG, insertion, D | 373 | −0.029 [−0.039, −0.013] | −0.033 [−0.046, −0.021] | 2.2e-07 | 4.3e-07 | confirmatory (gate passed) |
| E4.2 IG, deletion, D | 373 | −0.073 [−0.093, −0.050] | −0.089 [−0.106, −0.072] | 5.4e-25 | 3.8e-24 | confirmatory (gate passed) |

**Interpretation, restricted to eligible pairs**
- Both explainers outperformed the random-map control in both modalities and on both curves.
- The advantage was consistently **smaller on clinical images** (all D < 0).
- **Grad-CAM deletion D** is the weakest result: Holm p = 0.029, and the **sample-median** CI includes 0, while the HL CI does not. Here the conclusion depends on the estimand.

## 4. As-computed sensitivity analysis (non-primary)

**Rule** (frozen):
- Valid maps use their primary curves.
- Finite constant maps (Grad-CAM all-zero, ranked in the original pixel-index order) and finite completeness-failed IG maps use their stored as-computed curves.
- Non-finite maps or failed curves remain unavailable; there were none.
- The completeness-failed IG maps are **numerically unvalidated**.

**Coverage:** 498/498 pairs for both explainers (249 melanoma + 249 non-melanoma).

**HL results** (as computed vs primary):

| Explainer, curve | Dermoscopy | Clinical | D |
|---|---|---|---|
| Grad-CAM insertion | 0.120 vs 0.166 | 0.016 vs 0.059 | −0.079 vs −0.086 |
| Grad-CAM deletion | 0.053 vs 0.075 | 0.013 vs 0.032 | −0.028 vs −0.027 |
| IG insertion | 0.110 vs 0.116 | 0.060 vs 0.072 | −0.038 vs −0.033 |
| IG deletion | 0.176 vs 0.190 | 0.067 vs 0.079 | −0.084 vs −0.089 |

**Reading:**
- The advantages over random are smaller, especially for Grad-CAM, but they have the same sign.
- The shift contrasts are similar.
- This analysis is descriptive. It does not promote the excluded maps to valid status, and it does not establish their numerical validity.

## 5. Eligibility and alternative controls

**Eligibility** (`eligibility_table.csv`), as in Stage 7A:
- Grad-CAM exclusions are all all-zero maps: 16 / 75 / 69 / 125.
- IG exclusions are all completeness failures: 6 / 28 / 42 / 62.
- Both are listed for dermoscopy melanoma / dermoscopy non-melanoma / clinical melanoma / clinical non-melanoma.

**IG completeness:** 138/996 = **13.9% failures**. The pilot's adequacy of 2.1% did **not** transfer to the full cohort, and IG is **not** described as globally converged.

**Alternative controls** (`other_controls_descriptive.csv`; descriptive; HL in AUC units):

| Explainer | Control | Pairs | Insertion, dermoscopy / clinical / D | Deletion, dermoscopy / clinical / D |
|---|---|---|---|---|
| Grad-CAM | centre | 275 | 0.184 / 0.069 / −0.096 | −0.008 / −0.020 / −0.015 |
| Grad-CAM | Sobel | 275 | 0.266 / 0.117 / −0.130 | 0.005 / 0.004 / 0.002 |
| Grad-CAM | mismatch | **146** | 0.107 / 0.056 / −0.048 | 0.071 / 0.033 / −0.025 |
| IG | centre | 373 | 0.108 / 0.078 / −0.010 | 0.109 / 0.031 / −0.068 |
| IG | Sobel | 373 | 0.182 / 0.107 / −0.053 | 0.124 / 0.051 / −0.060 |
| IG | mismatch | **280** | 0.119 / 0.069 / −0.035 | 0.149 / 0.071 / −0.056 |

**Notes**
- Grad-CAM deletion is close to or below the centre prior and the Sobel control, descriptively.
- The mismatch populations (146 and 280) are subsets: both the recipient and the planned donor must be valid in both modalities. They are not comparable populations to the primary analysis.

## 6. Claim limits

- The cohort is class-enriched.
- Exclusions select on model evidence, concentrated in clinical and non-melanoma images.
- IG has substantial numerical (completeness) failures.
- Only seed 0 is analysed. Stage 6B showed large seed variability, so these explanations are not generalised to other seeds.
- 24 pilot pairs had been inspected earlier.
- The metrics assess behaviour under the chosen blurred-baseline pixel-replacement procedure. They do not assess clinical correctness or causal explanation.

## 7. Files

**Code and tests**
- New: `src/rv1/rq4_stats.py`, `src/rv1/rq4_analysis.py`, `tests/test_rq4_stats.py`, `configs/rq4_stats_7b_v1.json`.
- Changed: `src/rv1/xai_prod.py` (file I/O idioms only) and `tests/test_xai_prod.py` (main block moved; file I/O).

**Run outputs:**
- `analysis_spec.json` with its sidecar;
- `results.json`;
- `rq4_family_table.csv`, `eligibility_table.csv`, `other_controls_descriptive.csv`;
- `per_image_inputs.csv`, `per_pair_inputs.csv` (primary and as-computed);
- `bootstrap_replicates.npz`;
- `thesis_results_rq4.md`, which has its own sidecar;
- the receipts.

**Integrity:** 1056 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged. No `__pycache__` was created.

## 8. Deferred RQ4 components (RQ4 is **not** complete)

- §6.5 **normalised scores** (δ = 0.05, with sensitivity δ ∈ {0.02, 0.10}) and E4.3 confidence adjustment.
- **Secondary IG variants:** |attr| curves and positive maps with σ = 2 smoothing.
- **E4.4** melanoma vs non-melanoma subgroup contrast; **E4.5** Grad-CAM–IG agreement.
- CIs and tests for the centre, Sobel and mismatch controls beyond the descriptive level.
- Seeds 1 and 2 on a 100-pair subset (optional per protocol).
- The simulated (HAM) arm and RQ2/RQ3.

**Stage 7B is complete.**
