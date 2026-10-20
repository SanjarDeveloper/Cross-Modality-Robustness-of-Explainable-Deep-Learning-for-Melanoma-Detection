# Stage 7A Report — Production Seed-0 XAI on the Derm7pt XAI Cohort (Computation and Audit)

Date: 2026-10-03. All work in this stage used `research_v1/.venv_model`.

**Run:** `runs/xai_prod_7a/20261003T173258Z_3b655f88_548b3a05/`. The spec receipt and the audit receipt (`receipt_audit.json`) both verify.

**Stage records:** `runs/stage7a/`.

## Status

| Item | Result |
|---|---|
| Model | Stage 3A seed-0 ResNet-50 (`303d21e7…`), not the PAD-adapted model |
| Cohort | Frozen `xai_derm7pt_v1.csv`: **498 pairs (249 mel + 249 non-mel), 996 references**. All are in the primary cohort, with labels, modalities and image IDs verified |
| Contract frozen before computation | `spec.json` (hashed). It records the references in frozen order, the chunk membership, model, inputs, manifests, source hashes, precision policy and every setting. The full-cohort `mismatch_assignment.json` was hashed before any curve was computed |
| Computation | 21/21 chunks, **all 996 references**: Grad-CAM, IG m = 512, explainer curves, the 5 random controls, centre and Sobel. Mismatch curves (v2) |
| Endpoint failures | **0** |
| Non-finite maps | **0** |
| Random, centre and Sobel controls | Available for all 996 images |
| Tests | New `test_xai_prod.py` (8 tests, including 2 regression tests for the v1 bugs). Relevant XAI suites pass |
| Protected inputs | 0 changed, 0 removed |

**Not done:** final RQ4 tests (Wilcoxon, Hodges–Lehmann, Holm) and secondary statistics.

**Two implementation bugs were found and fixed** (§2). Neither affected the attribution or curve computation.

## 1. Production settings (from `spec.json`)

| Item | Setting |
|---|---|
| Preprocessing and precision | `prep_v1_cached256`; `xai_precision_v1` |
| Baseline | Blurred, k = 57, σ = 9.5 |
| Grad-CAM | `layer4`, batch 8 |
| **IG** | **m = 512**, Riemann midpoint rule, internal batch 16 |
| Curves | S = 32, `eval_batch` 11, endpoint tolerance 1e-4 |
| Controls | `controls_v1` |
| Outer batch | **48** references in frozen order: **20 chunks of 48 plus a final chunk of 36**. The baseline, the IG completeness logits and IG are computed per chunk. Resume never regroups |

**Scope of m = 512:** it was selected on the 24-pair Stage 3B.2 pilot, for this model and pipeline only. It is **not** universally converged, and every image was checked individually.

**Stored per reference**
- Grad-CAM map and logit.
- IG primary map, IG absolute map and **signed channel attribution**.
- Sobel map.
- logit(x) and logit(baseline), the baseline hash, and the completeness record (sum, Δlogit, e_abs, e_rel, failure).
- Validity flags and eligibility reasons.
- Every curve, with fractions, boundaries, AUCs, p_full, p_base and endpoint checks, for the explainers, the 5 random maps, centre, Sobel and mismatch (with the donor ID).
- Pilot membership.

**Secondary sources:** the IG absolute map and signed attribution are stored. **IG positive with σ = 2 smoothing is DEFERRED** and has not been computed.

## 2. Execution history and the bugs, fully disclosed

1. **Interruption.** The first compute process was stopped by the tool's background time limit after chunk 07.
   - Chunks 00–07 had been published atomically.
   - Chunk 08 was still in memory, so no partial file existed and none was quarantined.
   - A detached process resumed and verified chunks 00–07 by hash, then computed chunks 08–20 with the frozen membership.
2. **Bug 1 (donor eligibility).** `xai_prod.py` v1 passed the explainer record to `primary_ok()`, which reads `analysis_status`. Explainer records store `status`, so every donor was treated as ineligible and every **mismatch** control came out "unavailable".
3. **Bug 2 (audit).** The v1 audit's directory filter matched `mismatch_assignment.json` and crashed. Before crashing, it had published three files based on the v1 mismatch results, with no receipt.
4. **Handling.**
   - All v1 mismatch directories and the three v1 audit files were **moved unchanged** to `invalid_v1_donor_bug/`. `README_INVALID.json` lists their hashes. Nothing was deleted.
   - The code was fixed with `donor_eligible()` and `is_result_dir()`, and regression tests were added.
   - A hashed amendment, `amendment_mismatch_v2.json`, records the old and new `xai_prod.py` hashes.
   - The diff touches **no** compute-path code, so chunks 00–20 were reused after hash verification.
   - The mismatch phase was rerun as `mismatch2_*` and the audit as `*_v2`.
5. **What did not change.** The frozen donor mapping is unchanged. Donor eligibility comes only from the computed eligibility records, not from any score.

## 3. Eligibility audit (`coverage_audit_v2.json`, `image_status_v2.csv`)

### Grad-CAM

| | n | Valid (primary) | Invalid reason |
|---|---|---|---|
| Dermoscopy, melanoma | 249 | 233 | all-zero (constant): 16 |
| Dermoscopy, non-melanoma | 249 | 174 | all-zero: 75 |
| Clinical, melanoma | 249 | 180 | all-zero: 69 |
| Clinical, non-melanoma | 249 | 124 | all-zero: 125 |

- An all-zero Grad-CAM map means the map has **no positive evidence for the melanoma target**. It is not a correct "non-melanoma" explanation, and it is not a diagnosis.

### IG m = 512

| | n | Valid (primary) | Invalid reason |
|---|---|---|---|
| Dermoscopy, melanoma | 249 | 243 | completeness failure: 6 |
| Dermoscopy, non-melanoma | 249 | 221 | completeness failure: 28 |
| Clinical, melanoma | 249 | 207 | completeness failure: 42 |
| Clinical, non-melanoma | 249 | 187 | completeness failure: 62 |

- **Overall IG completeness failure rate: 138/996 = 13.9%.** This is far above the 2.1% seen in the 24-pair pilot and above the pilot's 5% criterion.
- m was **not** changed and nothing was retried per image. This must be reported, and is flagged for the statistics stage (§5).

### Matched eligible pairs (valid in both modalities) and control availability

| Explainer | Eligible pairs | Mel / non-mel | Random | Centre | Sobel | Mismatch (separate denominator) | Of which pilot pairs |
|---|---|---|---|---|---|---|---|
| Grad-CAM | **275 / 498** | 175 / 100 | 275 | 275 | 275 | **146** (114 / 32) | 15 |
| IG m = 512 | **373 / 498** | 204 / 169 | 373 | 373 | 373 | **280** (164 / 116) | 23 |

Mismatch unavailability is caused by the planned donor map being ineligible (Grad-CAM 200 image-explainer rows; IG 109). No donors were replaced.

**As-computed curves** for finite invalid maps, for the protocol sensitivity analysis: Grad-CAM 285 and IG 138. These are labelled `as_computed_non_primary`. There were no NaN or Inf maps, so no map was excluded from them.

### Selection effects

- Eligibility is **selection by model evidence**. Exclusions are concentrated in non-melanoma and clinical images.
- The results will describe the **eligible pairs of the seed-0 model**.
- The cohort is class-enriched and does not reflect natural prevalence.
- Stage 6B showed large seed-to-seed differences, so seed-0 explanations are not generalised to other seeds.
- 24 of the 498 pairs were used in the Stage 3B pilots. They are flagged in every record.
- No filtering by correctness or probability.

## 4. Measured cost

| Phase | Time | Detail |
|---|---|---|
| Compute | 4353 s (72.6 min) | IG 2724 s; curves 1577 s; Grad-CAM 5.4 s; baseline 0.4 s. About 208–212 s per 48-reference chunk; 156 s for the final 36 |
| Mismatch v2 | 288 s | |
| Peak CUDA memory | 1651 MiB allocated / 2412 MiB reserved | |
| Disk | 1.11 GB | |

The Stage 3B projection was 1.37 h. The measured time was 1.29 h, excluding the interruption.

## 5. Completed and deferred RQ4 components (open issues for the statistics stage)

**Completed**
- Maps, curves and AUCs for all 996 references.
- Controls, including the mismatch controls.
- Per-image Δ inputs (`scores_v2.csv`: Δ_ins and Δ_del per explainer and control).
- Eligibility, coverage and as-computed curves.
- p_full and p_base, for the secondary normalisation.

**Deferred to the statistics stage**
- E4.1 and E4.2: Wilcoxon, Hodges–Lehmann, case-bootstrap CIs and Holm across the 12 tests.
- The "as computed" sensitivity analysis.
- §6.5 normalised scores.
- E4.4 and E4.5.
- IG |attr| curves and the σ = 2 smoothed IG secondary analysis (curves not yet computed).

**Open issues to decide before the statistics stage**
1. **IG completeness failures (13.9%)** exceed the pilot's 5% criterion. The protocol keeps m fixed. The statistics stage must report the failure rates and treat IG primary results as conditional on completeness-passing pairs. Any change to m would need a new, explicitly authorised stage.
2. **"As computed" scope.** For IG, the protocol's "as computed" sensitivity analysis includes maps that failed completeness, whereas Addendum B3 blocks completeness-failed maps from the primary boundary. The as-computed curves exist and are labelled non-primary. **How they are used in the sensitivity analysis needs an explicit rule.**
3. **Mismatch denominators** are smaller (146 and 280 pairs) and must be reported separately.

## 6. Test-count correction

`docs/STAGE6B_TESTCOUNT_NOTE_v1.md`: `test_seed_spread.py` discovers **4** tests, not 5. No fifth seed-spread test exists elsewhere.

## 7. Files

**New code and tests:** `src/rv1/xai_prod.py` (v2; the v1 copy is at `runs/stage7a/xai_prod_v1_with_donor_bug.py.txt`), `tests/test_xai_prod.py`.

**New config and documents:** `configs/xai_prod_7a_v1.json`, `docs/STAGE6B_TESTCOUNT_NOTE_v1.md`.

**Run directory:**
- `spec.json`, `mismatch_assignment.json` and `amendment_mismatch_v2.json`, each with a sidecar;
- `chunk_00..20/`, each with `maps.npz`, `records.json` and `chunk_manifest.json`;
- `mismatch2_00..20/`;
- `coverage_audit_v2.json`, `image_status_v2.csv`, `scores_v2.csv`;
- `invalid_v1_donor_bug/`;
- `progress.jsonl`;
- the receipts.

**`runs/stage7a/`:** stdout logs, batch scripts and integrity files.

**Integrity:** 874 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged. No `__pycache__` was created.

**Stage 7A is complete (computation and audit only).**
