# Stage 11B.1 Report — E4.6 Reproducibility and Reporting Closure; RQ4 Completion Inventory

Date: 2026-10-10. Environment: `research_v1/.venv_model`. CPU only.

**Stage records:** `runs/stage11b1/`.

**Not done:** model tuning, new features, new scientific comparisons, GPU work, and any change to conclusions.

## 1. Interpretation note (`docs/STAGE11B_INTERPRETATION_NOTE_v1.md`)

1. **Main result.** Use "no improvement in out-of-fold error discrimination was demonstrated relative to the specified spline probability baseline". M1 − M0 is −0.024 [−0.060, 0.012]. The wording must not suggest that the features add no information in general.
2. **Calibration.** The lower calibration slopes of M1 and M2 (0.79 / 0.76 vs 0.96) are **compatible with** overfitting; they are not proof of its cause.
3. **Solver test.** It verifies agreement **on the implemented objective** only. It does not show equivalence to scikit-learn's full implementation.

## 2. Feature provenance

**Regeneration**
- The feature table was rebuilt with the current source into a new directory: `runs/e46_cv_11b1_verify/20261010T060552Z_e04857d3_2ddfb00a/`. Its receipt verifies.
- Comparison with the frozen Stage 11B table (`runs/stage11b1/verification_11b1.json`):
  - 498 = 498 rows, identical identifiers in identical order, identical columns;
  - **byte-identical files** (same SHA-256);
  - **0 value mismatches**;
  - missingness and provenance are equal, because the meta files are equal apart from the builder hash.
- **No substantive difference.**

**Old receipt (not rewritten)**
- The original features receipt still reports `source: hash mismatch …/e46_cv.py`, because the module was edited after the features phase (the IG-valid subset change).
- The new verification receipt `runs/stage11b1/receipt_verification_11b1.json` links:
  - the old and new feature runs;
  - the builder SHA-256 at the original features phase and now;
  - the output hashes.

## 3. Full-cohort out-of-fold predictions

**How they were obtained:** reconstructed from the saved fold-model parameters in `fold_models.json` (knots, means, standard deviations, coefficients, intercepts), using the frozen Stage 11A fold algorithm. No refitting.

**Saved to** `runs/stage11b1/oof_full_clinical_reconstructed.csv`. For all 1,002 cases it contains:
- `case_id` and `record_uid`;
- label and error;
- fold;
- `oof_M0` and `oof_M0_linear`.

**Reproduction against `cv_results.json`**
- Declared tolerance for AUROC, Brier and log-loss: 1e-12.
- **Maximum absolute difference: 0.0.** Calibration intercept and slope also differ by 0.0.
- The original `cv_results.json` and `fold_models.json` are unchanged, and their hashes are recorded.

## 4. Verification and integrity

**Tests pass:** `test_e46_cv` 9/9, `test_e46_design` 6/6, and `test_xai_secondary`.

**Integrity**
- 4782 → 4783 files: **0 changed and 0 removed**. Added: the interpretation note.
- The new verification run and the `stage11b1` files are outside that scope, and all earlier E4.6 artefacts are untouched.

## 5. RQ4 completion inventory

### Completed (with authoritative artefacts)

| Component | Protocol reference | Stage |
|---|---|---|
| E4.1 / E4.2 primary family (12 tests, Holm), Derm7pt | 02_rev2 §6.3 | 7B |
| As-computed sensitivity analysis, normalised scores (§6.5), E4.3 | §6.4–6.5 | 7B / 7C |
| E4.4, E4.5 (exploratory) | §6.7 | 7C / 7D |
| Secondary IG variants (\|attr\| and σ = 2 smoothed) vs the **random** control | §6.1, "Secondary attribution" row | 10A |
| Simulated HAM arm: computation, audit and descriptive statistics, including localisation | §6.7, "Simulated arm" | 10B / 10B.1 / 10C / 10D |
| E4.6 (clinical-only error prediction) | §6.7 E4.6; §9 row 8 | 11A / 11B / 11B.1 |

### Remaining required work: secondary IG variants against controls (b)–(d)

**Protocol basis**
- 02_rev2 §6.1 defines four controls: (a) random, (b) centre Gaussian prior, (c) Sobel edge magnitude, (d) mismatched map.
- The "Secondary attribution" row requires the IG |attr| and σ = 2 smoothed variants to be "reported separately".
- Stage 10A computed only the random-referenced contrasts and listed centre, Sobel and mismatch as deferred (`STAGE10A_REPORT.md`, status table and §5).

**What each comparison needs**

| Comparison | Needed artefacts | Status | New computation |
|---|---|---|---|
| **Centre control** for `ig_abs` and `ig_pos_smooth` | Per-image centre AUCs; variant AUCs | Both exist: centre AUCs in the 7A `records.json`, under `controls.centre.auc_ins/auc_del`; variant AUCs in the 10A `var_XX/records.json` | **None.** CPU analysis only, after an identity check that x, baseline, model and settings match (as done for random in 10A) |
| **Sobel control** | Same as centre (`controls.sobel`) | Both exist | **None.** CPU only |
| **Mismatch control** for the variants | (i) the frozen 7A donor assignment (`mismatch_assignment.json`); (ii) **donor variant maps**: `ig_abs` from 7A `maps.npz` `ig512_absolute`; `ig_pos_smooth` from 10A `maps_smooth.npz` (a float32 copy; the float64 hash is recorded); (iii) **new insertion/deletion curves** of each donor variant map on the recipient image | (i) and (ii) exist; (iii) does not | **Yes: GPU curves.** About 2 × 858 eligible references × 1 curve set, roughly 0.18 s each (7A rate), so about 5 min. Donor eligibility must use the **variant-specific** validity. Primary mismatch curves must **not** be relabelled |

**Rules for these comparisons**
- They are descriptive and must not be added to the 12-test Holm family.
- The populations must be the variant-specific matched pairs (373 in 10A), with control availability reported.

### Optional extensions (not required)

- **Additional-seed XAI.** Protocol §6.1 allows seeds 1–2 on a 100-pair subset "only if the budget allows".
- **DDI or XAI for non-primary RQ3 cells.** Not required by E1.1–E1.5 or RQ4.

**RQ4 status:** complete except for the secondary-variant centre / Sobel / mismatch comparisons above. Centre and Sobel need CPU only; mismatch needs a short GPU curve run.

**Stage 11B.1 is complete.**
