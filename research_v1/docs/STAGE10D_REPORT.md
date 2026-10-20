# Stage 10D Report — Descriptive Statistical Analysis of the Simulated-HAM XAI Arm

Date: 2026-10-10. Environment: `research_v1/.venv_model`. CPU only.

**Run:** `runs/xai_sim_stats_10d/20261010T052654Z_8e5b69aa_c3087880/`.

**Stage records:** `runs/stage10d/`.

**Not done:** GPU computation, training and E4.6.

## Status

| Item | Result |
|---|---|
| Analysis spec (`analysis_spec.json`, hashed before any summary) | **Existing estimands are kept:** Δ_ins = explainer − random and Δ_del = random − explainer (`controls_v1.control_scores`, the 5 random AUCs averaged within each image first); §6.4 eligibility; area-matched IoU; pointing. The **Stage 10B plan, "descriptive"**, is followed |
| Retrospective additions (labelled in the spec) | Bootstrap seed 20261200; the paired change-from-severity-0 summaries; the IG − Grad-CAM intersection comparison; the localisation summaries (median IoU, pointing rate, explainer − control); the as-computed sensitivity summaries; mismatch summaries as point estimates only |
| Statistical scope | **No hypothesis tests.** No testing family was prespecified for the simulated arm. The Derm7pt 12-test Holm family is untouched. Intervals are pointwise |
| Input validation | Stage 10C spec and audit outputs verified by hash; all 88 chunk and 88 mismatch dirs verified, with no `.tmp` or superseded artefacts. **Audit reproduced:** 3,740 references, 31 IG completeness failures, 1,163 Grad-CAM exclusions, and every jointly eligible population count equal to the 10C table |
| Bootstrap | Lesion-cluster, B = 2,000, **one shared draw per replicate** across all statistics. All images and all 11 conditions of a lesion move together, with multiplicity kept. **0 undefined replicates** |
| Tests | New `tests/test_xai_sim_stats.py`, 5/5: lesion weights, no zero substitution, undefined counting, population counts, contrast directions, paired subset. The full suite was not rerun, because no shared module changed |
| Exit / runtime | analyze 0, 160 s. Receipts verify |
| Integrity | 4712 → 4713 files, **0 changed and 0 removed**. Added: the config |

## 1. Outputs (run dir)

**Tables**
- `faithfulness_by_condition.csv`: random control with CIs; centre, Sobel and mismatch as point estimates.
- `change_from_s0.csv`: both levels and the paired change on the identical subset.
- `ig_vs_gradcam.csv`: intersection populations.
- `localisation_matched.csv`: matched populations; IoU, pointing, explainer − control differences, change from s = 0, mismatch point estimates and median mask fraction.
- `as_computed_sensitivity.csv`.
- `eligibility_by_condition_class.csv`.
- `populations.csv`.

**Every row reports** images, lesions and mel / non-mel counts.

**Plots:** `plot_faithfulness_{gradcam,ig2048}.png` and `plot_localisation.png`, each labelled with its n per condition. Lines in the localisation plot only guide the eye; the conditions are categorical.

**Thesis text:** `runs/stage10d/thesis_results_rq4_simulated_arm.md` (Tables S1–S3, results, interpretation and limitations). Its narrative numbers were checked against the CSVs; one statement about Grad-CAM under scale was corrected before finalising.

## 2. Main findings (descriptive; 95% lesion-bootstrap CIs)

**IG2048 faithfulness**
- **Severity 0:** Δ_ins 0.121 [0.105, 0.138] and Δ_del 0.215 [0.185, 0.239], on n = 335 images in 251 lesions (170 mel / 165 non-mel).
- **Under shift:**
  - Blur reduced faithfulness most. Paired change at blur 1.0: Δ_ins −0.120 [−0.138, −0.105] and Δ_del −0.178 [−0.204, −0.149], on 335 images. The blur 1.0 level is indistinguishable from random for insertion.
  - Scale 1.0: −0.061 / −0.130.
  - Illumination 1.0: −0.041 / −0.077.
  - JPEG and nuisance: small changes, at most −0.028.

**Grad-CAM**
- **Severity 0:** Δ_ins 0.138 [0.097, 0.172] and Δ_del 0.133 [0.103, 0.166], on n = 286.
- **Blur:** the insertion advantage became negative (blur 0.5: −0.121), and the eligible population shrank to 131 at blur 1.0 (98 mel / 33 non-mel).
- **Scale:** the higher levels coincide with population selection (176 eligible at scale 1.0).

**IG vs Grad-CAM (intersection):** at severity 0, IG was higher for deletion, +0.109 [0.092, 0.127] on n = 283. For insertion the difference was compatible with zero.

**Localisation (matched populations)**
- **IG:** IoU 0.26–0.29 and pointing 0.82–0.91 in every condition. It is above random (+0.07 to +0.23 IoU) but well below the centre prior (−0.44 at s = 0), except under scale 1.0 (+0.18 over the centre prior).
- **Grad-CAM:** IoU 0.49 at s = 0, collapsing to 0.11 under blur 1.0.
- **Sobel:** high IoU under scale (0.62), consistent with edges at the inset boundary and the lesion.

**Selection and as-computed sensitivity**
- IG summaries are essentially unchanged with invalid maps included.
- Grad-CAM summaries change materially (e.g. scale 1.0 Δ_ins 0.209 on 176 eligible images vs 0.092 on all 340), so Grad-CAM shift results describe selected subsets.

**Measurement caveats (also stated in the thesis text)**
- Area-matched IoU uses the true mask area to choose k; it is not independent segmentation performance.
- Under scale, the mask fraction falls from about 0.27 to 0.05, and the lesion is displaced with a synthetic surround and seam.
- Masks represent lesion extent, not the visible lesion under hair and glare.
- Mismatch donors are fixed. Recipient-lesion resampling does not capture donor dependence, so mismatch values are point estimates only.

**What the metrics measure:** perturbation-based faithfulness and mask localisation, not clinical correctness.

## 3. Files

**Code, test and config:** `src/rv1/xai_sim_stats.py`, `tests/test_xai_sim_stats.py`, `configs/xai_sim_stats_10d_v1.json`.

**Run dir:** `analysis_spec.json`, `results.json`, the 7 CSV tables, the 3 plots and the receipts.

**`runs/stage10d/`:** `phase_analyze.txt`, `status.txt`, the thesis text and the integrity files.

## 4. Remaining for RQ4

1. **E4.6.**
2. The deferred control comparisons for the secondary IG variants (Stage 10A).
3. Optional XAI for additional seeds.

**RQ4 is not complete. Stage 10D stops here.**
