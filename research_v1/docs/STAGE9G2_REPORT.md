# Stage 9G2 Report — Heavy-Model RQ2 Perturbation Sweeps and Matched Standard-vs-Heavy Robustness Contrasts

Date: 2026-10-04. Environment: `research_v1/.venv_model`.

**Run:** `runs/rq3h_rq2_9g2/20261004T163507Z_7d7bac38_438cbc0c/` (269 MB).

**Stage records:** `runs/stage9g2/`.

**Scope:** HAM TEST, 5 factors × 9 severities, for the 9 frozen heavy models. All 9 standard sweeps (Stage 9C2, incl. Stage 8B) were reused. No threshold fitting, no simulator change, no DDI, no adaptation, no XAI.

## Status

| Item | Result |
|---|---|
| Code | New `src/rv1/rq3h_rq2.py`. Its spec and compare phases are new. The sweep and analyze phases are the **unchanged, tested Stage 9C2 functions** (`rq3_rq2.phase_sweep` / `phase_analyze`), driven by a 9C2-format spec. No shared module changed |
| Tests | New `tests/test_rq3h_rq2.py`, **6/6 pass**. Covered: binding and matching, identical perturbation, 18-cell arithmetic, per-replicate worst loss and benefit sign, equal-model zero and undefined handling, censored crossing ordinal. Also pass: `test_rq3_rq2`, `test_rq2_stats`, `test_rq2_predict`, `test_perturb`, `test_rq3_heavy_eval`. The full suite was not rerun, because no shared module changed |
| Spec | `spec.json` (hashed), checked against: model checkpoints (hash, epoch, `aug_version = aug_heavy_v1`); the frozen Stage 9G1 thresholds (bound to the checkpoint SHA); each model's own Stage 9G1 HAM TEST reference; simulator `perturb_msim_v1` (`855cf57b…`), grid and pipeline as in 9C2; source hashes; Stage 9G1 receipts (verify); Stage 9C2 outputs (hashes unchanged). Contrasts frozen before analysis |
| Phase exits | sweep 0 (21:35–22:02, **27.1 min**: about 10 s transform plus about 30 s inference per unit for 9 models), analyze 0 (83 s), compare 0 (159 s). Receipts verify |
| Coverage | **405/405 heavy unit dirs** = 9 × 69,660 = **626,940 new rows**. Together with the reused standard sweeps: **18 × 69,660 = 1,253,880 logical rows**, verified in the compare phase (18 cells × 45 units × 1548). Record identity, labels and lesion IDs are identical across all 18 cells |
| Identical perturbations | Audit digests (first 4 records per unit) equal Stage 8B in the sweep phase, and **equal the Stage 9C2 standard unit manifests in 405/405 units** |
| Baselines | Severity 0 against each heavy model's own Stage 9G1 HAM TEST predictions: 45/45 checks, **max \|Δp\| = 0.0** (tolerance 1e-6, frozen); predictions and labels identical |
| Statistics | Stage 8C definitions unchanged (`rq2_stats`), B = 2000 lesion-cluster bootstrap, seed 20261050: the same draws as 8C and 9C2, shared across all 18 models, factors and levels. **0 undefined replicates.** A single compare loop over all 18 cells **reproduces the Stage 9C2 standard worst-case points and CIs (max \|Δ\| = 0.0)** and this run's heavy analyze output (0.0) |
| Integrity | 3110 → 3112 files, **0 changed and 0 removed**. Added: the config and `STAGE9G1_CLARIFICATION_NOTE_v1.md`. Cache unchanged |

**Analyze phase note:** its built-in "ResNet-50 s0 vs Stage 8C" reproduction block does not apply here, because there is no reused model in this run. Its 0.0 is vacuous. The substantive reproduction checks are the ones in the compare phase above.

## 1. Per-model RQ2 statistics (heavy models)

**Files:** `per_level_table.csv`, `worst_case_table.csv`, `crossing_table.csv`, `ranking_table.csv`, `seed_ranges.csv`, and 10 per-factor plots.

**Content:**
- per-level sensitivity, specificity and AUROC;
- changes from each model's own baseline;
- worst-case loss (including s = 0, maximum recomputed per replicate);
- the 10-pp crossing with censoring;
- within-model factor ranking shares with split ties.

## 2. Matched contrasts

**Files:** `matched_worst_case.csv`, `matched_per_level.csv`, `matched_crossing.csv`, `matched_seed_ranges.csv`, `plot_matched_worst_{sens,auc}.png`.

**For each matched pair (architecture, seed) and factor, the files give:**
- both thresholds;
- both baselines and their paired difference;
- both worst losses with their worst levels and intervals;
- **benefit = worst_std − worst_heavy** (per-replicate CI, share of replicates > 0);
- both grid minima (absolute) and their difference (per-replicate CI);
- absolute sensitivity, specificity and AUROC at every severity for both strategies.

**Crossings** are shown side by side for both strategies (observed markers, censoring shares, censored quantiles), plus a per-replicate **ordinal** comparison: heavy later, later because heavy is censored, equal, both censored, earlier, or undefined. Censored markers are never subtracted.

**Summary (from `matched_worst_case.csv`, read programmatically)**

| Factor | Relation to recipe | Sensitivity benefit > 0 | Min sensitivity higher | AUROC benefit > 0 | Min AUROC higher |
|---|---|---|---|---|---|
| Blur | overlaps | **9/9** (9 CIs > 0; +21 to +59 pp) | **9/9** | **9/9** (7 CIs > 0) | 8/9 |
| Illumination | overlaps | 7/9 (4 CIs > 0, 1 < 0) | 6/9 | **8/9** (7 CIs > 0) | 8/9 |
| Compression | overlaps | 6/9 (1 tie) | 5/9 | **9/9** (6 CIs > 0) | 6/9 |
| Scale | partial | 3/9 (1 tie; 3 CIs < 0) | 3/9 | 6/9 (4 CIs > 0, 1 < 0) | 5/9 |
| Nuisance | **not in recipe** | **2/9** (1 exact tie; 6 negative, 2 CIs < 0) | 3/9 | 5/9 (all \|·\| ≤ 0.014) | **1/9** |

**Tie note:** for ResNet-50 s0 nuisance, both worst losses are 12/170. The CSV shows a benefit of +1.1e-14 because of floating-point rounding; it is counted as an exact tie.

**Crossing (observed, 45 pairs)**
- **Blur:** heavy crossing later, or censored, in 9/9 pairs; per-replicate share "heavy later" = 1.00 in all nine.
- **Compression:** both models censored in 7/9 pairs; heavy censored where standard crossed in EfficientNet-B3 s1 and s2.
- **Nuisance:** heavy crossed earlier in 6/9 pairs (ResNet-50 s1 and s2, EfficientNet-B3 s0 and s2, DenseNet-121 s1 and s2).
- Full data in `matched_crossing.csv`.

**Baselines:** heavy models had lower unperturbed HAM TEST AUROC in 7/9 pairs, so absolute values are shown beside every loss.

**Thesis-ready text:** `runs/stage9g2/thesis_results_rq3_heavy_rq2.md` (with `.sha256`). It was corrected against the CSV before hashing.

## 3. Interpretation (exploratory)

**Where heavy augmentation helped.** Robustness gains were clearest and most consistent across seeds for the families that overlap the training augmentation:
- blur, for both metrics and in absolute terms;
- illumination, for AUROC;
- JPEG, for AUROC, with large sensitivity gains only in the compression-susceptible EfficientNet-B3 seeds.

**Where it did not.**
- **Scale** (only partly overlapping) was heterogeneous.
- **Nuisance** (glare and hair, absent from the recipe) mostly showed **larger** sensitivity losses and lower absolute AUROC under heavy augmentation.

The results support family-specific, not general, robustness. They do not establish robustness to unseen real-world shifts. They concern one fixed recipe, an 8-epoch budget, these severity ranges, and pretrained variants at 224 px.

**No strategy or architecture is declared a winner.** No between-architecture tests or combined p-values were computed. The primary model (standard ResNet-50 s0) is unchanged.

## 4. Stage 9G1 clarification

`docs/STAGE9G1_CLARIFICATION_NOTE_v1.md` (new; Stage 9G1 files unchanged) covers three points:
1. **The sensitivity-gap narrowing count is 8/9, not 7/9**, verified from the 9G1 `matched_table.csv`. The report's consistency table was right; its prose and the thesis text were wrong.
2. The narrowing is a real measurement, but it went together with specificity losses.
3. The ECE statements are specific to Derm7pt PRIMARY and `ece_pos_v1` (15 equal-width bins, positive class).

## 5. RQ3 requirement inventory (`runs/stage9g2/rq3_inventory_18cell_v3.json`)

| Requirement | Standard × 9 | Heavy × 9 |
|---|---|---|
| Training and verification | done (6A/9B) | done (9F) |
| Unperturbed HAM TEST / Derm7pt / PAD evaluation | done (4A/4C/6B/9C1) | done (9G1) |
| RQ2 HAM TEST sweep (5 × 9) | done (8B/9C2) | done (9G2) |
| Matched standard-vs-heavy contrasts | — | done (9G1 unperturbed, 9G2 perturbed) |

**Every required RQ3 matrix evaluation is accounted for, so RQ3 is complete.**

**Optional extensions, not done:**
- DDI for the non-primary cells. DDI was not required for E1.1–E1.5, and it was done for the primary model in Stage 4C.
- XAI for the non-primary cells.

**Deferred RQ4 work remains separate.**

## 6. Files

**Code, test and config:** `src/rv1/rq3h_rq2.py`, `tests/test_rq3h_rq2.py`, `configs/rq3h_rq2_9g2_v1.json`.

**Document:** `docs/STAGE9G1_CLARIFICATION_NOTE_v1.md`.

**Run dir:**
- `spec.json`;
- 405 `u_<heavy model>__<factor>__<sev>/` dirs, each with predictions and a hashed manifest holding the audit digests and baseline difference;
- `progress.jsonl`;
- the per-model tables and plots (§1);
- the matched tables and plots (§2);
- `contrast_results.json` (coverage and reproduction);
- receipts for the spec, analyze and compare phases.

**`runs/stage9g2/`:**
- `run_phases.bat`, `status.txt` and the phase logs;
- the thesis text and its hash;
- the inventory;
- the integrity files.

**Stage 9G2 is complete.**
