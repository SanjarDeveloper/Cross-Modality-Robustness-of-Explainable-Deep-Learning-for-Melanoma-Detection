# Stage 11A Clarification Note v2 — 2026-10-10 (after a structural stop; before any result was produced)

This note adds to v1 and does not replace it. The following are unchanged:
- `e46_spec.json`, the cohorts, the folds and the feature dictionary;
- the primary-cohort models (M0, M0_linear, M1, M2) and the full-cohort models.

## What happened

The first Stage 11B cross-validation attempt **stopped**, as clarification note v1 §4 requires. The stop occurred in the IG-valid sensitivity refit, with the message `zero-variance training column ['ig_completeness_failure']`.

- No cross-validation output was published and no metric was printed or inspected.
- The log is kept: `runs/stage11b/phase_cv_attempt1_failed.txt`.

## Why it happened

The IG-valid subset is defined as `ig_completeness_failure == 0`. Inside it, that indicator is **identically 0 by construction**. Its standardisation is therefore undefined, whatever the data. This is a structural property of the frozen design, not a data-dependent degeneracy.

## Resolution (structural; not result-driven)

**IG-valid refit only:** the model `M1_igvalid` = M1 **without** `ig_completeness_failure`. That is M0 plus `ig_delta_ins`, `ig_delta_del`, `ig_conc_top10`, `gradcam_zero` and `gradcam_conc_top10`. A predictor that is constant by subset definition carries no information.

**Unchanged:** every other setting, the restricted parent folds, the refitting within the subset, and the primary analysis.

**If anything else is degenerate,** the run still stops (v1 §4).
