# 06 — Decision Log (rev2)

This version supersedes `06_DECISION_LOG.md`, which is kept unchanged.

Status labels:

- **PRIOR-APPROVED**: approved earlier, according to `audit/DECISION_LOG.md`.
- **RECOMMENDED**: adopted unless the researcher or supervisor objects.
- **OPEN (scientific)**: needs an informed choice. A default is given.
- **UNRESOLVED (administrative)**: the answer depends on someone outside this work.

## Decisions

| ID | Decision | Status | Rationale | RQs | Consequence |
|---|---|---|---|---|---|
| D-00 | Train on HAM10000 only | PRIOR-APPROVED | — | all | — |
| D-01 | The 10 pp reference is an absolute value, used only as a descriptive marker | PRIOR-APPROVED + RECOMMENDED | No inferential test of whether the drop exceeds 10 pp | RQ1, RQ2 | — |
| D-02 | Malignant-vs-benign results are off-target | PRIOR-APPROVED | — | RQ5 | — |
| D-03 | Primary Derm7pt cohort 1002/249; 1011/252 as a sensitivity analysis | RECOMMENDED | Removes the same-file pairs | RQ1, RQ4 | — |
| D-04 | RQ1 primary family = E1.1 only, α = 0.05 with no adjustment; E1.3 secondary and unadjusted | RECOMMENDED | Removes the rev1 contradiction | RQ1 | — |
| D-05 | One fixed split; 3 architectures × 2 strategies × 3 seeds, sized by the pilot | RECOMMENDED | — | RQ3 | Reductions per 04_rev2 §R |
| D-06 | XAI target is the melanoma logit; enriched manifest of 249+249 pairs | RECOMMENDED | — | RQ4 | Always labelled "enriched" |
| D-07 | Raw trapezoidal scores v1 are primary; normalised scores are secondary with the δ rule; the v0 mean is kept for comparison | RECOMMENDED | Raw scores are always defined; normalisation has denominator problems | RQ4 | — |
| D-08 | Integrated Gradients as the second explainer | RECOMMENDED | — | RQ4 | — |
| D-09 | Cluster bootstrap; undefined replicates are reported | RECOMMENDED | — | all | — |
| D-10 | **CMRS is deferred** until its components, directions, denominators, populations and weights are fully defined; component tables are produced instead | RECOMMENDED | The rev1 formula was underspecified | CMRS | Does not block any RQ |
| D-11 | Drift→error: grouped cross-validation using only features available at deployment | RECOMMENDED | — | RQ4 | — |
| D-12 | No calibration step; ECE is descriptive | RECOMMENDED | — | RQ1 | — |
| D-16 | V1 preprocessing `prep_v1_cached256`, with fp32 inference | RECOMMENDED | Explicit and recorded | all | Separate from legacy reproduction |
| D-17 | Stable-hash perturbation seeds | RECOMMENDED | — | RQ2, RQ4 | — |
| D-18 | Masks are transformed jointly with geometric perturbations | RECOMMENDED | — | RQ2, RQ4 | — |
| D-19 | Stateless augmentation seeding per (seed, epoch, image) | RECOMMENDED | No hidden state to restore | RQ3 | Behaviour to be verified in T-B2 |
| D-23 | Retrospective framing; analyses called "primary retrospective" | RECOMMENDED | — | all | — |
| D-25 | Threshold = the highest t giving image-level validation sensitivity ≥ 0.95; ties counted as positive; not estimable if n₊ < 20 | RECOMMENDED | — | RQ1, RQ3, RQ5 | — |
| D-26 | Legacy reproduction (LR) and V1 analysis kept separate; ±0.005 is a diagnostic tolerance only | RECOMMENDED | — | RQ1 | — |
| D-27 | RQ4 primary uses raw control-referenced contrasts (insertion: explainer − control; deletion: control − explainer); random seeds averaged within each image; population = matched eligible pairs; Holm correction over 12 tests | RECOMMENDED | — | RQ4 | — |
| D-28 | Primary attribution = positive, melanoma-supporting evidence for both explainers; absolute IG attribution is secondary | RECOMMENDED | Makes Grad-CAM and IG comparable | RQ4 | IG positive maps may be all zero; the rate is reported |
| D-29 | RQ5: frozen patient partition (strata S1–S3, 50/35/15 split by stable hash); primary contrast = patient-cluster paired bootstrap AUROC difference | RECOMMENDED | Patients contribute repeated images | RQ5 | — |

## Open scientific decisions

None of these blocks the next task. Each has a default.

| ID | Question | Default |
|---|---|---|
| OS-1 | Normalisation threshold δ | 0.05, with a sensitivity analysis at {0.02, 0.10} |
| OS-2 | Size of the XAI manifest | 249+249 if the pilot allows it; otherwise 150+150 |
| OS-3 | Final RQ3 matrix size | Chosen by the pilot, following §R |
| OS-4 | Whether to specify CMRS at all in this thesis | Defer; report component tables |
| OS-5 | RQ5 adaptation recipe (learning rate, number of epochs) | 1e-5, 5 epochs, selection on adapt-val AUROC |
| OS-6 | Supervisor review of 02_rev2 | Desirable before the main runs; not assumed to have happened |

## Unresolved administrative items (not blockers for the next task)

| ID | Item | Status |
|---|---|---|
| A-1 | Institutional ethics requirement for public, de-identified data | **UNRESOLVED**, pending actual ITPU guidance. No rule is assumed. |
| A-2 | Status of the IOCTe abstract | Unknown. No external messaging is authorised. |
| A-3 | Streamlit installation | Implementation detail for task T-H; not a blocker. |
| A-4 | AI-assistance declaration | Draft in Appendix A below, for the author's review and edits. |

---

## Appendix A — Draft AI-assistance declaration (for author review; not final)

> *Draft.* During this project I used AI assistants: Claude-based coding agents and chat assistants, and possibly others that I should list. They helped with more than language editing. They wrote substantial parts of the source code, including the data loaders, the training and evaluation scripts, the Modality-Shift Simulator, the explanation and metric code, the analysis and campaign scripts, and the demonstration application. They also produced audit reports, protocol drafts and planning documents. Some earlier AI-generated statements were later found to be unsupported or incorrect, for example an invalid power estimate and an inappropriate comparison with a non-melanoma study. These were identified in later audits and corrected. I reviewed, directed and verified the work, and I take full responsibility for the methodology, results and conclusions. AI tools did not generate or alter data or results. Every reported number comes from executed code whose outputs are archived with provenance records. [Author: confirm which tools were used, the dates, whether Grammarly or Copilot were used, and edit to reflect your own contribution accurately.]

This draft replaces the Research Plan's statement that the simulator core and the CMRS logic were "hand-written". The project history does not support that statement.
