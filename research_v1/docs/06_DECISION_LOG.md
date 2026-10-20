# 06 — Decision Log (research_v1)

Status labels:

- **PRIOR-APPROVED** — approved in an earlier session, per `audit/DECISION_LOG.md`.
- **RECOMMENDED** — this package recommends it. It is adopted unless the researcher objects.
- **BLOCKING** — needs the researcher's answer before the affected task starts.

| ID | Decision | Status | Rationale | RQs | Consequence |
|---|---|---|---|---|---|
| D-00 | Train on HAM10000 only | PRIOR-APPROVED | Keeps Derm7pt/PAD as evaluation data | all | — |
| D-01 | The 10 % marker = 10 absolute pp | PRIOR-APPROVED (delegated) | — | RQ1, RQ2 | Used only as a descriptive marker (D-20) |
| D-02 | Malignant-vs-benign is off-target | PRIOR-APPROVED | The model is a melanoma detector | RQ5 | Off-target tables go to the appendix, if shown at all |
| D-03 | Derm7pt primary cohort 1002/249; 1011/252 as sensitivity analysis | RECOMMENDED | The 9 same-file pairs carry no modality contrast | RQ1, RQ4 | Legacy numbers are reported as the sensitivity analysis |
| D-04 | E1.1 melanoma-only exact McNemar is primary; DeLong secondary; all-case McNemar descriptive | RECOMMENDED | Matches the estimand | RQ1 | Inference must be re-run (T-07) |
| D-05 | Fixed split_seed 0; training seeds {0,1,2}; 3 arch × 2 strategy matrix, confirmed by the pilot | RECOMMENDED | Seed spread then measures training variability | RQ3 | Up to 18 models; reductions in §R of `04_…` |
| D-06 | XAI target = melanoma logit; enriched manifest 249+249, shared by all explainers and controls | RECOMMENDED | Fixed, label-independent target | RQ4 | Must always be labelled "enriched" |
| D-07 | Faithfulness metrics v1 = trapezoid + normalised; v0 kept for comparison; curves persisted | RECOMMENDED | Correct integration; handles confidence dependence | RQ4 | — |
| D-08 | Second explainer = Integrated Gradients with the blurred baseline | RECOMMENDED | Methodologically distinct from Grad-CAM; cheaper than Score-CAM | RQ4 | Completeness check required |
| D-09 | Cluster bootstrap by lesion/case/patient; Holm within the primary family of each RQ | RECOMMENDED | Honest uncertainty | all | — |
| D-10 | CMRS v1: equal weights, no zero imputation, axis signature, raw components shown, no colour bands | RECOMMENDED | Avoids circular weights and false equivalence | CMRS | Exploratory only |
| D-11 | Drift→error: grouped CV, deployment-available features, probability baseline | RECOMMENDED | The legacy model is in-sample and uses twin-image features | RQ4 | A null result is acceptable |
| D-12 | No post-hoc calibration; ECE descriptive; "calibrated" removed from the text | RECOMMENDED | Calibrating on HAM val and evaluating a shift adds an extra estimand, and time is limited | RQ1, CMRS | Calibration is listed as future work |
| D-16 | Preprocessing `prep_v1_cached256` recorded in every row | RECOMMENDED | Legacy comparability; explicit provenance | all | Raw pathway used only in a 50-image consistency test |
| D-17 | Perturbation seeding via SHA-256 stable_seed; runner sets PYTHONHASHSEED=0 | RECOMMENDED | Fixes F-05 | RQ2, RQ4 | Legacy sweep values are not exactly reproducible; v1 supersedes them |
| D-18 | `scale_v1` transforms masks; any factor without mask support gives localisation = NaN | RECOMMENDED | Fixes F-07 | RQ2, RQ4 | — |
| D-19 | Albumentations seeded through `Compose(seed=…)` plus worker reseeding | RECOMMENDED | Fixes F-09 | RQ3 | Bitwise GPU determinism is not claimed |
| D-20 | ">10 pp" is not tested inferentially on Derm7pt | RECOMMENDED | Conditional power is about 0.5; data already seen | RQ1 | The magnitude is reported as an estimate with CI |
| D-21 | RQ5 adaptation on the primary model only; PAD patient partition roughly 50/50 of melanoma patients | RECOMMENDED | Separates adaptation from evaluation | RQ5 | All results descriptive |
| D-22 | Own implementations instead of Quantus/MLflow/pytorch-grad-cam; no Docker | RECOMMENDED | Minimal dependencies, tested code | all | The deviation is disclosed in Methods |
| D-23 | The protocol is described as a retrospective revision | RECOMMENDED (integrity) | Test outcomes were already inspected | all | Stated in Methods and Limitations |

## Blocking questions (only these need an answer)

| ID | Question | Why blocking | Blocks | Default if no answer |
|---|---|---|---|---|
| **D-13** | May Streamlit be installed (one pinned package) for the demo in Week 3? | Installing software was not authorised | T-17 | Static HTML demo |
| **D-14** | Has ITPU confirmed that no ethics review is needed for public, de-identified datasets? The documents assert this, but no record exists | Only the institution can determine this | Thesis Ch. 3 / ethics statement (T-21) | The thesis states "institutional confirmation pending" — **not acceptable for final submission** |
| **D-15** | What is the actual status of the IOCTe 2026 abstract (submitted? accepted?), and is a correction to the organisers needed? | Affects external communication, not the experiments | T-21 | No action on the abstract; the thesis uses corrected wording |
| **D-24** | The AI-assistance declaration must be rewritten by the researcher to match the real history: AI assistants wrote substantial code and the audit documents | Only the author can make this declaration | T-21 | — (cannot be defaulted) |

Non-blocking: the supervisor is expected to review `02_…`. Their approval is desirable before T-11 but is not assumed. No other decision needs the researcher's input. Library choices, batch sizes and step counts are settled by the pilot.
