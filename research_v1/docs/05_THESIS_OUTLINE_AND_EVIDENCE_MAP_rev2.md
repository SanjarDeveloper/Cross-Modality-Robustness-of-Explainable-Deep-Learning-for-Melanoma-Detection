# 05 — Thesis Outline and Evidence Map (rev2)

Supersedes `05_THESIS_OUTLINE_AND_EVIDENCE_MAP.md`, which is preserved unchanged. The university template has not been provided yet, so this document does not assume any of its requirements.

**Marker:** [PLACEHOLDER] = text that depends on results and must be written from the v1 analysis outputs only.

## 1. Provisional title

*Melanoma Detection under Dermoscopy-to-Clinical and Smartphone Image Shift: A Reproducible Evaluation of Performance and Explanation Behaviour in Convolutional Neural Networks*

## 2. Outline and writing schedule

| Chapter | Content | Written in |
|---|---|---|
| Abstract | Aim, data, methods, results, limits | Week 4 [PLACEHOLDER] |
| 1 Introduction | Motivation; the melanoma vs non-melanoma task; RQ1–RQ5; contributions, with no novelty claim unless the literature check supports one | **Week 1 (W-1)** |
| 2 Related Work | CNN melanoma classification; modality shift, noting that [4] concerns non-melanoma skin cancer (NMSC); saliency and faithfulness evaluation; skin-tone datasets | **Week 1 (W-1)**, using verified references only |
| 3 Data | Datasets, licences, cohort flow, same-file pairs, identifiers, negative-class composition, RQ5 partition | Week 2 |
| 4 Methods | Threshold rule; legacy reproduction (LR) vs V1; preprocessing; perturbations; XAI (attribution interpretation, controls, estimands, eligibility); statistics and testing families; seeds; provenance; **Protocol history and data exposure** | Week 2 |
| 5 Results | RQ1–RQ5; component tables; LR report in the appendix | Week 3 [PLACEHOLDER] |
| 6 Demonstration application | Design; consistency check | Week 3 |
| 7 Discussion | Interpretation of positive, null and inconclusive results, as they occur | Week 4 [PLACEHOLDER] |
| 8 Limitations and threats to validity | Retrospective revision; clinical photos are not smartphone images; single split; threshold sampling variability; selection caused by invalid maps; parameterised simulator; small RQ5 sample | Draft Week 2, final Week 4 |
| 9 Conclusion and future work | Includes the deferred composite score and calibration | Week 4 [PLACEHOLDER] |
| Appendices | Manifest cards, configs, tests, v0→v1 changes, LR difference report, AI-assistance declaration (author-approved) | Week 4 |

## 3. Tables and figures → required outputs

| ID | Content | Source |
|---|---|---|
| Tab 3.1 / Fig 3.1 | Dataset summary; cohort flow | `manifests/*card*.json` (T-A2) |
| Tab 3.2 | RQ5 partition: patients and images per role and stratum | RQ5 manifest (T-A2) |
| Tab 4.1 | Version keys and threshold rule | `run.json` |
| Tab 4.2 | Endpoints and testing families | 02_rev2 |
| Tab 5.1 | HAM baseline per model, with validation threshold statistics | V1 analysis |
| **Tab 5.2** | E1.1–E1.4, primary cohort 1002; sensitivity analysis 1011 | analysis_rq1 |
| Tab 5.3 | PAD results with patient-cluster CIs | analysis_rq1 |
| Fig 5.1 / Tab 5.4 | RQ2 curves; worst-case loss; share of bootstrap replicates with no crossing; rank proportions | analysis_rq2 |
| Tab 5.5 | RQ3 per-seed matrix | analysis_rq3 |
| **Tab 5.6** | E4.1 control-referenced raw contrasts, per explainer, curve and modality, with Holm-adjusted p-values | analysis_rq4 |
| **Tab 5.7** | E4.2 shift contrasts; invalid-map rates; matched-pair n; sensitivity analysis on all pairs | analysis_rq4 |
| Tab 5.8 | Normalised scores (secondary): defined, negative-d and undefined counts by modality × class; E4.3 adjusted vs unadjusted | analysis_rq4 |
| Fig 5.2 | Mean curves with control bands | curves |
| Tab 5.9 | IG completeness: absolute and relative errors; failure rate | xai rows |
| Tab 5.10 | E4.6 grouped-CV results vs probability-only baseline | analysis_rq4 |
| Tab 5.11 | RQ5 primary AUROC difference; secondary A and B; forgetting check | analysis_rq5 |
| Tab 5.12 | Skin-tone counts and descriptive metrics | analysis_rq5 |
| Tab 5.13 | Reliability component table (the composite is deferred) | analysis outputs |
| Tab A.1 | LR vs legacy aggregates, with difference investigation | LR run |

## 4. Major claims → required evidence

The wording below is a template. It does not assume the direction of any result.

| Claim | Evidence | If not supported |
|---|---|---|
| "In this cohort, melanoma sensitivity on clinical photographs [was lower / higher / not detectably different] than on paired dermoscopic images (primary retrospective analysis)" | E1.1 (a one-test family) + E1.2 CI | Report the null or inconclusive result with its CI |
| "AUROC differed …" (secondary, unadjusted) | E1.3 | as above |
| PAD performance | E1.5 | Descriptive only |
| "Within the chosen parameters, factor X most often produced the largest loss" | Rank proportions | "No stable ranking" |
| Architecture consistency | Tab 5.5 | Descriptive only |
| "Explainer E [exceeded / did not exceed] the random control on [curve] for [modality], among eligible pairs" | E4.1, Holm-adjusted | No faithfulness-under-shift statement for that explainer and curve |
| "Control-referenced scores changed under shift" | E4.2, Holm-adjusted, with its eligibility population | Report the null result |
| "The difference was reduced after adjustment for the change in confidence" | E4.3 | Wording: "**associated with** the confidence change". **Never**: "attributable to" or "caused by" |
| Error prediction beyond probability | E4.6 | Report the null result |
| RQ5 adaptation | Tab 5.11 (primary AUROC difference) | "Insufficient evidence to determine" |
| Skin tone | Tab 5.12 | No fairness verdict |
| **Forbidden** | Clinical readiness; "smartphone" causation inferred from Derm7pt; novelty without a literature check; any "validated" composite score; "no leakage of any kind"; "preregistered" or "confirmatory on untouched data"; eligible-pair results presented as if they covered all cases | — |

## 5. External abstract

Its status is unknown. No external action is authorised. The thesis uses the corrected wording in 01_rev2 §7. Any correction sent to the conference is solely the author's decision.
