# 05 — Thesis Outline and Evidence Map

The university template has not arrived yet. The chapter structure below is provisional and makes no assumptions about template requirements (length, citation style, front matter).

Marker: **[PLACEHOLDER — results-dependent]** means the text may only be written from the v1 analysis outputs.

## 1. Provisional title

*Melanoma Detection under Dermoscopy-to-Clinical and Smartphone Image Shift: A Reproducible Evaluation of Performance and Explanation Behaviour in Convolutional Neural Networks*

The original title is also acceptable. Its phrase "Generalise to Smartphone Images" must be qualified in the text, because Derm7pt clinical photographs are not verified smartphone captures.

## 2. Outline

| Ch. | Content | When it can be written |
|---|---|---|
| Abstract | Aim, data, methods, key results, limits | Week 4 **[PLACEHOLDER]** |
| 1 Introduction | Motivation (melanoma, triage, modality shift); the melanoma vs non-melanoma task; RQ1–RQ5; contributions stated without novelty claims until T-02 checks the literature | Week 1–2 |
| 2 Related Work | CNN melanoma classification; modality shift (with the NMSC caveat on [4]); saliency methods and faithfulness evaluation (RISE insertion/deletion, IG, sanity checks); skin-tone datasets | Week 1–2, after T-02 |
| 3 Data | Datasets, licences, cohort flow, same-file pairs, missing patient IDs, negative-class composition | Week 2 (from T-04 cards) |
| 4 Methods | Model training, thresholds, preprocessing versions, perturbations, XAI and controls, statistics, multiplicity, seeds, provenance; **Protocol history and data exposure** | Week 2 |
| 5 Results | RQ1–RQ5, CMRS | Week 3 **[PLACEHOLDER]** |
| 6 Demonstration application | CMEX design and consistency check | Week 3 |
| 7 Discussion | Interpretation per RQ, including null and inconclusive outcomes | Week 4 **[PLACEHOLDER]** |
| 8 Limitations and Threats to Validity | Retrospective revision, clinical ≠ smartphone, one split, small subgroups, parameterised simulator, faithfulness definition | Draft Week 2, finalise Week 4 |
| 9 Conclusion and Future Work | | Week 4 **[PLACEHOLDER]** |
| Appendices | Manifests summary, configs, test list, v0→v1 change table, AI-use declaration (written by the researcher) | Week 4 |

## 3. Planned tables and figures → required outputs

| ID | Table / figure | Source output | Task |
|---|---|---|---|
| Tab 3.1 | Dataset summary and negative-class composition | `manifest_card.json` | T-04 |
| Fig 3.1 | Cohort flow diagram | manifests | T-04 |
| Tab 4.1 | Version keys (prep, aug, perturb, metric, xai, cmrs) | `run.json` | T-03 |
| Tab 4.2 | Endpoint table (E1.1 … E4.6, RQ5) | `02_…` | — |
| Tab 4.3 | Perturbation parameters | `perturb_version` config | T-05 |
| Fig 4.1 | Perturbation examples, image and mask | QC grid | T-05 |
| Tab 5.1 | HAM baseline per model | predictions + analysis_rq1 | T-11/T-14 |
| Tab 5.2 | **E1.1–E1.4 Derm7pt paired results (primary 1002; sensitivity analysis 1011)** | analysis_rq1.json | T-07/T-11/T-14 |
| Tab 5.3 | PAD results with patient-cluster CIs | analysis_rq1.json | T-14 |
| Fig 5.1 | Severity curves per factor with CIs | analysis_rq2.json | T-14 |
| Tab 5.4 | Worst-case loss and rank stability | analysis_rq2.json | T-14 |
| Tab 5.5 | RQ3 per-seed matrix | analysis_rq3.json | T-15 |
| Tab 5.6 | **E4.1 explainer vs controls, per modality** | analysis_rq4.json | T-12/T-14 |
| Tab 5.7 | **E4.2 shift effect: raw / normalised / adjusted** | analysis_rq4.json | T-14 |
| Fig 5.2 | Mean insertion/deletion curves with control bands | curves.npz | T-14 |
| Fig 5.3 | Qualitative examples (pre-selected by rule, not cherry-picked) | demo subset | T-12 |
| Tab 5.8 | Drift→error CV results vs probability-only baseline | analysis_rq4.json | T-14 |
| Tab 5.9 | RQ5 adaptation, PAD evaluation patients | analysis_rq5.json | T-16 |
| Tab 5.10 | Skin-tone counts and descriptive metrics | analysis_rq5.json | T-16 |
| Tab 5.11 | CMRS with raw components and axis signature | analysis_cmrs.json | T-14 |
| Tab A.1 | v0→v1 change table and legacy vs v1 reproduction | T-07 output | T-07 |

## 4. Major claims → required evidence

| Claim (template wording; direction not assumed) | Required evidence | If it is not supported |
|---|---|---|
| "Melanoma sensitivity on clinical images was [lower/not different] than on paired dermoscopic images in this cohort" | E1.1 exact McNemar on 249 pairs + E1.2 CI, seed 0; seeds 1–2 direction | State the null/inconclusive result with the CI |
| "Discrimination (AUROC) differed…" | E1.3 | as above |
| "On PAD-UFES-20 smartphone images the model achieved …" | E1.5 with patient-cluster CI | descriptive only |
| "Within the chosen perturbation parameters, factor X produced the largest worst-case loss" | E2.1 + rank-stability ≥ 0.8 (proposed descriptive rule) | "No stable ranking" |
| "Findings were [consistent/inconsistent] across architectures" | Tab 5.5 | descriptive only |
| "Grad-CAM / IG [outperformed / did not outperform] control maps" | E4.1 | If not outperformed → no faithfulness-under-shift claim for that explainer |
| "Control-referenced insertion/deletion scores changed under the clinical modality" | E4.2 + E4.3 | If the change disappears after adjustment → "attributable to the confidence change" |
| "Explanation features [added / did not add] predictive value for errors beyond probability" | E4.6 CV | Report the null |
| "Limited fine-tuning [changed / did not measurably change] PAD performance" | Tab 5.9 | "insufficient data" |
| Skin tone | Tab 5.10 counts only | Never a fairness verdict |
| **Forbidden claims** | Clinical readiness; causal attribution to "smartphone" from Derm7pt; novelty without a literature check; "validated" CMRS; "no leakage of any kind"; "preregistered" | — |

## 5. External abstract (IOCTe) — status and correction

The current "Results" section of the abstract states RQ4 findings that are not supported (`01_…` I-10). Its submission and acceptance status is unknown (I-15, I-16).

Proposed replacement sentence, for use until the v1 results exist:

> "Results. Analyses are in progress. A preliminary single-model analysis indicated lower melanoma sensitivity on paired clinical photographs than on dermoscopic images. Explanation analyses with control maps and confidence adjustment are pending."

The researcher must decide whether a correction to the organisers is needed (D-15).

## 6. Parallel writing schedule

| Week | Writing (≈ 12 h/week inside the backlog) |
|---|---|
| 1 | Introduction draft; Related Work skeleton linked to T-02 |
| 2 | Data and Methods complete; Limitations draft |
| 3 | Results from the analysis JSON; Demo chapter |
| 4 | Discussion, Conclusion, Abstract; full consistency pass |
