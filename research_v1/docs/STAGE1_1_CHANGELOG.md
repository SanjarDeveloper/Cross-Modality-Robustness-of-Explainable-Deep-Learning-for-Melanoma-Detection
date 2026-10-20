# Stage 1.1 Changelog

Date: 2026-09-27.

## Scope of Stage 1.1

- Six planning documents were revised and saved as `*_rev2.md`.
- The original six documents were preserved unchanged.
- A new file, `NEXT_TASK_SPEC_STAGE2A.md`, was added.
- No legacy files, datasets, source code or experimental outputs were changed.
- Nothing was installed, trained or run for inference. No one was contacted.

## Read-only checks added in Stage 1.1

- Line anchors were verified with `grep -n`. File lengths were recorded.
- Albumentations 2.0.8 `Compose` has `seed`, `set_random_seed` and `set_random_state`. Only their existence was checked; their behaviour was not tested.
- PAD-UFES-20 metadata:
  - 1373 patients;
  - 179 patients with more than one diagnosis;
  - 36 melanoma patients, 11 of whom also have other diagnoses.
- Reference [4] was re-read on PMC. The abstract and the body text report different values (see item 12).

## Review items

| # | Item | Correction | Where | Remaining uncertainty |
|---|---|---|---|---|
| 1 | Operating threshold | The threshold is the highest t with image-level validation sensitivity ≥ 0.95, i.e. the k-th largest melanoma probability with k = ⌈0.95·n₊⌉. Ties are counted as positive (p ≥ t). If n₊ < 20 or the scores are degenerate, the threshold is "not estimable" and there is no fallback. Lesion-level sensitivity is reported descriptively | 02_rev2 §1.1; 03_rev2 §14; D-25 | The variability of the threshold itself is not propagated into CIs; this is stated as a limitation |
| 2 | Legacy reproduction (LR) vs V1 | These are now separate activities: different cohorts, preprocessing and inference precision. LR uses the best-evidenced historical settings; unknown settings stay unknown. ±0.005 is a diagnostic tolerance only, with a defined investigation and reporting procedure. V1 is never tuned toward legacy values | 02_rev2 §1.3; 03_rev2 §1, §6, §13; 04_rev2 T-C1/T-C2; D-26 | Whether every legacy result used the cache is still unproven |
| 3 | RQ4 estimands | Raw v1 scores are primary. Insertion is scored as explainer − control and deletion as control − explainer. The five random seeds are averaged within each image. The population is the set of matched eligible pairs. The RQ4 family has 12 tests, Holm-adjusted. A gate is pre-specified | 02_rev2 §6.2–6.4; D-27 | — |
| 4 | Normalisation | Defined only when d ≥ δ = 0.05. Cases with d ≤ −δ form their own category; cases with \|d\| < δ are undefined. Values are not clipped. Missingness is reported by modality × class. Raw scores stay primary. It is stated that normalisation does not remove confounding. δ rationale: noise amplification ≤ 20×, with sensitivity analyses at δ = 0.02 and 0.10 | 02_rev2 §6.5 | The choice of δ is judgement-based (OS-1) |
| 5 | IG comparability | The primary attribution for both explainers is positive, melanoma-supporting evidence: IG uses a signed sum over channels followed by ReLU. Neither explainer is smoothed in the primary analysis. Absolute and smoothed IG are secondary. Completeness is checked with absolute and relative errors, handling near-zero denominators, a convergence protocol and failure-rate reporting | 02_rev2 §6.1, §6.6; 03_rev2 §10; D-28 | IG positive maps may often be all-zero for non-melanoma images; the rate is unknown until the pilot |
| 6 | Constant or undefined maps | Invalid maps are flagged, not dropped. The analysis uses matched eligible pairs. Invalid rates are reported by explainer × modality × class. The selection effect is described. A sensitivity analysis includes all pairs "as computed". Results are not presented as covering all cases | 02_rev2 §6.4; 03_rev2 §9; 05_rev2 §4 | — |
| 7 | RQ5 | Image-level McNemar was removed. The patient partition is frozen with a priority rule for strata (which also covers patients with multiple diagnoses) and a stable-hash split of 50/35/15. Model selection on adapt-val is defined (epoch AUROC, with a tie rule). Both models are evaluated on identical eval patients. The primary contrast is the AUROC difference with a patient-cluster paired bootstrap. Secondary A (original threshold rule) and Secondary B (threshold reselected on adapt-val) are reported separately. The prediction that the result would be "null/inconclusive" was removed | 02_rev2 §7; NEXT_TASK_SPEC; D-29 | With 18 melanoma patients in eval, precision will be limited. This is a property of the data, not a predicted outcome |
| 8 | CMRS | Deferred until its components, directions, denominators, populations and weights are fully defined. Component tables are required. CMRS does not block any RQ | 02_rev2 §8; 03_rev2 §4, §15; 05_rev2 Tab 5.13; D-10 | Whether to define it at all is OS-4 |
| 9 | Statistics and wording | The RQ1 family is E1.1 alone at α = 0.05; E1.3 is secondary and unadjusted. The RQ4 family is 12 tests with Holm correction. The term "primary retrospective analysis" is used, and "confirmatory" is no longer used. The bootstrap estimand, paired cluster resampling and handling of undefined replicates (with a 5% flag) are defined. The share of bootstrap replicates with no crossing is kept (censored at ">1.0"). Adjustment language is corrected to "associated with", not "caused by" | 02_rev2 §2–§4, §6.5; 05_rev2 §4; 01_rev2 C-7, C-14 | — |
| 10 | Correctness tests | Grad-CAM is tested on a controlled toy network, including a negative-weight case. The random-control symmetry is now a Monte Carlo expectation under a stated mean-pool assumption (500 maps, within 3 standard errors). The cluster-duplication test is precisely defined and preserves cluster identity. The resume test uses the same total epochs and `T_max` for both runs. Actual RNG states are saved, and augmentation is stateless. Numeric tolerances are predefined | 03_rev2 §8, §12, §14 | The behaviour of `set_random_seed` is still unverified (T-B2 test, with a fallback) |
| 11 | Dependencies and time | The XAI module (T-D2) now comes before the pilot (T-E1). Introduction and Related Work are budgeted for Week 1 (8 h). The Stage 1 and 1.1 work is counted (8 h, estimated). The total is 146 h. Compute is listed separately and unattended GPU time is not assumed. A validity-repair reserve is added for Week 4, and known-invalid results are never reported | 03_rev2 §3; 04_rev2 | All hours are estimates |
| 12 | References and evidence | Rev1 line numbers were wrong because they were taken from concatenated listings (for example, `xai.py:289` and `run_eval.py:291`). They were replaced with `file:function` anchors and line numbers checked with grep. Tool summaries are treated as pointers, not quotations. For [4]: the abstract reports 95.3%, while the body text and Table 2 report 95.5% (per the PMC page, extracted with a tool; still to be confirmed against the PDF). The key correction remains that [4] studies non-melanoma skin cancer (NMSC). [17] is still unresolved | 01_rev2 §4, §6 | Manual reading of the PDF of [4]; [5], [6], [9] and the other references are still unresolved |
| 13 | Administrative items | The Streamlit installation is now an implementation detail, not a blocker. Ethics is UNRESOLVED pending actual institutional guidance, with no rule assumed. An accurate AI-declaration draft has been written for the author to review. No external messaging is authorised | 06_rev2 A-1…A-4, Appendix A; 05_rev2 §5 | Depends on the author and the institution |

## Additional consistency fixes

- The rev1 claim "reproduce legacy within ±0.005 = acceptance" was removed from 03 and 04.
- The rev1 per-image CMRS and colour bands were removed from every rev2 document.
- The dependency order in 03_rev2 §3 now matches the task order in 04_rev2.

## Proposed next task

`NEXT_TASK_SPEC_STAGE2A.md` covers:
- the read-only environment inventory;
- the v1 manifests, with minimal validation.

It excludes training, perturbation, statistics, inference and XAI. **It has not been executed.**
