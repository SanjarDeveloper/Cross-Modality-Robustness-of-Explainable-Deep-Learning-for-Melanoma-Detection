# 01 — Current State and Corrections (rev2)

This is the Stage 1.1 revision. It supersedes the `01_CURRENT_STATE_AND_CORRECTIONS.md` document, which is kept unchanged. Prepared 2026-09-27.

Evidence labels:

- **[OBS]** Observed directly, either in a file or by a read-only command during Stage 1 or Stage 1.1.
- **[CODE]** A risk inferred by reading the code. It has not been confirmed by re-running anything.
- **[UNV]** Claimed in an earlier document but not independently re-checked.
- **[REC]** A recommendation.

**Anchor convention.** Code locations are written as `file:function` plus a line number. Line numbers were checked with `grep -n` against the files as they stood on 2026-09-27. The rev1 document took several line numbers from a concatenated listing; those were wrong. rev2 corrects them (see the changelog, item 12).

---

## 1. Inputs reviewed

The inputs are the same as in rev1 §1.

Read-only checks added in Stage 1.1:

- **Line numbers** were checked with `grep -n` in every cited file. File lengths: `data.py` 275, `run_eval.py` 282, `eval/xai.py` 166, `eval/metrics.py` 162, `msim/simulator.py` 384, `rq4_test.py` 123, `analyse.py` 255, `cmex_app.py` 181, `train.py` 199, `campaign.py` 269, `fairness.py` 153.
- **Albumentations 2.0.8.** `Compose.__init__` accepts a `seed` parameter. `Compose` has `set_random_seed` and `set_random_state` methods. Only their existence was checked; their behaviour was not tested.
- **PAD-UFES-20 metadata.**
  - 1373 patients.
  - 179 patients have more than one distinct diagnosis.
  - 36 patients have at least one melanoma image; 11 of these also have other diagnoses.
- **Reference [4]** was re-read on PMC (see §6).

---

## 2. What exists and what was executed

### 2.1 Legacy scientific runs (protocol v0, unfrozen; one model: ResNet-50 `standard`, seed 0, HAM-only training)

| ID | Artifact | Content [OBS] | Main limitation |
|---|---|---|---|
| L-01 | `checkpoints/resnet50_standard_s0.pt` | 8 epochs; threshold 0.12517 | No config block, no source or environment hash, and no optimizer, scheduler or scaler state |
| L-02a | `results/baseline_resnet50_standard_s0.json` | HAM test: n=1548, 170 melanomas; sens 0.965, spec 0.583, AUROC 0.917, ECE 0.173 | Aggregates only. Uncertainty was computed per image, not per lesion |
| L-02b | `results/real_resnet50_standard_s0.json` | Derm7pt dermoscopy: sens 0.948, AUROC 0.817, ECE 0.193. Derm7pt clinical: 0.802, 0.688, 0.087. PAD: sens 0.942 (52 melanomas), spec 0.356, AUROC 0.765, ECE 0.314 | No per-image rows |
| L-03 | `results/derm7pt_paired.json` (untagged) | Drop 0.147, CI [0.095, 0.196]; DeLong p=3.8e-9; McNemar b=139, c=183, p=0.016 | Ambiguous provenance. The McNemar test measures overall correctness |
| L-02c | `results/sweep_resnet50_standard_s0.json` | 30 rows: 5 factors × severities {0, .2, .4, .6, .8, 1} | Stochastic factors cannot be reproduced exactly (F-05) |
| L-02d | `results/xai_resnet50_standard_s0.json` | 13 rows. Derm7pt: 400 enriched pairs (200 melanoma / 200 non-melanoma). HAM: 400 images (170/230) × 10 simulated conditions | Only means saved. The metric is a mean over steps. No controls |
| L-02e | `results/rq4_tests_resnet50_standard_s0.json` | Wilcoxon p=1.3e-22, d=−0.56. Mann-Whitney p=3.2e-15. In-sample drift→error ROC-AUC 0.66 (coefficients: insertion −1.41, deletion +1.80) | In-sample, and confounded by model confidence |
| L-04, L-05 | Figures; simulator QC grids | Illustrations and QC only | — |
| L-06 | `cache/*.npy` (7 arrays, 256 px) | Metadata and pixel spot checks reported [UNV] | Double-resize pathway (F-10) |

**Historical preprocessing, best evidence.**

- The legacy code chose the cached pathway whenever `cache_exists()` returned true: `run_eval.py:predict` line 46, `train.py:main` line 111, `rq4_test.py:per_image` line 29.
- According to the audit, all 7 caches existed when it was run [UNV]. The cache creation times relative to the legacy runs have not been established.
- Legacy inference used fp16 autocast: `run_eval.py:predict`, and `train.py:infer` line 58.

Two historical settings remain **unknown**:

1. Whether every legacy result was produced from the cache. This is the most likely case, but it is not proven.
2. The exact package versions at run time.

### 2.2 Software smoke runs (not scientific evidence)

- **S-01:** `runs/smoke/phase1-20260920`. Reported 44/44 stages [UNV].
- **S-02:** `runs/smoke/phase1-final-20260920`. Config [OBS]: `{'smoke': True, 'archs': ['resnet50'], 'strats': ['standard'], 'seed': 0, 'fairness': True}`. Nine step directories are present. Reported 9/9 [UNV].

Smoke profile: 1 epoch, a 300-image training subset, 8 deletion/insertion steps. **No smoke number may appear in scientific text.**

### 2.3 Audit and protocol work (2026-09-20)

The following exist and remain useful:

- manifests;
- label coverage;
- the duplicate audit;
- the cache pixel audit;
- the replacement power model;
- the endpoint drafts;
- the decision package.

Where their statements are too strong, the claims are corrected in §5.

---

## 3. Reuse, correct, or keep as unverified

This section follows rev1 §3, with these amendments:

- `metrics.threshold_at_sensitivity` (`eval/metrics.py` line 32): **replace** with the protocol-v1 threshold rule (see 02_rev2 §1). The legacy function returns the first `roc_curve` point where TPR ≥ target, i.e. the highest ROC threshold that reaches the target. Its handling of ties, and whether the result is sample-dependent, have not been specified or tested. The legacy value is kept only for legacy reproduction.
- `msim` Albumentations use: Albumentations 2.0.8 provides seed interfaces (§1). The research_v1 design uses a stateless per-sample seed (03_rev2 §8), so it does not need to restore an internal state.
- Unverified items that remain open:
  - which preprocessing pathway was used for each legacy number;
  - smoke receipt contents;
  - the arithmetic of the audit JSON files;
  - the reported test pass counts, which were not re-run.

---

## 4. Prioritized findings (line anchors checked)

**Priority:** P1 blocks a primary claim; P2 blocks a secondary claim or reproducibility; P3 affects documentation.

| ID | P | Finding | Type | Location (`file:function`, line) |
|---|---|---|---|---|
| F-01 | P1 | The legacy McNemar test covers all 1011 pairs and tests overall correctness. On that endpoint the clinical images do *better* (b=139 < c=183), while melanoma true positives fall (239→202; decomposition taken from the audit, [UNV]) | OBS | `run_eval.py:suite_real` line 92; `eval/metrics.py:mcnemar` lines 69–84 |
| F-02 | P1 | No per-image predictions, identifiers or curves were persisted. Only aggregates were written | OBS | `run_eval.py:suite_real` line 109; `run_eval.py:suite_xai` line 192; `rq4_test.py:main` lines 115–118 |
| F-03 | P1 | The Derm7pt loader applies no exclusion. The audit reports 9 same-file pairs, 3 of them melanomas [UNV here] | OBS / UNV | `data.py:derm7pt` line 81 |
| F-04 | P1 | Insertion and deletion integrate the raw melanoma probability. Grad-CAM targets the melanoma logit for every case. No control maps exist. Deletion improves (0.127→0.103) while insertion worsens (0.330→0.198) | OBS | `eval/xai.py:GradCAM.__call__` line 67; `eval/xai.py:deletion_insertion` line 127 |
| F-05 | P1/P2 | The simulator seeds its generator with `hash((seed, factor, h))`. Because this tuple contains a string, the result depends on PYTHONHASHSEED, which is unset and not set by the runner. A two-process check gave different values | OBS | `msim/simulator.py:ModalityShift._rng` line 335; `campaign.py:run_step` env at line 168 |
| F-06 | P2 | The "AUC" is the unweighted mean of the S+1 curve points, not a trapezoidal integral | OBS | `eval/xai.py:deletion_insertion` line 127 |
| F-07 | P2 | `scale` changes only the image, but the mask is read afterwards, unchanged. IoU is still computed for every factor | CODE | `data.py:CachedDataset.__getitem__` lines 213–215; `run_eval.py:suite_xai` lines 185 and 232 |
| F-08 | P2 | The split seed and the training seed are the same variable | OBS | `train.py:main` lines 90 and 96; `run_eval.py` lines 119, 132, 166 |
| F-09 | P2 | Albumentations is not seeded. Seeding torch and numpy does not control it | CODE | `data.py:LesionDataset._build` line 146; `train.py:_heavy_tf` line 29; `train.py:main` line 90 |
| F-10 | P2 | The pathway (raw or cached) is chosen silently and never recorded. The magnitude of the difference (max 1.93 normalised) comes from the audit [UNV] | OBS | `run_eval.py:predict` line 46; `run_eval.py:suite_xai` line 165; `train.py:main` line 111; `rq4_test.py:per_image` line 29 |
| F-11 | P2 | The demo CMRS uses a ratio of per-image probabilities, while the report CMRS uses a ratio of dataset sensitivities. The demo labels negatives "benign" and uses its own preprocessing | OBS | `cmex_app.py:cmrs_live` line 74; `cmex_app.py:to_tensor` line 44; lines 118, 132, 152; `analyse.py:cmrs` line 117 |
| F-12 | P2 | The replacement power model uses the observed test sensitivity together with assumed discordance scenarios. Its outputs are conditional planning figures only | OBS | `protocol/POWER.md`; `protocol/power_analysis.py:main` |
| F-13 | P2 | A constant Grad-CAM map normalises to all zeros. `argsort` is then applied to tied values, so the pixel order is implementation-defined. Localisation then marks every pixel as salient. Chunking uses `n // steps`, so full pixel coverage holds only when N is divisible by S (true for 224² with S=32) | CODE | `eval/xai.py:GradCAM.__call__` line 76; `eval/xai.py:deletion_insertion` lines 109–110; `eval/xai.py:localisation` line 155 |
| F-14 | P2 | No calibration step exists anywhere, although the documents say "calibrated probabilities" | OBS | `train.py:main` (no calibration code); `eval/metrics.py:ece` line 44 |
| F-15 | P2 | The drift→error model is fitted and scored on the same data. Its features need both twin images | OBS | `rq4_test.py:main` lines 102–106 |
| F-16 | P2 | The XAI subset is drawn from all 1011 pairs, including the same-file pairs | CODE | `run_eval.py:suite_xai` line 215; `rq4_test.py:main` line 64 |
| F-17 | P3 | The documentation cites `calibrate.py`, which does not exist | OBS | `msim/simulator.py` module docstring, line 16 |
| F-18 | P3 | A comment still cites the invalid power figure of 0.78 | OBS | `data.py` module docstring, line 7 |
| F-19 | P3 | The DDI fairness code reports malignant-vs-benign as its main table | OBS | `fairness.py:ddi_by_tone` line 32 |
| F-20 | P3 | Streamlit is not installed. **This is an implementation detail for later, not a blocker now** | OBS | `importlib.util.find_spec` |
| F-21 | P2 | Threshold crossings are interpolated on a 6-point grid and reported without any uncertainty | OBS | `analyse.py:thresholds` line 22 |
| F-22 | P2 | The legacy operating point is the highest ROC threshold with TPR ≥ 0.95 on validation *images*. Tie handling and the estimation unit are not documented | OBS | `eval/metrics.py:threshold_at_sensitivity` line 32 |

---

## 5. Inconsistencies between documents, code and results

The items are the same as rev1 §5, with these corrections:

- **I-3 (resolved to a discrepancy in the source itself).** The published abstract of reference [4] says 95.3 %; the body text and Table 2 say 95.5 % (see §6). The plan quotes the abstract figure. This matters little, because the key correction is that the paper studies non-melanoma skin cancer (NMSC), not melanoma.
- **I-14 Ethics.** The requirement is **unresolved**, pending actual institutional guidance. This document does not assume any rule about submission.
- **I-15, I-16 Abstract status.** Unknown. No external communication is authorised by this package.
- **I-17 AI declaration.** An accurate draft for the author to review is in 06_rev2, Appendix A.

---

## 6. Reference verification

| Ref | Status | Location of the evidence |
|---|---|---|
| [4] Dascalu et al., *J Cancer Res Clin Oncol* 148(9):2497–2505, doi:10.1007/s00432-021-03809-x | **Topic verified: non-melanoma skin cancer (NMSC), not melanoma.** Figures as rendered on the PMC page, obtained through a fetch tool (to be confirmed by a manual reading of the PDF): **Abstract, Results:** AUC 0.911 vs 0.821; sensitivity "95.3 %, CI 90.4–98.3 vs 75.3 %, CI 68.1–81.6". **Body, Results, confusion-matrix paragraph, and Table 2:** sensitivity 95.5 % vs 75.3 %. **Body, ROC paragraph:** AUC 0.911 (0.858–0.964) and 0.821 (0.738–0.905). The dermoscopic and smartphone datasets are different image sets (165 vs 198 images, per the same tool extraction) | https://pmc.ncbi.nlm.nih.gov/articles/PMC8453469/ |
| [17] doi:10.1002/ail2.77 | **Author and title identified from a web-search listing only:** Bokadia H, Yang SC, Li Z, Folke T, Shafto P, "Evaluating perceptual and semantic interpretability of saliency methods: a case study of melanoma", *Applied AI Letters*. The full text was not read. **The claim "lower faithfulness for melanoma than benign" remains UNRESOLVED** | Web search result; the DOI redirected to Wiley and was not read |
| [5], [6], [9] and all others | **UNRESOLVED.** To be checked in the reference-verification task, after the next task | — |

A figure that a tool has extracted from a page is treated as a pointer to where the evidence is. It is not treated as a verified quotation until it has been read in the source document itself.

---

## 7. Claim corrections

These are the same as rev1 §7, with the following amendments:

- **C-3.** "Dascalu et al. [4] reported higher sensitivity on dermoscopic than on smartphone images for **non-melanoma skin cancer**, using separate image sets (abstract: 95.3 % vs 75.3 %; the body and Table 2 report 95.5 % for dermoscopy). These figures are not a melanoma benchmark and are not directly comparable with the present task."
- **C-7 (XAI).** Additional sentence: "A reduced difference after confidence adjustment would indicate that the observed difference is statistically associated with the change in confidence. It would not establish that confidence caused it."
- **C-14.** "The protocol was revised retrospectively, after the legacy test results had been inspected. The primary analyses are **primary retrospective analyses**. They are not confirmatory tests on untouched data."
- **C-12 (CMRS).** "A composite reliability score was deferred until its components are fully defined. Component-level results are reported."

---

## 8. Priorities

1. Freeze the cohorts and the manifests (next task).
2. Separate the legacy reproduction from the protocol-v1 analyses.
3. Persist results per image, and add the melanoma-only E1.1 endpoint.
4. Add controlled XAI with explicit estimands and eligibility rules.
5. Make the reproducibility conditions explicit: seeds, preprocessing and mask geometry.
