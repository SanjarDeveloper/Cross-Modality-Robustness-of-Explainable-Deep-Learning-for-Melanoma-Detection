# 01 — Current State and Corrections

Stage 1 planning package · research_v1 · prepared 2026-09-27
Status: planning document. No code was changed, no model was trained, no inference was run.

Evidence labels used throughout:

- **[OBS]** Observed fact: read directly from a file or produced by a read-only command during this review.
- **[CODE]** Code-based risk: follows from reading the source. It has not been confirmed by re-running an experiment.
- **[REC]** Recommendation.
- **[UNV]** Unverified: stated in an earlier document, but not confirmed independently in this review.

---

## 1. Inputs reviewed

| Input | Scope of review |
|---|---|
| `Research_Plan_Sanjar_Bakhtiyorov (1).docx` | Full text and all tables (Ch. 1–10, references [1]–[24]) |
| `Detailed_Technical_Description_Sanjar_Bakhtiyorov (1).docx` | Full text and tables |
| `Publication_Sanjar_Bakhtiyorov (2).docx` | Full text: IOCTe 2026 abstract and approval note |
| `README.md`, `SESSION_HANDOFF.md` | Full |
| `audit/` | STAGE1_REPORT, STAGE2_REPORT, DECISION_LOG, RUN_REGISTRY, TASK_REGISTRY, DATASET_CARDS, `requirements_observed.txt`, file listing of the JSON evidence and manifests |
| `protocol/` | DECISION_PACKAGE, ENDPOINTS, POWER, `metric_versions.json`, and the structure of `power_analysis.py` |
| `code/` | Read line by line: `data.py`, `msim/simulator.py`, `eval/metrics.py`, `eval/xai.py`, `train.py`, `run_eval.py`, `rq4_test.py`, `analyse.py`, `cmex_app.py`. Partly read: `campaign.py` (environment and attempt logic), `fairness.py`, `precache.py`, `validate_labels.py` |
| `results/` | `baseline_*`, `real_*`, `derm7pt_paired.json`, `xai_*`, `rq4_tests_*`, and the structure of `sweep_*` |
| `runs/smoke/` | Directory structure and `config.json` of `phase1-final-20260920` |

Read-only checks run during this review:

1. `PYTHONHASHSEED` was empty. Two separate Python processes returned different values for `hash((12345,'blur',7)) % 2**32` (3172530629 and 2088632938).
2. `importlib.util.find_spec('streamlit')` returned `None`, so Streamlit is not installed.
3. The installed Albumentations version is 2.0.8.
4. A primary source was checked online for reference [4] (see §6).

---

## 2. What exists and what was actually executed

### 2.1 Legacy scientific runs (protocol v0, unfrozen)

All legacy runs use one model: ResNet-50, `standard` strategy, seed 0, trained only on HAM10000.

| ID | Artifact | What it contains [OBS] | Main limitation |
|---|---|---|---|
| L-01 | `checkpoints/resnet50_standard_s0.pt` | 8 epochs; threshold 0.12517 chosen on the validation set | No `config` block, no source hash, no environment record. No optimizer, scheduler or scaler state, so training cannot be resumed |
| L-02a | `results/baseline_resnet50_standard_s0.json` | HAM test: n=1548, 170 melanomas; sens 0.965, spec 0.583, AUROC 0.917, ECE 0.173 | Aggregates only. HAM test contains 91 melanoma lesions, but uncertainty was computed at image level |
| L-02b | `results/real_resnet50_standard_s0.json` | Derm7pt dermoscopy sens 0.948 / AUROC 0.817 / ECE 0.193. Derm7pt clinical 0.802 / 0.688 / 0.087. PAD sens 0.942 (52 melanomas), AUROC 0.765, spec 0.356, ECE 0.314 | No per-image predictions |
| L-03 | `results/derm7pt_paired.json` (untagged) | Drop 0.147, CI [0.095, 0.196]; DeLong p=3.8e-9; McNemar b=139, c=183, p=0.016 | Model provenance is ambiguous. The McNemar test measures overall correctness, not sensitivity |
| L-02c | `results/sweep_resnet50_standard_s0.json` | 30 rows: 5 factors × severities {0, .2, .4, .6, .8, 1} | Stochastic factors cannot be reproduced exactly (see F-05). Crossing points come from a 6-point grid |
| L-02d | `results/xai_resnet50_standard_s0.json` | 13 rows. Derm7pt 400 enriched pairs (200/200). HAM 400 images (170 melanoma / 230 non-melanoma) with 10 simulated conditions | Only means were saved. "AUC" is an arithmetic mean. No controls |
| L-02e | `results/rq4_tests_resnet50_standard_s0.json` | Wilcoxon p=1.3e-22, d=−0.56. Mann-Whitney p=3.2e-15. In-sample drift→error ROC-AUC 0.66 (coefficients: insertion −1.41, deletion +1.80) | In-sample. The comparison is confounded by confidence |
| L-04 | `results/figures/fig1–4.png` | Figures generated from L-02 | Illustrative only |
| L-05 | `results/simulator_check/mel_*.png` | Visual QC grids for the simulator | QC evidence only |
| L-06 | `cache/*.npy` (7 arrays, 256 px) | Metadata and pixel spot checks passed, according to `audit/STAGE2_REPORT.md` [UNV in this review] | Double-resize pathway (F-10) |

### 2.2 Software smoke runs (not scientific evidence)

| ID | Run directory | Configuration [OBS] | Status |
|---|---|---|---|
| S-01 | `runs/smoke/phase1-20260920` | Smoke mode; 3 architectures × 2 strategies; seed 0 | 44/44 stages reported complete [UNV] |
| S-02 | `runs/smoke/phase1-final-20260920` | `{'smoke': True, 'archs': ['resnet50'], 'strats': ['standard'], 'seed': 0, 'fairness': True}`. Nine step directories are present | 9/9 stages reported complete [UNV] |

Smoke profile: 1 epoch, a 300-image training subset, 8 deletion/insertion steps. These runs show that the software path executes from start to finish. **They contain no scientific evidence, and no number from them may appear in the thesis.**

### 2.3 Audit and protocol work (2026-09-20)

Useful work already exists:

- record-level manifests;
- label-coverage checks;
- a duplicate audit;
- a cache pixel audit;
- a replacement power model, whose 9 tests reportedly pass;
- a proposed endpoint table (`protocol/ENDPOINTS.md`);
- a decision package covering D-03 to D-11.

Much of this is reusable. Several statements in it are too strong, however (see §5).

---

## 3. Reuse / correct / unverified

| Component | Decision | Reason |
|---|---|---|
| `labels.py` label maps, which fail closed | **Reuse** | Coverage audited; `melanosis` ≠ melanoma is encoded correctly |
| `audit/manifests/*_v1.csv` | **Reuse as input** for the research_v1 manifests, after an adapter adds the missing fields | 14,991 rows; no personal attributes |
| Lesion-level HAM split (`data.ham10000`) | **Reuse the logic, but decouple the seed** | F-08 |
| `eval/metrics.py` sensitivity, specificity, AUROC and DeLong | **Reuse after unit tests** against a reference implementation | The DeLong midrank implementation has not been independently tested in this repository |
| `metrics.mcnemar` | **Correct the endpoint wiring** | The function is correct, but it was called on all cases (F-01) |
| `metrics.bootstrap_ci` | **Rewrite** as a cluster-aware bootstrap | It resamples rows only |
| `metrics.ece` | **Reuse**, renamed as a descriptive calibration metric | No calibration method is applied (F-14) |
| `eval/xai.GradCAM` | **Reuse after adding checks** | All-zero maps are not handled (F-13) |
| `eval/xai.deletion_insertion` | **Rewrite** as v1 | It uses an arithmetic mean, does not persist curves, and has conditional pixel-coverage and tie-ordering problems (F-06, F-13) |
| `eval/xai.localisation` | **Reuse after a guard** for constant maps | — |
| `msim/simulator.py` factor functions | **Reuse the image operations; rewrite seeding and the geometry interface** | F-05, F-09 |
| `train.py` | **Rewrite** the training loop | No checkpoint/resume support, seeds are coupled, preprocessing is implicit |
| `run_eval.py`, `rq4_test.py`, `analyse.py` | **Rewrite** as per-image producers plus a separate analysis layer | F-02, F-04 |
| `campaign.py` attempt/receipt design | **Reuse the design pattern**, in simplified form | It is sound in principle. It does not set `PYTHONHASHSEED` |
| `protocol/power_analysis.py` | **Reuse, but reframe** | The model is coherent, but its inputs are post hoc (F-12) |
| `cmex_app.py` | **Rewrite** as a viewer of verified outputs | F-11 |
| `fairness.py` | **Replace** | Only melanoma-only descriptive output is kept |
| Cache (`cache/`) | **Keep as-is, read-only** | Its use must be recorded explicitly (F-10) |

Unverified items that remain open:

- whether the raw or the cached pathway produced *each* legacy number;
- whether reported smoke stage counts match the receipts. This review did not open the receipts;
- the arithmetic of the pixel-audit JSON;
- the reported 11/11 and 9/9 test passes. These tests were not re-run, by design.

---

## 4. Prioritized findings

Priority: **P1** blocks a primary claim · **P2** blocks a secondary claim or reproducibility · **P3** affects presentation or documentation.

| ID | P | Finding | Type | Exact location |
|---|---|---|---|---|
| F-01 | P1 | The legacy McNemar test measures overall correctness on all 1011 pairs. It is not a test of melanoma sensitivity. Overall correctness *improves* (b=139 < c=183) while melanoma true positives fall (239→202, derived in the audit) | OBS | `code/run_eval.py:291` (`mcnemar(yd, pd_, pc, thr)` on all cases); `results/derm7pt_paired.json` |
| F-02 | P1 | No per-image predictions, identifiers or perturbation curves were saved. The melanoma-only discordance cannot be recovered, so inference must be re-run | OBS | `run_eval.py:285-308, 386-397`; `rq4_test.py:115-118` |
| F-03 | P1 | The Derm7pt cohort includes 9 same-file pairs (3 melanomas). The different-image cohort is 1002 pairs / 249 melanomas | OBS (audit) / UNV here | `audit/cohort_verification_20260920.json`; `code/data.py:81-101` has no exclusion |
| F-04 | P1 | The RQ4 insertion/deletion scores integrate the raw melanoma probability. Grad-CAM targets the melanoma logit for every case, and no null or control maps exist. Deletion *improves* (0.127→0.103) while insertion worsens (0.330→0.198) | OBS | `eval/xai.py:228-229, 289-290`; `results/xai_*.json` |
| F-05 | P1/P2 | The simulator seeds its generator with `hash((seed, factor, h))`. The tuple contains a `str`, so the value changes between processes unless `PYTHONHASHSEED` is fixed. `campaign.py` does not set it. Illumination sign, scale offset and noise, and glare/hair placement are therefore not reproducible across processes | OBS (two-process check) + CODE | `code/msim/simulator.py:335`; `code/campaign.py:168-171` |
| F-06 | P2 | The legacy insertion/deletion "AUC" is `torch.stack(curve).mean()`, an unweighted mean of steps 0..S. It is not trapezoidal integration. The bounded gap is about 0.01–0.03 | OBS | `eval/xai.py:289-290`; `protocol/metric_versions.json` |
| F-07 | P2 | Geometry-changing perturbations (`scale`) change only the image. The mask is loaded afterwards without being transformed, so localisation under `scale` compares the saliency map against a stale mask | CODE | `data.py:212-215` (shift applied to `img` only; `m` untouched); `run_eval.py:429-432` computes IoU for all factors |
| F-08 | P2 | `training_seed` and `split_seed` are one variable (`--seed`). Changing the training seed would change the split | OBS | `train.py:90-98`; `run_eval.py:318, 422` |
| F-09 | P2 | Albumentations 2.0.8 has its own RNG and is not seeded. `torch.manual_seed` and `np.random.seed` do not control it. `num_workers` > 0 adds further nondeterminism | CODE | `data.py:146-155`; `train.py:29-48, 90-91` |
| F-10 | P2 | The raw and cached pathways differ: the cache performs a double resize (256 INTER_AREA → 224), with a reported max difference of 1.93 in normalised units. The pathway is chosen silently from `cache_exists()` and is never recorded | OBS (code) + UNV (magnitude) | `run_eval.py:245-251`; `train.py:111`; `rq4_test.py:29-33` |
| F-11 | P2 | The demo CMRS uses the ratio of one image's probabilities. The report CMRS uses the ratio of dataset sensitivities. These are different quantities under the same name. The demo also labels negatives "benign" and uses `cv2.resize` preprocessing, which differs from the pipeline | OBS | `cmex_app.py:74-85, 44-47, 118, 152`; `analyse.py:240-284` |
| F-12 | P2 | The replacement power model is coherent, but it plugs in the *observed* test sensitivity (p1=0.9484) and assumed p01 scenarios. Its outputs are conditional planning figures on data that has already been seen, not guarantees | OBS | `protocol/POWER.md`; `protocol/power_analysis.py:169-179` |
| F-13 | P2 | An all-zero Grad-CAM map normalises to all zeros (`/(0+1e-8)`). `argsort` then orders the pixels arbitrarily (sorting is not stable), so deletion/insertion on such a map is undefined, and `localisation` counts every pixel as salient. There are no counters for such maps. Pixel coverage is complete only when H·W is divisible by `steps` (true for 224², steps 32; false for, e.g., 300²) | CODE | `eval/xai.py:230-238, 271-283, 316-318` |
| F-14 | P2 | No probability calibration is performed, although the documents say "calibrated probabilities". ECE is reported on raw sigmoid outputs, and clinical ECE (0.087) is *lower* than dermoscopy ECE (0.193) | OBS | `train.py` (no calibration step); `results/real_*.json` |
| F-15 | P2 | Drift→error is fitted and scored on the same 400 cases. Its features need both the dermoscopic and the clinical image of the same lesion, which would not exist in a smartphone-only deployment | OBS + CODE | `rq4_test.py:98-113` (`ind - inc`, `delc - deld`) |
| F-16 | P2 | The RQ4 Derm7pt subset is selected by `stratified_subset(..., seed=0)` on all 1011 pairs, including the 9 same-file pairs | CODE | `run_eval.py:414`; `rq4_test.py:64` |
| F-17 | P3 | The simulator docstring says severities are "calibrated" by `calibrate.py`. No such file exists anywhere in the project | OBS | `msim/simulator.py:15-18`; search result: absent |
| F-18 | P3 | The comment in `data.py` still cites the invalid power figure of 0.78 | OBS | `data.py:5-8` |
| F-19 | P3 | The DDI fairness code reports malignant-vs-benign as its main table, with only 7 melanomas per tone | OBS | `fairness.py:32-73` |
| F-20 | P3 | Streamlit is not installed | OBS | `find_spec('streamlit') is None`; `audit/requirements_observed.txt` |
| F-21 | P2 | Thresholds (`analyse.thresholds`) are linear interpolations on a 6-level grid, reported to two decimals. They have no uncertainty estimate, and the "first crossing" can depend on noise | OBS | `analyse.py:145-183`; `sweep_*.json` severities |
| F-22 | P3 | The operating threshold is taken from `roc_curve` as the first point with TPR ≥ 0.95. It is chosen on HAM validation images, not lesions (160 melanoma images) | OBS | `eval/metrics.py:32-41` |

---

## 5. Document / code / result inconsistencies

| # | Statement (source) | Evidence | Assessment |
|---|---|---|---|
| I-1 | "binary melanoma-versus-benign" (Research Plan §2.3, §3.3, §8.3; Technical Description; abstract) | Negatives include BCC/SCC/ACK; 78.7% of PAD negatives are malignant or premalignant (audit) | Incorrect task name |
| I-2 | "reproduces 95.3% → 75.3%" (README l.47; HANDOFF §2) | Ref [4] studies **non-melanoma** skin cancer and uses unpaired image sets (§6) | Invalid comparison |
| I-3 | "95.3%" (Research Plan Exec. Summary) | The PMC text returned 95.5% for dermoscopic sensitivity in the tool summary | **Unresolved; check against the PDF** |
| I-4 | "power above 0.80" (Research Plan §4.4); 0.78 (data.py, HANDOFF) | The legacy formula is invalid | Remove |
| I-5 | "calibrated probabilities" (Plan §4.2; Tech Desc.) | No calibration step exists (F-14) | Incorrect |
| I-6 | "Severities are calibrated against real smartphone statistics" (Plan §4.3; simulator docstring) | `calibrate.py` does not exist (F-17) | Unsupported |
| I-7 | "patient-level train/validation/test splits" (Tech Desc.; Plan Week 1) | HAM has no patient ID; the split is lesion-level | Overstated |
| I-8 | "Quantus", "MLflow", "Docker/CI", "Score-CAM, Grad-CAM++, ROAD" (Plan Ch. 6) | None of these is used in the code | Planned but not implemented |
| I-9 | ISIC 2019/2020, PH2 and Fitzpatrick17k listed as datasets | Not used. Fitzpatrick17k images are not downloaded | Out of scope |
| I-10 | "explanation faithfulness and localisation degrade at least as strongly as accuracy; drift is associated with misclassification" (IOCTe abstract, Results) | No controls; confidence confound; deletion moves in the opposite direction; drift is in-sample | **Unsupported at the time it was written**. See the correction in §7 |
| I-11 | "Explanations degrade MORE than accuracy… supports the study's hypothesis" (HANDOFF §2 RQ4) | Same as I-10 | Unsupported |
| I-12 | "no leakage of any kind exists" (STAGE2_REPORT summary) | Checks were only hash- and perceptual-duplicate based. HAM and Derm7pt have no patient IDs | Overstated |
| I-13 | "data-derived CMRS weights" / "validated 0–100 readiness metric" (Plan, Tech Desc.) | The code uses equal weights with 2 of 4 axes; nothing has been validated | Overstated |
| I-14 | "no ethics-board approval required" (Plan §2.3, §9.2) | This is an institutional determination, and no document records it | **Researcher must confirm with ITPU** |
| I-15 | "Abstract has been reviewed and approved by the scientific advisor… for submission" (Publication doc) | No evidence of approval or submission is in the repository | Cannot be verified here; do not restate as fact |
| I-16 | Timeline "8-week internship, June 2026"; abstract deadline 13 July 2026 | Today is 2026-09-27. The conference is 11–13 Nov 2026 | The status of the abstract and its promised "full results" is unknown |
| I-17 | AI declaration: Copilot only for boilerplate; "Simulator core and CMRS logic hand-written" (Plan Table 9.2) | Repository history (HANDOFF, README) shows AI assistants wrote substantial code, including the simulator and the analysis | **Declaration does not match history**. It must be rewritten by the researcher |
| I-18 | "Kaggle 30 h/week, 100–200 GPU-h" vs HANDOFF "166 h" vs "~5 h" | All are estimates, not measurements of the proposed design | Runtime is unknown until the pilot |
| I-19 | README "Localisation only measurable on simulated shift" | Under `scale` the masks are stale (F-07) | Qualify further |
| I-20 | "prospective" / "Week-2 decision gate passed" framing | Test outcomes were inspected before the protocol revision | Must be described as a retrospective revision |

---

## 6. Reference verification (this review)

| Ref | Claim in plan | Verification | Status |
|---|---|---|---|
| [4] Dascalu et al., J Cancer Res Clin Oncol 148(9):2497–2505, doi:10.1007/s00432-021-03809-x | Used as a melanoma replication target | PMC full text (https://pmc.ncbi.nlm.nih.gov/articles/PMC8453469/) reports **NMSC** detection with dermoscopy sens 95.5% (tool-extracted; the plan says 95.3%) vs smartphone 75.3%, AUROC 0.911 vs 0.821. Different image sets (165 vs 198 images) | **Verified: NOT melanoma.** The exact dermoscopy figure needs a check against the PDF |
| [17] doi:10.1002/ail2.77 | "lower faithfulness for melanoma than benign" | Search result: Bokadia H, Yang SC, Li Z, Folke T, Shafto P, "Evaluating perceptual and semantic interpretability of saliency methods: a case study of melanoma", Applied AI Letters (source: web search listing; full text not read) | Authors identified. **The specific claim is UNRESOLVED** |
| [5] Tjiu et al. 2025 | Pooled AUROC 0.90 vs 0.81 | Not checked | UNRESOLVED |
| [6] Chang et al. 2026 | 99% → 64–69% | Not checked; the date needs verification | UNRESOLVED |
| [9] Sangers et al. 2022 | iOS 91% vs Android 83% | Not checked | UNRESOLVED |
| All others | — | Not checked in Stage 1 | UNRESOLVED. Planned for backlog task T-02 |

---

## 7. Claim corrections with replacement wording

| # | Original | Proposed replacement |
|---|---|---|
| C-1 | "melanoma-versus-benign classification" | "binary melanoma-versus-non-melanoma classification. The negative class includes other malignant and premalignant lesions" |
| C-2 | "Derm7pt clinical (phone-like / smartphone)" | "Derm7pt clinical close-up photographs (acquisition device not verified as a smartphone)" |
| C-3 | "This closely reproduces the 95.3% → 75.3% reported in the literature" | Delete. If context is needed: "Dascalu et al. [4] reported lower performance on smartphone than dermoscopic images for **non-melanoma** skin cancer using unpaired image sets. Their figures are not directly comparable with the present melanoma task" |
| C-4 | "McNemar p = 0.016 shows significant degradation" | "An all-case McNemar test of overall correctness (legacy analysis) showed a difference in the direction favouring clinical images. It does not address melanoma sensitivity, which will be tested on melanoma cases only" |
| C-5 | "power above 0.80" / "power 0.78" | "A paired binary power model, conditional on assumed discordance scenarios and on a baseline sensitivity estimated from already-inspected data, indicates adequate power to detect a decline but limited ability to establish a decline larger than 10 percentage points. These are planning approximations, not guarantees" |
| C-6 | "insertion/deletion AUC" (legacy values) | "mean step score (v0)". New work: "trapezoidal area under the insertion/deletion curve (v1)" |
| C-7 | "Explanations degrade more than accuracy, worst for melanoma" | "In a legacy exploratory analysis without control maps, the mean insertion score fell on clinical images while the mean deletion score moved in the direction conventionally read as more faithful. Because both scores depend on model confidence, these results do not establish a change in explanation faithfulness" |
| C-8 | "explanation drift predicts misclassification" | "An in-sample association (ROC-AUC 0.66) was observed. Its out-of-sample value and its availability at deployment are untested" |
| C-9 | "calibrated probabilities / calibrated severities" | "uncalibrated sigmoid outputs (ECE reported descriptively)"; "severity levels are parameter choices of the simulator and are not calibrated to real smartphone captures" |
| C-10 | "patient-level splits" | "lesion-level splits for HAM10000 (no patient identifier available); patient-level grouping for PAD-UFES-20" |
| C-11 | "no leakage of any kind" | "no exact or verified near-duplicate images were found within or across the datasets. Patient-level overlap cannot be assessed for HAM10000 and Derm7pt" |
| C-12 | "CMRS: a validated deployment-readiness score" | "CMRS: an exploratory, equally weighted summary of retention ratios, reported next to its raw components. It is not a readiness or safety certification" |
| C-13 | "DDI fairness breakdown" | "descriptive melanoma counts and metrics by skin-tone group (7 melanomas per group); no inferential fairness conclusion" |
| C-14 | "protocol pre-registered / prospective" | "protocol revised retrospectively after legacy test results had been inspected" |
| C-15 | Abstract Results sentence | See `05_…` §5. Replace it with a statement limited to what the v1 results show, once they exist |

---

## 8. Summary of priorities

1. Re-run inference with per-image persistence and melanoma-only endpoints (F-01, F-02, F-03).
2. Build a controlled XAI evaluation with null maps and a confidence-adjusted analysis (F-04, F-06, F-13).
3. Fix reproducibility: explicit seeds including the hash seed and Albumentations, a recorded preprocessing pathway, decoupled split and training seeds (F-05, F-08, F-09, F-10).
4. Transform masks with geometric perturbations, or exclude localisation for geometric factors (F-07).
5. Correct every document statement listed in §5 and §7 before any external text is reused.
