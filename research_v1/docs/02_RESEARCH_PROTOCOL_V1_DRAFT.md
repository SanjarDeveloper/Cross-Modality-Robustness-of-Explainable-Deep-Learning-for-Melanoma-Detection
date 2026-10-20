# 02 — Research Protocol v1 (DRAFT)

Status: **draft for the researcher's and supervisor's review.** This protocol is not frozen.
Markers used in this document:

- **[FACT]** verified in Stage 1 (see `01_…`);
- **[PROPOSED]** a design decision that needs approval;
- **[PILOT]** a value that the pilot must confirm.

---

## 0. Exposure history and the status of this protocol

**[FACT]** Before this protocol was written, a ResNet-50 model had already been trained and evaluated on:

- the HAM10000 test split;
- all 1011 Derm7pt pairs;
- all of PAD-UFES-20;
- a stratified subset of 400 Derm7pt pairs, used for the XAI analysis.

The aggregate results were read by the researcher and by earlier assistants. Endpoint definitions were then revised in response to those results: the McNemar endpoint, the treatment of the 10 pp threshold, and the population for the 9 same-image pairs.

Consequences:

1. This protocol is a **retrospective revision**. It is **not** a prospective pre-registration. It must not be described as preregistered or as confirmatory on untouched data.
2. "Primary" and "confirmatory" in this document mean *pre-specified before the v1 re-analysis*. They do not mean *blind to the data*.
3. The v1 re-run of ResNet-50 seed 0 replicates the legacy analysis under corrected definitions. The new models (seeds 1–2, other architectures) are **new fits**, but they are evaluated on test sets that have already been used.
4. The thesis must contain a "Protocol history and data exposure" subsection. It will state these facts and list every change from v0 to v1 (see §8).
5. HAM10000 contains no untouched held-out partition. No new external test set is available locally.

---

## 1. Common definitions

| Term | Definition |
|---|---|
| Task | Binary **melanoma vs non-melanoma**. Positive: labels mapped to melanoma by `labels.py` (includes "melanoma metastasis"; excludes "melanosis" and "melanoma-like lesion"). Negative: all other diagnoses, including BCC, SCC, AK and other malignancies |
| Training source | HAM10000 train split only **[FACT: D-00 approved]** |
| Split | Lesion-level HAM split with `split_seed = 0`, fractions train/val/test 0.70/0.15/0.15, **fixed for all models** **[PROPOSED]** |
| Operating threshold | Per model: the smallest threshold reaching ≥ 95% sensitivity on HAM **validation**. It is frozen for every downstream condition |
| Probability | Uncalibrated sigmoid output. ECE is reported only as a *descriptive* measure of miscalibration. No calibration step is applied **[PROPOSED; see D-12]** |
| Preprocessing | `prep_v1_cached256`: source → RGB → 256×256 INTER_AREA (cache) → 224×224 Albumentations `Resize` → ImageNet normalisation. It is recorded in every output row **[PROPOSED]** |
| Derm7pt primary cohort | **1002 different-image pairs / 249 melanomas**. Sensitivity analysis: all 1011 / 252 **[PROPOSED; counts FACT per audit]** |
| PAD-UFES-20 cohort | All 2298 images, 52 melanomas from 36 patients **[FACT]**. For RQ5 it is split into adaptation and evaluation patients (§6) |
| Clustering units | HAM: lesion. Derm7pt: case (pairs resampled together). PAD: patient. DDI: image **[FACT: identifiers available]** |
| Primary model | ResNet-50, `standard` strategy, training seeds {0, 1, 2}. The primary RQ1/RQ4 estimate uses **seed 0**, and seeds 1–2 are reported as reproducibility spread **[PROPOSED; D-05]** |

---

## 2. RQ1 — Real modality shift

**Question.** When the same lesions are imaged as clinical photographs instead of dermoscopic images (Derm7pt), does melanoma sensitivity decrease for a HAM-trained model? How large is the change? How does the model behave on real smartphone images (PAD-UFES-20)?

**Hypotheses (Derm7pt).**

- H1₀: P(detected under dermoscopy, missed under clinical) = P(missed under dermoscopy, detected under clinical) among melanoma pairs.
- H1₁: these two probabilities differ (two-sided).

| Item | Specification |
|---|---|
| Population | Derm7pt primary cohort, 249 melanoma pairs for E1.1; all 1002 pairs for E1.3/E1.4. PAD-UFES-20 all images, analysed separately |
| Unit | Derm7pt case (pair). PAD patient for uncertainty |
| Roles | Train: HAM train. Threshold and checkpoint selection: HAM val. Evaluation only: Derm7pt, PAD, HAM test |
| **E1.1 primary** | Paired melanoma sensitivity difference (dermoscopy − clinical) in pp. Test: exact McNemar on melanoma pairs only (b, c discordant detections) |
| E1.2 secondary (estimation) | The same difference, with a 95% case-level paired bootstrap CI (2000 resamples) and a Newcombe-style paired interval as a cross-check |
| E1.3 secondary | Paired AUROC difference on all pairs; DeLong test and bootstrap CI |
| E1.4 descriptive | Specificity change; all-case correctness McNemar reported with its decomposition; ECE per modality |
| E1.5 descriptive (PAD) | Sensitivity (52 melanomas), specificity, AUROC, with patient-cluster bootstrap CIs. Compared descriptively with HAM test. **No paired test**, because PAD confounds modality, population, site and device |
| Controls | Same model, threshold, preprocessing and cohort manifest for both modalities. The 9 same-file pairs are analysed separately as a sanity check: their predictions must be identical |
| Multiplicity | E1.1 is the only confirmatory test in the RQ1 family. E1.3 is Holm-adjusted together with E1.1 within the family. Everything else is estimation |
| Claim limits | Applies to this model, this cohort, and clinical close-up photography, **not verified smartphone capture**. The ">10 pp" question is not tested inferentially: the width of the CI is reported (conditional power is about 0.5; `protocol/POWER.md`) |
| Interpretation | **Positive:** p<0.05 with b>c: "sensitivity was lower on clinical images in this cohort", with the CI. **Null:** p≥0.05: "no difference detected; the CI [..] does not exclude a decline of X pp". **Inconclusive:** the CI spans both 0 and clinically relevant declines, and is reported as such. Direction b<c is reported as observed |

---

## 3. RQ2 — Controlled synthetic perturbations

**Question.** How do melanoma sensitivity and AUROC change as five parameterised perturbations are applied to HAM test dermoscopic images? These are blur, illumination/colour, JPEG compression, lesion scale and capture nuisance.

| Item | Specification |
|---|---|
| Population | HAM test: 1548 images / 170 melanomas / 91 melanoma lesions [FACT] |
| Unit | Image. Lesion-cluster bootstrap for CIs |
| Design | One factor at a time. Severity grid {0, 0.125, 0.25, …, 1.0} (9 levels) **[PROPOSED; PILOT for runtime]**. Perturbation seed = f(split_seed, image_id, factor), computed with a stable hash; it is independent of severity and process |
| Primary metric (E2.1) | Worst-case sensitivity loss across the grid, per factor, with a lesion-cluster bootstrap CI |
| Secondary | AUROC loss per level; the full severity–sensitivity curve; the first severity at which the point estimate falls ≥10 pp below severity 0 (descriptive marker, D-01), with a bootstrap distribution of the crossing severity |
| Controls | Severity 0 is the identity, and a test asserts it is bit-identical. A verification test checks monotone image statistics per factor. Geometric factor (`scale`): masks are transformed identically, or localisation is omitted for that factor |
| Statistics | Ranking stability: the proportion of bootstrap resamples in which each factor ranks first. No ANOVA or η² claims (the factors have no common physical severity scale) |
| Multiplicity | Descriptive family; no confirmatory tests |
| Claim limits | Results apply **only to the chosen parameterisation** (radius_max 7 px, quality_min 18, zoom_out_max 2.4, etc.). The severity axis is not calibrated to real phone images (`calibrate.py` does not exist). Cross-factor comparisons depend on arbitrary endpoint choices and must be worded as such |
| Interpretation | **Positive:** a factor produces a loss whose CI excludes 0 at some level. **Null:** CIs include 0 across all levels (within the sampled range). **Inconclusive:** overlapping ranking distributions mean no ranking is claimed |

---

## 4. RQ3 — Architecture and training-strategy dependence (exploratory)

**Question.** Are the RQ1/RQ2 patterns similar across ResNet-50, EfficientNet-B3 and DenseNet-121, and under standard versus heavy augmentation?

| Item | Specification |
|---|---|
| Matrix | 3 architectures × 2 strategies × 3 training seeds = 18 models, fixed split **[PROPOSED; PILOT confirms feasibility]** |
| Training | Identical epoch budget. Checkpoint selected by best HAM validation AUROC. Per-architecture batch size confirmed on the 8 GB GPU during the pilot |
| heavy_aug | A fixed Albumentations recipe, versioned `aug_heavy_v1`. It must not import the simulator |
| Evaluation | The same RQ1 and RQ2 pipeline. RQ4 XAI **only for the primary model** unless the pilot budget allows more (§5, backlog reductions) |
| Metrics | Per model: HAM AUROC; Derm7pt paired sensitivity drop and AUROC drop; PAD AUROC; RQ2 worst-case losses |
| Statistics | Per-seed values, mean and range per cell. **No significance tests between architectures.** An optional descriptive mixed model is permitted only if clearly labelled exploratory |
| Claim limits | Three seeds describe training variability on one split. They are not a population of models, and they add no patients |
| Interpretation | "Consistent direction across all 18 models" or "direction varies" is reported descriptively. No ranking claim unless the seed ranges fail to overlap, and even then it is stated as descriptive |

---

## 5. RQ4 — Explanations under shift (main evidence, together with RQ1)

**Question.** Under the Derm7pt shift, how do Grad-CAM and Integrated Gradients insertion/deletion curves change? Do the explainers beat null/control maps under both modalities? How much of any change is explained by the change in model confidence?

No directional hypothesis is assumed. Explanations may improve, worsen or stay the same.

| Item | Specification |
|---|---|
| Population | A fixed XAI manifest from the Derm7pt primary cohort. **All 249 melanoma pairs + 249 non-melanoma pairs** drawn with a fixed seed (enriched; always labelled as enriched) **[PROPOSED; PILOT runtime]**. Fallback: 150 + 150 |
| Simulated arm (secondary) | A fixed HAM test manifest (170 mel + 170 non-mel) at severity {0, 0.5, 1.0} per factor. Localisation only against correctly transformed masks |
| Model | Primary ResNet-50 seed 0. Seeds 1–2 on a 100-pair subset as a stability check **[PROPOSED]** |
| Explanation target | The **melanoma logit** for every image (fixed target, independent of the label and the prediction). Stated explicitly |
| Explainers | Grad-CAM (last conv stage, fp32, no autocast). Integrated Gradients (Captum or an equivalent hand implementation; baseline = the blurred image; 32 steps **[PILOT: completeness error < 5%]**) |
| Controls | (a) Uniform-random map, 5 fixed seeds. (b) Centre-Gaussian prior map. (c) Edge-detector (Sobel magnitude) map. (d) Explainer map from a **mismatched image** (a map from another case with the same label) |
| Perturbation curve | Blur baseline (fixed kernel). 32 steps. **Every pixel covered** (last chunk absorbs the remainder). Ties broken by a fixed secondary key (pixel index). Full curves persisted |
| Metrics (v1) | `ins_auc_v1`, `del_auc_v1`: trapezoidal area over the fraction of pixels revealed or removed, x ∈ [0,1]. **Normalised versions**: `ins_norm = (AUC − p_base)/(p_full − p_base)` and `del_norm = (p_full − AUC)/(p_full − p_base)`, computed where \|p_full − p_base\| ≥ 0.05. Other cases are reported as undefined and counted |
| **E4.1 primary** | Per image: Δ_explainer = metric(explainer) − metric(random control). Test on each modality: paired Wilcoxon, explainer vs random, for insertion and deletion |
| **E4.2 primary** | Shift effect on control-referenced scores: paired Wilcoxon of Δ_clinical − Δ_dermoscopy on the same pairs; Hodges–Lehmann estimate and a case-bootstrap CI |
| E4.3 confidence confounding | Regression of the per-pair change in normalised score on the change in p_full and in (p_full − p_base). Report raw, normalised and adjusted estimates as **separate estimands**. Stratified results for pairs where the prediction does not change class |
| E4.4 exploratory | Melanoma vs non-melanoma difference in E4.2. Described as a subgroup contrast, **not** an interaction test |
| E4.5 exploratory | Grad-CAM vs IG agreement: Spearman rank correlation of maps and top-10% overlap. Agreement is not correctness |
| E4.6 exploratory | Drift→error. Only **deployment-available** features (clinical image alone: p, entropy, saliency concentration, ins/del on the clinical image) versus a probability-only baseline. 5-fold case-grouped CV. Features that need the dermoscopic twin are reported separately as a mechanistic analysis, not a deployable signal |
| Correctness checks | All-zero or constant maps counted and reported. Per-image IG completeness error. Curve length = steps+1. Last deletion step = fully baselined image. Last insertion step = original image |
| Multiplicity | The RQ4 primary family = {E4.1 × 2 explainers × 2 curves × 2 modalities, E4.2 × 2 explainers × 2 curves}, Holm-adjusted within the family. E4.3–E4.6 are unadjusted and labelled exploratory |
| Claim limits | Faithfulness here means *how the model output responds to removing or inserting pixels in the order given by the map, with a blur baseline*. It does not measure clinical meaningfulness. With no masks on Derm7pt, no localisation claim is made about the real shift |
| Interpretation | **If an explainer does not beat the controls on dermoscopy**, no claim about its faithfulness under shift is made (pre-specified). **Positive E4.2:** "control-referenced insertion/deletion scores differed under the clinical modality", reported with the E4.3 result stating whether the difference persists after confidence adjustment. **Null:** "no change detected", with CIs. **Inconclusive:** raw and adjusted results disagree, or insertion and deletion disagree; both are reported without selecting one |

---

## 6. RQ5 — Limited smartphone adaptation and descriptive skin-tone analysis

**Question.** (a) Does limited fine-tuning of the pre-specified primary model (ResNet-50 standard seed 0) on PAD-UFES-20 adaptation patients change melanoma sensitivity and AUROC on separate PAD evaluation patients? (b) How do counts and metrics vary across skin-tone groups (descriptive only)?

| Item | Specification |
|---|---|
| PAD partition | Patient-level partition with fixed seed, stratified so that melanoma patients are split about 50/50. Roughly **18 melanoma patients** go to each side **[PROPOSED; exact counts computed in T-04 and frozen]**. Adaptation patients are further split into adapt-train / adapt-val by patient |
| Adaptation | Fine-tune the full network for a fixed small number of epochs (e.g. 5) at low learning rate (1e-5), with HAM-style standard augmentation. The threshold is re-selected on **adapt-val** (reported), and the original HAM threshold is also reported **[PROPOSED]** |
| Evaluation | PAD evaluation patients only: the base model vs the adapted model on identical images. Paired: exact McNemar on melanoma images (descriptive, given n≈26 images) and a patient-cluster bootstrap AUROC difference. Also Derm7pt clinical and HAM test, to check for forgetting |
| Skin tone | DDI: counts, melanoma sensitivity (7 melanomas per group) and AUROC with Wilson/bootstrap intervals, **descriptive only**. PAD: Fitzpatrick counts and metrics per group where the evaluation partition holds ≥ 5 melanomas; otherwise counts only. PAD has 804 images with missing tone and no melanomas in types 5–6 [FACT] |
| Multiplicity | None; everything is descriptive |
| Claim limits | One model, one adaptation recipe, very few melanoma patients. No claim about fairness, bias or equity. Malignant-vs-benign results are off-target (D-02) and are not reported as primary |
| Interpretation | **Positive:** improvement with CI excluding 0 on PAD evaluation patients, stated for this recipe only. **Null / inconclusive** is the expected outcome given the sample size, and is reported as "insufficient data to determine" |

---

## 7. CMRS (exploratory composite)

- **Components:** sensitivity retention (clinical/dermoscopy); AUROC retention; ECE change (descriptive); insertion-norm retention and deletion-norm retention for the primary explainer; localisation retention (simulated arm only).
- **Formula:** the equal-weighted mean of the available clipped retention ratios × 100, with `cmrs_version = v1`.
- **Missing axes** are never set to zero. Every CMRS value carries its **axis-set signature** (for example `{sens, auroc, ins, del}`). Values with different signatures are never placed in the same comparison or ranking.
- The raw components are always shown next to the composite. Colour bands are not used.
- **Claim limit:** CMRS is a descriptive summary. It is not a readiness score, a safety threshold or a clinical criterion.

---

## 8. Changes from v0 to v1 (to be reported in the thesis)

| # | v0 (legacy) | v1 |
|---|---|---|
| 1 | Derm7pt 1011/252 | Primary 1002/249; 1011 as sensitivity analysis |
| 2 | All-case McNemar | Melanoma-only McNemar (E1.1) |
| 3 | ">10% drop" inferential | Estimation with CI; 10 pp used as a descriptive marker only |
| 4 | Mean-of-steps "AUC" | Trapezoidal v1 plus normalised variants; curves persisted |
| 5 | No XAI controls | Four control maps; control-referenced primary endpoints |
| 6 | Image-level bootstrap | Cluster bootstrap by lesion, case or patient |
| 7 | In-sample drift model | Grouped CV; deployment-available features |
| 8 | Split and training seed coupled | Decoupled; split fixed |
| 9 | Process-dependent perturbation RNG | Stable-hash seeding |
| 10 | Masks not transformed | Masks transformed or axis dropped |
| 11 | Implicit preprocessing | `prep_v1_cached256` recorded |
| 12 | "benign" | "non-melanoma" |
| 13 | Demo CMRS = probability ratio | The demo only displays stored v1 report values |

---

## 9. Verified facts vs proposed decisions (summary)

| Verified facts | Proposed decisions |
|---|---|
| Dataset counts, missing patient IDs, 9 same-file pairs (from the audit) | Primary cohort 1002/249 |
| Legacy McNemar measures correctness | E1.1 melanoma-only |
| No per-image outputs exist | Re-run with per-image outputs |
| Legacy AUC is a mean | v1 trapezoid plus normalisation |
| Hash seed varies across processes | Stable-hash seeding |
| Masks are not transformed under `scale` | Joint geometry transform |
| No calibration step | ECE descriptive only |
| Test data already inspected | Retrospective-revision framing |
| — | 18-model matrix (subject to the pilot), XAI sample size, IG steps, RQ5 partition |
