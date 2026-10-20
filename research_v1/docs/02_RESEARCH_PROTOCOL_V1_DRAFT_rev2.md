# 02 — Research Protocol v1 (DRAFT, rev2)

Status: this is a draft for review by the researcher and the supervisor. It is not frozen. It supersedes `02_…_DRAFT.md`, which is preserved unchanged.

Tags used in this document:
- **[FACT]** — verified in Stage 1 or Stage 1.1.
- **[PROPOSED]** — a design decision that still needs approval.
- **[PILOT]** — a value that must be measured before it can be fixed.

---

## 0. Data-exposure history and the status of this protocol

**[FACT]** Before this protocol was written, the following had already been done:
- A ResNet-50 model was trained and evaluated on HAM10000 test, on all 1011 Derm7pt pairs, on all of PAD-UFES-20, and on a 400-pair Derm7pt XAI subset.
- The aggregate results were inspected.
- The endpoint definitions were then revised.

What this means for the protocol:
1. **This is a retrospective revision.** It is not a prospective preregistration, and it must never be described as one.
2. **"Primary" means pre-specified in this document before the v1 analyses are run.** Primary analyses are called **primary retrospective analyses**. This protocol does not use the word "confirmatory".
3. **The legacy model has already been evaluated on the test sets.** The v1 re-analysis of that model re-analyses data that has already been seen. New models (additional seeds and architectures) are new fits, but they are evaluated on the same test sets, which have already been used.
4. **The thesis must say this.** It needs a subsection titled "Protocol history and data exposure", together with the v0→v1 change table (§9).
5. **No untouched external test set is available locally.**

---

## 1. Common definitions

| Term | Definition |
|---|---|
| Task | Binary classification: **melanoma vs non-melanoma**. The positive class is defined by the `labels.py` mapping. The negative class includes other malignant and premalignant diagnoses. |
| Training source | HAM10000 train split only [FACT: D-00]. |
| Split | Lesion-level split of HAM10000 with `split_seed = 0`, reproducing the legacy assignment. Nominal fractions are train/val/test = 0.70/0.15/0.15, giving 6938/1529/1548 images [FACT]. The split is fixed for all models [PROPOSED]. |
| Probability | Uncalibrated sigmoid output of the melanoma logit, computed in fp32 in v1. No calibration step is applied. ECE is reported descriptively only. |
| Preprocessing (v1) | `prep_v1_cached256`: the legacy 256×256 INTER_AREA cache, then Albumentations `Resize(224)` (bilinear, the library default, to be recorded during T-A1), then ImageNet normalisation. Every output row records this label. |

### 1.1 Operating threshold (v1 rule) [PROPOSED]

The threshold is set per model, on the HAM **validation** split, and then frozen for every downstream condition.

- **Estimation unit.** The unit is the image, which is also the unit at which the classifier operates. The validation split contains n₊ = 160 melanoma images from 88 lesions [FACT, per audit]. At the chosen threshold, lesion-level sensitivity (a lesion counts as detected if any of its images is detected) is reported descriptively and does not change the rule.
- **Decision rule.** Predict melanoma if p ≥ t.
- **Choice of t.** Choose the highest t such that the validation image sensitivity is at least 0.95. Equivalently: sort the melanoma validation probabilities in descending order, let k = ⌈0.95·n₊⌉, and set t equal to the k-th largest melanoma probability. For n₊ = 160, k = 152.
- **Ties.** Because the rule uses p ≥ t, every image whose probability equals t is classified as positive, so ties can only raise sensitivity. The realised validation sensitivity and specificity are reported.
- **When the target cannot be estimated.** If n₊ < 20, or if the scores are degenerate (t cannot be computed, or t is not finite), the threshold is **not estimated**. The run is marked invalid for any threshold-based endpoint, and no fallback threshold is used. Threshold-free endpoints (AUROC) remain valid.
- **Precision.** With n₊ = 160, validation sensitivity is estimated in steps of 1/160. The resulting threshold is a sample estimate, and its variability is not propagated into the test CIs. This is stated as a limitation.

### 1.2 Cohorts [counts FACT per audit; roles PROPOSED]

| Cohort | Definition |
|---|---|
| Derm7pt primary | 1002 pairs with different image files; 249 melanomas |
| Derm7pt all | 1011 pairs, 252 melanomas. Used as a sensitivity analysis and for legacy reproduction |
| PAD-UFES-20 | 2298 images, 52 melanoma images, 36 melanoma patients, 1373 patients [FACT]. For RQ5, patients are partitioned as described in §6 |
| DDI | 656 images, 21 melanomas (7 per tone group) |

**Clustering units:** HAM uses the lesion, Derm7pt uses the case (paired), PAD uses the patient, and DDI uses the image.

**Primary model:** ResNet-50, standard training, training seed 0. Seeds 1 and 2 are used only to show seed spread.

### 1.3 Legacy reproduction vs protocol-v1 analyses

These are two separate activities. Their results are reported separately.

| | Legacy reproduction (LR) | Protocol-v1 analysis (V1) |
|---|---|---|
| Purpose | Check that the v1 infrastructure can regenerate legacy predictions under the historical settings, as far as those settings are known | Produce the study's results |
| Model | Legacy checkpoint | Legacy checkpoint for seed 0 (re-analysed under v1 rules), plus new fits |
| Cohort | Derm7pt all (1011), HAM test, PAD all | Protocol cohorts (§1.2) |
| Preprocessing | The best-evidenced historical pathway: cache, `Resize(224)`, fp16 autocast inference, legacy threshold 0.12517. Settings that are unknown are recorded as unknown. | `prep_v1_cached256`, fp32 inference, threshold from the §1.1 rule |
| Comparison target | Legacy aggregates (sensitivity counts, AUROC, ECE) | None. **v1 is never tuned to match legacy values.** |

**LR difference procedure.** A difference of ±0.005 is a *diagnostic tolerance*. It is not a correctness criterion. Sensitivity is compared as integer counts (for example 239/252), and an exact match is expected.

If any difference exceeds the tolerance, or any count differs:
1. Check that the cache order, labels and identifiers match.
2. Check whether the difference comes from fp16 vs fp32 inference, from batch composition, or from library versions.
3. Record every hypothesis tested and its outcome. The time box is 3 hours.
4. Report both values, the difference, and the explanation, or state "unexplained".

An unexplained difference in LR does **not** invalidate V1, which has its own validity tests (03_rev2 §14). It is reported in the thesis appendix.

---

## 2. Statistical conventions (apply to all RQs)

- **Bootstrap.** Percentile cluster bootstrap with B = 2000 and a fixed seed. Each replicate samples clusters with replacement (lesion, case or patient) and includes every row of each sampled cluster. For paired designs the cluster is the pair or patient, so both conditions move together. The estimand is recomputed on each replicate from the same definition used for the point estimate.
- **Undefined replicates.** Some replicates produce an undefined statistic, for example a resample with no melanoma, or an AUROC with a single class. These are **counted and reported** as a proportion of B. They are excluded from the percentile calculation. If more than 5% of replicates are undefined, the CI is flagged as unreliable and a cluster-count table is reported instead.
- **Multiplicity.** Families and decision rules are stated separately for each RQ. Any p-value outside a stated family is reported unadjusted and labelled secondary or exploratory.
- **Interpretation of adjustment.** If an association shrinks after adjustment, the adjustment variable is *associated* with the difference. It is not shown to be the cause.

---

## 3. RQ1 — Real modality shift

**Question.** Using the primary model, does melanoma sensitivity on paired Derm7pt lesions differ between clinical photographs and dermoscopic images? How large is the difference? How does the model perform on PAD-UFES-20 smartphone images?

**Hypothesis.** H1₀: P(detected under dermoscopy and missed under clinical) = P(missed under dermoscopy and detected under clinical), among melanoma pairs. The test is two-sided.

| Item | Specification |
|---|---|
| Population | E1.1 and E1.2 use the 249 melanoma pairs. E1.3 and E1.4 use all 1002 primary pairs. The sensitivity analysis repeats everything on 1011 pairs. PAD is analysed separately |
| Roles | Training data: HAM train. Checkpoint selection and threshold: HAM val. Evaluation only: Derm7pt, PAD, HAM test |
| **Primary testing family (RQ1)** | **One test: E1.1, exact McNemar on melanoma pairs, α = 0.05, no adjustment.** Decision rule: reject H1₀ if p < 0.05. The direction is given by b vs c |
| E1.2 (estimation) | Paired sensitivity difference (dermoscopy − clinical) in percentage points. Estimand: the difference in proportions over melanoma pairs. 95% CI from a case-cluster bootstrap. 10 pp is a descriptive reference only and is not tested |
| E1.3 (secondary) | Paired AUROC difference on all 1002 pairs. DeLong p-value reported **unadjusted, labelled secondary**, with a case-bootstrap CI. It is not part of the primary family |
| E1.4 (descriptive) | Specificity change; the all-case correctness McNemar together with its decomposition; ECE by modality |
| E1.5 (descriptive, PAD) | Sensitivity, specificity and AUROC with patient-cluster bootstrap CIs. No paired inference, because site, device and population are all confounded |
| Checks | The 9 same-file pairs must produce identical predictions under both modalities. This is a pipeline check and is not an endpoint |
| Seeds | Primary analysis uses seed 0. For seeds 1 and 2, E1.1 and E1.2 are reported as spread only |
| Claim limits | Results apply to this model and this cohort. The "clinical" images are clinical close-up photographs and are not verified smartphone captures |
| Interpretation | **Positive:** p < 0.05, reported with the direction and the CI. **Null:** p ≥ 0.05, reported with the CI, stating which magnitudes remain compatible with the data. **Inconclusive:** the CI includes both 0 and differences that would matter, and this is stated explicitly. No outcome is assumed in advance |

---

## 4. RQ2 — Controlled synthetic perturbations (descriptive)

| Item | Specification |
|---|---|
| Population | HAM test: 1548 images, 170 melanomas, 91 melanoma lesions. Cluster: lesion |
| Design | One perturbation factor at a time: blur, illumination/colour, JPEG compression, lesion scale, capture nuisance. Severity grid {0, 0.125, …, 1} [PILOT]. `perturb_seed` = stable hash of (split_seed, image_uid, factor) |
| Estimands | Per factor and level: the change in sensitivity from severity 0 and the change in AUROC, each with a lesion-bootstrap CI. Worst-case loss = the maximum loss over levels, with its own bootstrap distribution (the maximum is recomputed in each replicate) |
| Crossing marker | The first severity at which the sensitivity point estimate falls ≥ 10 pp below severity 0, found by linear interpolation. Bootstrap: the crossing is recomputed in each replicate. The report gives **the proportion of replicates with no crossing** and quantiles of the crossing severity, with no crossing coded as ">1.0" (censored). This means an upper bound can be ">1.0" |
| Ranking | The proportion of replicates in which each factor has the largest worst-case loss. No ANOVA and no η² |
| Multiplicity | None. The whole RQ is descriptive |
| Claim limits | Results apply only to the chosen parameterisation (radius_max 7 px, JPEG quality_min 18, zoom_out_max 2.4, and so on). The simulator is not calibrated to real smartphone images |

---

## 5. RQ3 — Architecture and training strategy (exploratory)

| Item | Specification |
|---|---|
| Matrix | 3 architectures (ResNet-50, EfficientNet-B3, DenseNet-121) × 2 strategies (standard, heavy augmentation) × 3 training seeds, on one fixed split. Feasibility is set by the pilot, and reductions are listed in 04_rev2 |
| Evaluation | The same E1.1–E1.5 and RQ2 estimands, computed per model |
| Statistics | Per-seed values and the range. No significance testing between architectures |
| Claim limits | Seeds show training variability. They do not add patients |

---

## 6. RQ4 — Explanation behaviour under shift

**No directional hypothesis is assumed.**

### 6.1 Set-up

| Item | Specification |
|---|---|
| Model | Primary model (ResNet-50, seed 0). Seeds 1–2 on a 100-pair subset only if the budget allows |
| XAI manifest | Fixed sample from the Derm7pt primary cohort: all 249 melanoma pairs plus 249 non-melanoma pairs, drawn with a fixed seed. The sample is enriched and always labelled as such [PILOT fallback: 150 + 150] |
| Target | The melanoma logit, for every image |
| Explainers | Grad-CAM and Integrated Gradients (IG) |
| **Attribution interpretation (primary)** | **Positive, melanoma-supporting evidence** for both explainers. Grad-CAM: ReLU of the channel-weighted activations, as in standard Grad-CAM. IG: attributions are summed over the RGB channels (signed sum), then ReLU is applied. Neither explainer receives extra smoothing in the primary analysis |
| Secondary attribution | IG with absolute attribution (sum over channels of \|attr\|). IG positive attribution with Gaussian smoothing (σ = 2 px). Both are reported separately and are never pooled with the primary analysis |
| Controls | (a) **Random**: uniform random maps from 5 fixed seeds. (b) Centre Gaussian prior. (c) Sobel edge magnitude. (d) Mismatched map: the map of another case with the same label and the same modality, assigned by a fixed derangement |
| Curves | Blurred baseline. S = 32 steps. Every pixel is covered exactly once. Ties are broken deterministically by pixel index. Every curve is stored |
| Raw metrics (v1) | `ins_auc_v1` and `del_auc_v1`: the trapezoidal area over the fraction of pixels, x ∈ [0, 1] |

### 6.2 Control aggregation
The 5 random-control seeds are **averaged within each image**, giving one control value per image and metric: `ctrl_rand = mean over k of metric(random_k)`. The seeds are never treated as separate observations. Controls (b)–(d) each give one value per image.

### 6.3 Primary estimands (raw scores are primary)
Quality-oriented, control-referenced contrasts per image, where a positive value means "better than the random control":
- Insertion: **Δ_ins = ins(explainer) − ins(ctrl_rand)** (higher insertion is better).
- Deletion: **Δ_del = del(ctrl_rand) − del(explainer)** (lower deletion is better).

**E4.1**, for each explainer ∈ {Grad-CAM, IG}, each curve ∈ {ins, del} and each modality ∈ {dermoscopy, clinical}: the median of Δ over the analysis population. Test: Wilcoxon signed-rank against 0. Estimate: Hodges–Lehmann, with a case-bootstrap CI.

**E4.2**, for each explainer and curve: the shift contrast per pair, **D = Δ_clinical − Δ_dermoscopy**. Test: Wilcoxon signed-rank against 0. Estimate: Hodges–Lehmann, with a case-bootstrap CI.

**Primary testing family (RQ4):** E4.1 (2 × 2 × 2 = 8 tests) plus E4.2 (2 × 2 = 4 tests) = **12 tests, Holm-adjusted at family-wise α = 0.05.** Decision rule: a hypothesis is rejected if its Holm-adjusted p < 0.05.

**Pre-specified gate:** if an explainer does not show Δ > 0 (Holm-rejected, positive direction) on dermoscopy for a given curve, no statement is made about that explainer's faithfulness under shift for that curve. E4.2 is still reported, but only descriptively.

### 6.4 Analysis population and invalid maps
- An image's map for a given explainer is **invalid** if it is constant (max − min < 1e-12 after the primary transformation), non-finite, or, for IG, if it fails completeness (§6.6).
- The **primary analysis population** for each explainer is the set of **matched eligible pairs**: pairs in which that explainer's map is valid in **both** modalities. Observations are never dropped from one modality independently of the other.
- E4.1 and E4.2 use the same matched population for a given explainer, so the paired contrasts are consistent.
- **Invalid-map rates are reported** by explainer × modality × class, and it is stated which rule caused each exclusion. For example, IG positive maps may be all-zero when there is little melanoma-supporting evidence. This is more likely for non-melanoma images, so exclusion is **selection by model evidence**. It is described in the thesis as a selection effect.
- **Sensitivity analysis:** all pairs are included, and invalid maps are processed "as computed" (the deterministic pixel-index order is used for constant maps). Results are compared with the primary analysis. Primary results are described as applying to eligible pairs only.

### 6.5 Normalised scores (secondary) and confidence
- Let d = p_full − p_base, where p_full is the probability on the original image and p_base the probability on the blurred baseline.
- **Normalisation is defined only when d ≥ δ = 0.05** (the image is more melanoma-like than its blurred baseline by at least 5 percentage points). In that case:
  - `ins_norm = (ins_auc − p_base) / d`
  - `del_norm = (p_full − del_auc) / d`
  - Neither is clipped, so values outside [0, 1] are kept.
- If **d ≤ −δ**, the image is labelled "baseline more melanoma-like". It is not normalised and is reported as its own category.
- If **|d| < δ**, the score is undefined (near-zero denominator).
- **Rationale for δ.** δ = 0.05 caps the amplification of noise at 20×. Sensitivity analyses use δ ∈ {0.02, 0.10}.
- Missingness is reported by modality × class × explainer. The normalised analysis uses matched pairs in which both modalities are defined, and its population is reported next to the raw primary analysis.
- **Normalisation does not remove confounding by confidence.** It rescales by the probability range, but it does not account for changes in the shape of the curve.
- **E4.3 (exploratory):** a linear model of D on the change in p_full and the change in d, reported alongside the unadjusted estimate. A smaller adjusted estimate is described as "associated with the confidence change". It is not described as caused by it.

### 6.6 IG validity
- Baseline: the same blurred image. Integration: Riemann midpoint rule with m steps.
- Absolute completeness error: e_abs = |Σ attr − (logit(x) − logit(b))|.
- Relative completeness error: e_rel = e_abs / |logit(x) − logit(b)|. It is computed only when the denominator is ≥ 0.1 logit. Otherwise only e_abs is reported.
- **Per-image failure** occurs if (e_rel is defined and > 0.05) or (e_rel is undefined and e_abs > 0.05).
- **Convergence (pilot):** m ∈ {16, 32, 64, 128} on 40 pairs. Choose the smallest m for which the failure rate is ≤ 5%, the 95th percentile of e_rel is ≤ 0.20, and the median primary metric changes by < 0.005 relative to 2m.
- The failure rate is reported in every run.

### 6.7 Exploratory analyses
- **E4.4:** subgroup contrast of D for melanoma vs non-melanoma. This is not a formal interaction test.
- **E4.5:** agreement between Grad-CAM and IG (Spearman correlation, top-10% overlap).
- **E4.6:** does a model built on deployment-available features computed from the clinical image alone improve error prediction over a probability-only baseline? Method: 5-fold cross-validation grouped by case. Features that need the twin image are reported as mechanistic only.
- **Simulated arm:** HAM (170 melanomas + 170 non-melanomas), severities {0, 0.5, 1}. Localisation uses masks transformed with the image.

### 6.8 Claim limits
"Faithfulness" here means how the model's output responds when pixels are revealed or removed in map order, starting from a blurred baseline. It does not measure clinical meaning. No localisation claim is made on Derm7pt.

---

## 7. RQ5 — Limited smartphone adaptation and descriptive skin-tone analysis

### 7.1 Frozen PAD patient partition [PROPOSED; computed and frozen in the next task]
- **Patient stratum**, determined from all of the patient's images in priority order:
  - S1: any melanoma image;
  - S2: otherwise, any image with BCC, SCC or ACK;
  - S3: otherwise.
- Patients with multiple diagnoses are placed by this priority rule [FACT: 179 multi-diagnosis patients, and 11 of the 36 melanoma patients have other diagnoses too].
- **Assignment.** Within each stratum, patients are sorted by `stable_hash(split_seed = 0, patient_id)`. The first ⌈0.50·n⌉ go to **eval**, the next ⌈0.35·n⌉ go to **adapt-train**, and the remainder go to **adapt-val**. For S1 (36 patients) this gives 18 / 13 / 5.
- **The assignment is frozen as a manifest, with its hash.** It is never changed after any adapted model has been evaluated.

### 7.2 Adaptation
- Starting point: the primary model.
- Training: all layers, AdamW with learning rate 1e-5 and weight decay 1e-4, standard augmentation, E = 5 epochs with a cosine schedule over the 5 epochs, training seed 0.
- **Selection:** after each epoch, compute the image-level AUROC on adapt-val. The selected checkpoint is the epoch with the highest AUROC. Ties are broken by the earlier epoch. If AUROC cannot be computed, the epoch-5 checkpoint is used and this is recorded.

### 7.3 Evaluation (identical evaluation patients for both models)
- **Primary contrast (descriptive, pre-specified):** AUROC(adapted) − AUROC(base) on PAD eval patients. Estimate with a 95% CI from a **patient-cluster paired bootstrap** (each patient's images are resampled together, and both models are scored on the same replicate). No image-level McNemar is used, because PAD patients contribute repeated images.
- **Secondary A (fixed original rule):** each model's threshold comes from the §1.1 rule on HAM val. For the adapted model the rule is recomputed on HAM val. Sensitivity and specificity differences are reported with patient-cluster paired bootstrap CIs.
- **Secondary B (reselected threshold):** the adapted model's threshold is re-estimated with the §1.1 rule on the adapt-val melanoma images. If there are fewer than 20 such images, it is flagged as not estimable. Reported separately and never mixed with Secondary A.
- **Forgetting check:** Derm7pt clinical AUROC and HAM test AUROC before and after adaptation.

### 7.4 Skin tone (descriptive only)
- DDI: counts, and sensitivity and AUROC with Wilson or bootstrap intervals, per tone group.
- PAD eval: Fitzpatrick counts, with metrics only for groups that contain ≥ 5 melanoma images. Otherwise counts only.
- No claims about fairness, bias or equity. Malignant-vs-benign analysis is off-target (D-02).

### 7.5 Interpretation
Results apply to this adaptation recipe and this partition only. Every possible outcome (improvement, no detectable change, deterioration, or insufficient data) is reported with its CI. No outcome is predicted.

---

## 8. CMRS — deferred

The composite score is **deferred**. It will not be computed until each of the following has been fully specified and approved:
- the transform and quality direction of each component;
- how denominators are handled, including negative or undefined normalised XAI scores;
- the aggregation population of each component;
- the weights.

Clipped ratios of normalised XAI scores that may be negative are not treated as meaningful. **Component tables are required** (sensitivity, AUROC, ECE, E4 raw contrasts, localisation from the simulated arm only). Deferring CMRS does not block any RQ.

---

## 9. Changes from v0 to v1 (to be reported in the thesis)

| # | v0 | v1 |
|---|---|---|
| 1 | Derm7pt 1011/252 | Primary 1002/249; all 1011 as a sensitivity analysis |
| 2 | McNemar on all cases | E1.1: melanoma-only, one-test primary family |
| 3 | ">10 %" as an inferential claim | Estimation only; 10 pp as a descriptive marker |
| 4 | Threshold = first ROC point with TPR ≥ 0.95 | Explicit rule (§1.1): highest t, ties and estimability defined |
| 5 | Mean-of-steps "AUC" | Trapezoidal v1 curves, persisted |
| 6 | No controls | Controls with within-image aggregation; quality-oriented contrasts |
| 7 | Image-level bootstrap | Cluster bootstrap; undefined replicates reported |
| 8 | In-sample drift model | Grouped cross-validation with deployment-available features |
| 9 | Split and training seed coupled | Decoupled |
| 10 | Process-dependent perturbation RNG | Stable hash |
| 11 | Masks not transformed | Masks transformed with the image |
| 12 | Implicit preprocessing, fp16 | `prep_v1_cached256`, fp32 |
| 13 | "benign" | "non-melanoma" |
| 14 | CMRS 2-axis score, demo probability ratio | CMRS deferred; component tables |

---

## 10. Verified facts vs proposed decisions

| Verified | Proposed |
|---|---|
| Cohort counts, identifiers, same-file pairs (audit); PAD multi-diagnosis counts (Stage 1.1) | Cohort roles, the 1002/249 primary cohort, the RQ5 partition rule |
| The legacy McNemar test measures correctness; there are no per-image outputs; the legacy "AUC" is a mean | E1.1 family, per-image persistence, v1 metrics |
| The hash seed varies between processes; masks are not transformed; no calibration is applied | Stable seeding, joint geometry, ECE descriptive only |
| Test data has already been inspected | "Primary retrospective analysis" framing |
| — | Matrix size, XAI n, IG step count, δ (all PILOT or approval) |
