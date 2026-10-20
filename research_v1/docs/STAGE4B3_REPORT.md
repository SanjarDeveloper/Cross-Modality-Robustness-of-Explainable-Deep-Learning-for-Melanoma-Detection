# Stage 4B3 Report — E1.4 Correctness, Descriptive ECE, and HAM TEST Uncertainty

Date: 2026-10-01/02. This stage was a CPU analysis of the saved Stage 4A predictions only.

**Run:** `runs/analysis_desc_4b3/20261001T190117Z_f8f81799_9aa7fa10/`. Spec and analyze receipts verify.

**Stage records:** `runs/stage4b3/`.

## Status

| Item | Result |
|---|---|
| Specification frozen before computation | `analysis_spec.json` (separate process, hashed). It records the inputs (predictions, threshold, receipts, checkpoint, HAM manifest, legacy `metrics.py`), cohorts, formulas, seeds, quantiles and cluster units |
| Cross-checks derived from rows | Melanoma 117/9, non-melanoma 53/260, all-case 170/269 ✓. HAM TEST TP/FN/TN/FP = 157/13/911/467, n = 1548 ✓ |
| E1.4 correctness (PRIMARY) | Accuracy dermoscopy 59.4%, clinical 69.3%, Δ = −9.9 pp. Correctness McNemar p = 2.65 × 10⁻⁶ (**descriptive, unadjusted**) |
| ECE, positive class, 15 bins | Derm7pt dermoscopy 0.123; clinical 0.134; HAM TEST 0.152 (descriptive) |
| HAM TEST, lesion-cluster bootstrap | Sensitivity 92.4% [86.7, 97.3]; specificity 66.1% [63.3, 69.1]; AUROC 0.913 [0.884, 0.941] |
| Undefined replicates | 0 for every metric |
| Tests | New `test_desc_stats.py` (7 tests). Relevant suites (`desc_stats`, `rq1_stats`, `auroc_paired`) pass |
| Protected inputs | 0 changed, 0 removed |

**Why the full suite was not run:** no shared module changed.

**Output safety:** every output was JSON-validated and published through a temporary file and a non-overwriting rename.

## 1. Retrospective implementation decisions (newly specified here, not preregistered)

**ECE convention `ece_pos_v1`**
- The protocol says only that ECE is "descriptive by modality". The read-only legacy `code/eval/metrics.py` uses **positive-class** ECE with 15 equal-width bins, right-closed.
- v1 uses positive-class calibration (melanoma probability against the melanoma indicator), **not** top-label confidence.
- Bins: 15 equal-width bins with **exact rational edges b/15**, left-closed and right-open; the last bin [14/15, 1] includes 1.
- ECE = Σ (n_b/N)·|mean p_b − mean y_b|. Empty bins are reported with weight 0; none occurred.
- The bins were fixed before any score was seen. The boundary convention differs from the legacy code, and v1 is not tuned to legacy values.
- Note on exact edges: the floating-point number nearest to 1/15 lies just below the exact edge, so it falls in bin 0. This is tested.

**HAM TEST cluster**
- The cluster is the manifest `cluster_id`, i.e. the HAM10000 lesion_id (protocol §4: "Cluster: lesion"). Lesions are **not** patients.
- The same seed is used for all three metrics, so they share identical draws. Undefined replicates are counted per metric.

**Seed:** bootstrap seed 20261004 (config).

## 2. E1.4 — Derm7pt all-case correctness (descriptive)

| PRIMARY (1002 pairs) | Both correct | Dermoscopy only | Clinical only | Both incorrect | Accuracy, dermoscopy | Accuracy, clinical |
|---|---|---|---|---|---|---|
| All cases | 425 | **170** | **269** | 138 | 595/1002 = 59.4% | 694/1002 = 69.3% |
| Melanoma (249) | 97 | 117 | 9 | 26 | | |
| Non-melanoma (753) | 328 | 53 | 260 | 112 | | |

- **McNemar on correctness:** b = 170, c = 269, exact p = 2.65 × 10⁻⁶. This is descriptive and unadjusted.
- **ALL cohort** (sensitivity analysis): 428/170/269/144, with the same b and c, so the same p.
- **Decomposition:** the all-case counts equal the melanoma counts plus the non-melanoma counts (verified).

**Interpretation**
- Overall correctness mixes melanoma detection with non-melanoma specificity, and it depends on the class mix (24.9% melanoma).
- Clinical images received fewer positive calls. That gives more correct non-melanomas (260 vs 53) but far fewer detected melanomas (9 vs 117).
- The opposite direction of the accuracy difference therefore does **not** contradict or replace the primary result: **model performance on clinical images was lower for sensitivity and AUROC.**

## 3. Descriptive ECE and reliability (`reliability_bins.csv`, `ece_full.json`, PNGs)

| Population | N | ECE | Empty bins |
|---|---|---|---|
| Derm7pt PRIMARY, dermoscopy | 1002 | 0.1226 | 0 |
| Derm7pt PRIMARY, clinical | 1002 | 0.1344 | 0 |
| HAM TEST | 1548 | 0.1522 | 0 |
| Derm7pt ALL, dermoscopy (sensitivity) | 1011 | 0.1257 | 0 |
| Derm7pt ALL, clinical (sensitivity) | 1011 | 0.1365 | 0 |

**What the reliability tables show (descriptive)**
- On HAM TEST and Derm7pt dermoscopy, the mean predicted p is mostly **above** the observed melanoma fraction at intermediate and high p. This is consistent with training under `pos_weight` ≈ 7.9 and no calibration step.
- On Derm7pt clinical images, predictions are **below** the observed fraction in the low-p bins. For example, in bin 0 the mean p is 0.027 against an observed fraction of 0.157.

**How not to read them**
- The model is not called calibrated.
- No ECE test or calibration fitting was done.
- A lower ECE does **not** imply better melanoma detection.

**Plots:** `reliability_derm7pt_primary.png` and `reliability_ham_test.png` show the reliability points against the diagonal, with a count panel underneath.

## 4. HAM TEST uncertainty

**Join:** each prediction row was joined to the frozen manifest by `record_uid`. The mapping is one-to-one, with labels and the test split verified, and the set equals the manifest's test split exactly.

**Clusters:** 1548 images in **1155 lesions**. Melanoma images (170) fall in 91 lesions; no lesion has mixed labels; at most 4 images per lesion.

| Metric (image-level) | Estimate | 95% CI (lesion-cluster bootstrap, B = 2000) | Undefined |
|---|---|---|---|
| Sensitivity | 157/170 = 0.924 | 0.867–0.973 | 0 |
| Specificity | 911/1378 = 0.661 | 0.633–0.691 | 0 |
| AUROC | 0.913 | 0.884–0.941 | 0 |

**How the bootstrap works:** each replicate samples lesion IDs with replacement and includes every image of each sampled lesion, keeping repeats. The metrics stay at image level. Clustering affects only the uncertainty, not how the point estimate is weighted.

## 5. Tests (`tests/test_desc_stats.py`, 7 tests)

| Area | What is checked |
|---|---|
| Correctness decomposition | All cases = melanoma + non-melanoma |
| Modality swap | Swaps the discordant cells, reverses Δ and keeps p |
| ECE bin boundaries | 0 → bin 0, 1 → bin 14; exact-edge behaviour around 1/15 and 0.2 |
| ECE hand calculation | A hand-calculated example matches; empty bins have weight 0 and a None mean |
| Invalid inputs | Probabilities outside [0, 1] or non-finite, bad labels and empty input are rejected |
| Lesion bootstrap | Includes every image of each drawn lesion with the correct multiplicity; reproducible under a fixed seed; no-melanoma replicates counted and not redrawn |

## 6. Limitations

- A fixed seed-0 model and a frozen threshold; the CIs are conditional on both.
- Derm7pt is previously exposed (legacy work and the Stage 3B numerical XAI pilot), and its patient-level dependence is unknown.
- HAM VAL was used for selection.
- There is no causal modality interpretation, and the clinical images are not described as smartphone images.
- Results are not generalised beyond the evaluated setting.

## 7. Files

**New code and config:** `src/rv1/desc_stats.py`, `src/rv1/desc_4b3.py`, `tests/test_desc_stats.py`, `configs/analysis_desc_4b3_v1.json`.

**Run outputs:**
- the spec with its sidecar;
- `results.json`, `ece_full.json`, `reliability_bins.csv`;
- two reliability PNGs;
- `bootstrap_replicates_ham_test.npz`;
- `thesis_results_4b3.md`, which was written after the receipt and has its own sidecar;
- the receipts.

**Integrity:** 617 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged.

## 8. Remaining work (RQ1 is **not** complete)

- **PAD-UFES-20 (E1.5)** and **DDI**.
- **Seeds 1–2** and the other architectures.
- **Full XAI (RQ4)**, plus RQ2, RQ3 and RQ5 as planned.

**Stage 4B3 is complete.**
