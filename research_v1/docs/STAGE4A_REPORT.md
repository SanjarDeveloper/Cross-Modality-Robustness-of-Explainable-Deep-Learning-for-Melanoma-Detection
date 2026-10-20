# Stage 4A Report — Frozen v1 Threshold and Prediction Tables (ResNet-50, seed 0)

Date: 2026-10-01. All work in this stage used `research_v1/.venv_model`.

**Run:** `runs/predict_v1/20261001T183557Z_20b3af67_4959f943/`. Two receipts (threshold and predict) both verify.

**Stage records:** `runs/stage4a/`.

## Status

| Item | Result |
|---|---|
| Addendum G and config `predict_v1_resnet50_s0.json` (hash `20b3af67…`) frozen before inference | Done |
| Checkpoint `303d21e7…03dd`: file = sidecar = `best.json` (epoch 8) | Verified before both phases, and again afterwards |
| HAM VAL reproduction of the stored Stage 3A epoch-8 outputs | **Exact**: same 1529 IDs and labels, max \|Δp\| = 0.0, AUROC 0.8960 |
| **Threshold** | **t = 0.17569470405578613** (ESTIMATED), SHA-256 `469a4d45…9ffb` |
| Threshold written before any test or external inference | Yes. Written at 18:35:57Z by a separate process; HAM TEST and Derm7pt inference started at 18:36:05Z |
| Prediction rows | HAM TEST 1548/1548; Derm7pt ALL 2022/2022. **0 failures** |
| Verification checks (§3) | All pass |
| Tests | New `test_threshold.py` (6 tests). Full suite **141/141** |
| Protected inputs | 0 changed, 0 removed |

**Not done:** PAD, DDI, statistics, full XAI and training.

## 1. Fixed settings

| Item | Setting |
|---|---|
| Model | Seed-0 ResNet-50, Stage 3A epoch 8 (the only model used) |
| Preprocessing | `prep_v1_cached256` |
| Precision | `xai_precision_v1` |
| Output | One logit; p = sigmoid in float32, stored with `repr`. **No rounding** |
| Batch | **16**, the same as the Stage 3A validation, in manifest or validation-list order |

**Final partial batches:**

| Data | Images | Full batches | Final partial batch |
|---|---|---|---|
| HAM VAL | 1529 | 95 | 9 |
| HAM TEST | 1548 | 96 | 12 |
| Derm7pt dermoscopy | 1011 | 63 | 3 |
| Derm7pt clinical | 1011 | 63 | 3 |

## 2. Threshold (HAM VAL only; 02_rev2 §1.1)

**Rule**
- n₊ = 160, so k = ⌈95·160/100⌉ = **152**, computed with integer arithmetic.
- t = the 152nd-largest melanoma probability = **0.17569470405578613**.
- Predict melanoma iff p ≥ t.

**Ties and neighbours**
- Exactly one image (a melanoma) has a probability equal to t.
- The next higher melanoma score is 0.178624. Using it would give 151/160 < 0.95.
- The next lower melanoma score is 0.175506.

**Achieved on HAM VAL**
- Sensitivity **152/160 = 0.950**.
- Specificity **884/1369 = 0.646**.
- Lesion-level sensitivity, descriptive only: 86/88 lesions.

**Freezing**
- `threshold.json` identifies the checkpoint, its hash, the validation manifest, the validation list, the batching and the precision.
- It has a SHA-256 sidecar and was re-verified before use.

**Caveat:** HAM VAL was already used for checkpoint selection and has now been used for the threshold. Its numbers are **not** independent estimates.

## 3. Verification (`verification.json`)

**HAM TEST**
- 1548 unique rows, exactly the test split of the manifest.
- 170 melanoma.

**Derm7pt ALL**
- 2022 unique rows, covering the manifest exactly.
- Every image_uid and label agrees with the manifest join.
- 1011 pairs. Every pair has exactly one dermoscopy and one clinical row, with the same label and the same cohort flag.

**Primary cohort**
- Derived from the frozen `in_derm7pt_primary` flag: **1002 pairs, of which 249 are melanoma**.
- The ALL cohort has 252 melanoma pairs.
- The 498-pair XAI sample was **not** used as a population, and nothing was filtered by XAI eligibility.

**Same-file pairs**
- All 9 pairs keep both rows.
- In every pair the two cached images are byte-identical, and |Δp| = **0.0** (declared tolerance 1e-5), so the predictions are identical.

**Recompute and reload**
- Reloading the persisted CSVs reproduces the in-memory rows exactly.
- Recomputing the first batch of each cache gives max |Δp| = 0.0.
- The checkpoint and threshold hashes were unchanged at the end.

## 4. Preliminary descriptive tables (`descriptive_table.csv`; no inference)

Threshold t = 0.17569…; raw numerators and denominators.

| Data | Cohort | Modality | n (mel / non-mel) | TP | FN | TN | FP | Sensitivity | Specificity | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| HAM10000 TEST | test split | dermoscopy | 1548 (170/1378) | 157 | 13 | 911 | 467 | 157/170 = 0.924 | 911/1378 = 0.661 | 0.913 |
| Derm7pt | **PRIMARY** (1002 pairs) | dermoscopy | 1002 (249/753) | 214 | 35 | 381 | 372 | 214/249 = 0.859 | 381/753 = 0.506 | 0.777 |
| Derm7pt | **PRIMARY** | clinical | 1002 (249/753) | 106 | 143 | 588 | 165 | 106/249 = 0.426 | 588/753 = 0.781 | 0.668 |
| Derm7pt | ALL (1011; sensitivity analysis) | dermoscopy | 1011 (252/759) | 215 | 37 | 383 | 376 | 215/252 = 0.853 | 383/759 = 0.505 | 0.773 |
| Derm7pt | ALL | clinical | 1011 (252/759) | 107 | 145 | 590 | 169 | 107/252 = 0.425 | 590/759 = 0.777 | 0.665 |

**Paired counts for melanoma pairs (`paired_melanoma_counts.json`)**

| Cohort | Detected in both | Dermoscopy only | Clinical only | Missed in both | Pairs |
|---|---|---|---|---|---|
| **PRIMARY** | 97 | 117 | 9 | 26 | 249 |
| ALL | 98 | 117 | 9 | 28 | 252 |

These counts are consistent with the table: 97 + 117 = 214 TP in dermoscopy, and 97 + 9 = 106 TP in clinical.

**How to read these numbers**
- These are descriptive counts only. There are no tests, CIs, DeLong, bootstrap or Holm.
- AUROC is a ranking measure, not accuracy.
- The Derm7pt "clinical" images are not described as smartphone images, because there is no evidence for that.
- The difference between dermoscopy and clinical images is not interpreted causally. Image source, acquisition and case mix are confounded.

## 5. Data roles

| Data | Role |
|---|---|
| **HAM VAL** | Used for checkpoint selection and the threshold. Not independent |
| **HAM TEST** | First use for evaluation in v1 |
| **Derm7pt** | Evaluation only, but **not pristine**. It was inspected in legacy work and in the Stage 3B numerical XAI pilots (24 pairs, attribution settings only; no thresholds or weights were tuned there) |

## 6. Files

**Code**
- New: `src/rv1/threshold.py`, `src/rv1/predict_v1.py`, `tests/test_threshold.py`.
- Changed: `src/rv1/config.py`, which adds the purpose `predict_v1`. The prior copy is `runs/stage4a/config_before.py.txt`.

**Documents and config:** `docs/PROTOCOL_ADDENDUM_2026-10-01_G.md`, `configs/predict_v1_resnet50_s0.json`.

**Run outputs**
- `threshold.json` with its sidecar;
- `val_reproduction.json`, `ham_val_predictions.csv`;
- `predictions_ham_test.csv`, `predictions_derm7pt_all.csv`;
- `verification.json`;
- `descriptive_table.csv`, `paired_melanoma_counts.json`;
- the receipts.

Each prediction row includes the identifiers, cohort flags, logit, probability, threshold, prediction, checkpoint hash, threshold hash, preprocessing, precision, batch and status.

**Integrity**
- 553 files before and after, with 0 changed, 0 removed and 0 added.
- The addendum and config were written before the snapshot, and the run directory is new.
- The cache `.npy` files are unchanged.

## 7. Remaining statistical work (not started)

1. E1.1: exact McNemar on the 249 primary melanoma pairs (b = 117, c = 9), the single primary test of the RQ1 family.
2. Secondary estimands (specificity, AUROC differences), with the planned cluster/paired CIs.
3. A sensitivity analysis on the ALL cohort.
4. HAM TEST reporting with CIs.
5. PAD and DDI evaluation, seeds 1–2, the other architectures, and full XAI.

Each requires separate authorisation.

**Stage 4A is complete.**
