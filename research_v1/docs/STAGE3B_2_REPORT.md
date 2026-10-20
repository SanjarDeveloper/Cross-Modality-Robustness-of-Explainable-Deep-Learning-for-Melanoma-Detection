# Stage 3B.2 Report — Fresh IG m = 512 vs m = 1024 Assessment

Date: 2026-10-01. All work in this stage used `research_v1/.venv_model`.

**Run:** `runs/xai_pilot_fresh/20261001T181809Z_694763bb_fd41514b/`. The receipt verifies.

**Stage records:** `runs/stage3b2/`.

## Status

| Item | Result |
|---|---|
| Addendum F and config `xai_pilot_stage3b2_v1.json` (hash `694763bb…`) frozen before computation | Done |
| Checkpoint `303d21e7…03dd` (file = sidecar = `best.json`) | Verified |
| Fresh m = 512 and m = 1024 for all 48 references, with **outer batch 48** (single batch, frozen order, no partial batch); IG `internal_batch` 16; curve `eval_batch` 11 | Done |
| Endpoint logits for completeness, compared between the two fresh calls | **Identical** (max \|Δ\| = 0.0 for logit(x) and logit(b); tolerance 1e-6) |
| Reuse check of the m-independent Stage 3B outputs (Grad-CAM maps, Sobel maps, baseline logits recomputed) | **Exact** (max \|Δ\| = 0.0). Stage 3B receipt and outputs verified, so the controls and Grad-CAM curves were reused |
| **Candidate m = 512 against m = 1024 (pooled rule)** | **PASS → m = 512 selected** for the seed-0 ResNet-50 on the unperturbed Derm7pt pipeline only |
| Stage 3B.1 failed recheck | Preserved unchanged, as evidence of batch-shape numerical differences |
| Protected inputs | 0 changed, 0 removed |

Only the fresh m = 512 and m = 1024 results were used in the comparison; no cached completeness values were used.

## 1. Assessment (rules unchanged: Addenda D5, E3, F4)

| Group | n | (a) failures | p95 e_rel (undefined) | Eligible at both m | Median \|ΔAUC\| insertion (max) | Median \|ΔAUC\| deletion (max) | Overall |
|---|---|---|---|---|---|---|---|
| **Pooled** | 48 | **1 (2.1%) ✓** | **0.046 ✓** (0) | **47/48** | **0.0024 ✓** (0.011) | **0.0007 ✓** (0.005) | **pass** |
| Dermoscopy | 24 | 1 | 0.046 | 23 | 0.0032 | 0.0011 | pass |
| Clinical | 24 | 0 | 0.046 | 24 | 0.0018 | 0.0005 | pass |
| Melanoma | 24 | 1 | 0.046 | 23 | 0.0020 | 0.0011 | pass |
| Non-melanoma | 24 | 0 | 0.043 | 24 | 0.0026 | 0.0005 | pass |
| Melanoma × dermoscopy | 12 | 1 (8.3%) | 0.094 | 11 | 0.0026 | 0.0018 | fail (a) |
| Melanoma × clinical | 12 | 0 | 0.045 | 12 | 0.0015 | 0.0006 | pass |
| Non-melanoma × dermoscopy | 12 | 0 | 0.039 | 12 | 0.0036 | 0.0008 | pass |
| Non-melanoma × clinical | 12 | 0 | 0.040 | 12 | 0.0022 | 0.0003 | pass |

The subgroup rows are descriptive. The decision uses the pooled row. The melanoma × dermoscopy subgroup fails (a) because one failure in 12 references is above 5%.

**Excluded from (c): 1 reference.**
- case812 dermoscopy failed completeness at both 512 (e_rel 0.152) and 1024 (e_rel 0.064).
- It stays in the completeness denominator.

**Map and curve checks at both m:**
- 0 e_rel undefined;
- 0 all-zero, constant or non-finite maps;
- 0 endpoint failures.

Per-image ΔAUC values are in `per_image_512_vs_1024.csv`. The largest single-image |Δ| is 0.011 for insertion and 0.005 for deletion.

**Scope of the decision**
- **m = 512 is frozen** only for the seed-0 ResNet-50 (Stage 3A epoch 8) on the unperturbed Derm7pt pipeline, with the outer batch specified.
- This does **not** prove completeness for every production image. Every image keeps its own completeness and eligibility check.
- Other models, seeds and shift conditions need their own assessment.
- **m = 1024 is not called converged**, and m = 2048 was not run.

## 2. Paired eligibility (IG m = 512 = production setting for the pilot; Grad-CAM handling unchanged)

**Mismatch donors**
- The mismatch donor identities are the same as in Stage 3B (asserted). The mapping was persisted before the mismatch curves.
- Donor eligibility comes from the fresh m = 512 maps.
- Invalid donors were not replaced.

| Explainer | Explainer pair | Random | Centre | Sobel | Mismatch |
|---|---|---|---|---|---|
| IG m = 512 | **23/24** (mel 11, non-mel 12) | 23 | 23 | 23 | **22** (mel 10, non-mel 12) |
| Grad-CAM | **15/24** (mel 10, non-mel 5) | 15 | 15 | 15 | **9** (mel 9, non-mel **0 → NOT ESTIMABLE**) |

**Exclusions**
- **IG:**
  - case812, completeness failure in dermoscopy;
  - for the mismatch contrast, case740 additionally, because its planned dermoscopy donor is the ineligible case812 map.
- **Grad-CAM:** all-zero maps in 9 pairs (3 dermoscopy, 8 clinical; 2 melanoma, 9 non-melanoma references). For the mismatch contrast, planned donor maps are ineligible in 7 further pairs.
  - The handling is unchanged: no absolute-value or alternative-target substitution.
- **Interpretation:** primary paired map comparisons describe **eligible pairs only**, and they cannot automatically be generalised to all cases.
- **Records kept:** all records are retained for later missingness and sensitivity work.
- **Earlier IG scores:** the Stage 3B m = 256 scores remain **diagnostic only** and were not relabelled.

## 3. Numerical limitation (recorded)

**Batch shape matters**
- Repeated calls with the **same batch shape** were exactly deterministic here: the endpoint logits from the two fresh IG calls and the Stage 3B baseline logits (also batch 48) agree with 0.0 difference.
- **Different batch shapes** differ by about 1e-6 in the logits. Stage 3B.1 found up to 4.8e-6 for the images and up to 7.2e-6 for the baselines, between batch 8 and batch 48.
- **No batch-size independence is claimed.**

**Production runs**
- Future production runs must specify how the completeness endpoint batches are formed.
- For 996 references at outer batch 48 there are 20 full batches and one final partial batch of 36. That partial batch must be stated explicitly.

## 4. Cost

| Kind | Value |
|---|---|
| **Measured time** (this run) | 452.2 s in-process. Wall-clock 18:18:04Z–18:25:39Z |
| Of which | IG m = 512: 139.4 s; IG m = 1024: 278.2 s; IG curves (2 × 47): 17.5 s; mismatch curves: 8.6 s; reuse check: 0.4 s; persistence: 2.3 s |
| **Measured GPU memory** | IG m = 1024: 1686 MiB allocated / 2360 MiB reserved. IG m = 512: 1640 / 2360 MiB |
| **Projected full-cohort time** (996 references, m = 512; a projection, not a measurement) | ≈ 4.96 s per reference → **≈ 1.37 h**. Assumes the same GPU and model, `eval_batch` 11, 11 curve pairs per reference and the Stage 3B curve rate |
| **Projected disk storage** (not GPU memory) | ≈ 1.2 GB of float32 maps (uncompressed) plus ≈ 0.025 GB of curves |

The projection excludes other architectures, seeds and shifts.

## 5. Outputs and integrity

**Outputs:**
- `maps_fresh.npz`;
- `curves_fresh.json`;
- `endpoint_logit_agreement.json`, `reuse_verification.json`;
- `ig_assessment_512_vs_1024.json`, `per_image_512_vs_1024.csv`;
- `mismatch_assignment_ig512.json`, `mismatch_ig512_results.json`;
- `scores_ig512.csv`, `pair_eligibility_ig512.csv`;
- `summary.json`, `receipt.json` and its sidecar.

**Code:** new `src/rv1/xai_pilot_fresh.py`. No existing module was changed, so no new tests were required.

**Integrity:** 530 files before and after, with **0 changed, 0 removed and 0 added**. The cache `.npy` files are unchanged. No `__pycache__` was created.

**Next (not started):** full XAI, RQ1 and retraining. Each needs separate authorisation.
