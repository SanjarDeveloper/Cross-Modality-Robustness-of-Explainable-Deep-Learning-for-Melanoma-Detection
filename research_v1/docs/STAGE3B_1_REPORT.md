# Stage 3B.1 Report — IG Candidate m = 512 vs m = 1024: STOPPED at the Compatibility Recheck

Date: 2026-10-01. All work in this stage used `research_v1/.venv_model`.

**Run:** `runs/xai_pilot_ext/20261001T181214Z_19ba11df_534bbf32/`. Receipt status: `failed` (stopped by design).

**Stage records:** `runs/stage3b1/`.

## Status

| Item | Result |
|---|---|
| Addendum E and config `xai_pilot_stage3b1_v1.json` (hash `19ba11df…`) frozen before computation | Done |
| Checkpoint `303d21e7…03dd` (file = sidecar = `best.json`) | Verified |
| Stage 3B receipt and all output hashes; settings and precision policy; pair and reference lists | Verified and compatible |
| **m = 512 compatibility recheck (8 references)** | **FAILED the pre-fixed tolerance.** The run stopped |
| m = 1024 computation | **Not run** |
| Candidate 512 assessment | **Not performed. The production IG m remains UNRESOLVED** |
| Stage 3B run | Untouched |
| Protected inputs | 0 changed, 0 removed |
| Tests (`test_xai_pairs`, 11) | Pass. No existing module was changed |

## 1. What the recheck found

Addendum E2 fixed the acceptance tolerances before the run:
- identical completeness flags;
- |Δe_rel| ≤ 1e-6;
- primary-map |Δ| ≤ 1e-6 × max|map|;
- |ΔAUC| ≤ 1e-6;
- identical curve and endpoint statuses.

| Quantity (8 references) | Result |
|---|---|
| IG m = 512 primary maps | **Bit-identical** (max \|Δ\| = 0.0) |
| Insertion and deletion AUCs | **Identical** (Δ = 0.0); endpoints identical |
| Completeness failure flags | Identical, including the known failure on case812 dermoscopy |
| **e_rel** | Differs by **1.8e-7 to 5.5e-6**. **6 of 8 references exceed 1e-6** |

**Diagnosed cause** (a diagnostic only, not used for any decision):
- `integrated_gradients` computes logit(x) and logit(baseline) in one no-grad forward pass over the whole batch passed in. That batch held 48 images in Stage 3B and 8 in the recheck.
- The batch size changes the GPU convolution results by a few 1e-6 in the logits. The measured logit differences between batch 48 and batch 8 are up to 4.8e-6 for the images and up to 7.2e-6 for the baselines.
- The attributions are computed image by image, which is why the maps are identical.
- The discrepancy is therefore confined to the completeness denominator, the delta-logit. It is **not** a difference in maps or curves.

**Margin to the threshold:** in Stage 3B, the m = 512 e_rel value nearest the 0.05 threshold was 0.0033 away. That is about 600× the observed discrepancy, so none of the Stage 3B failure flags could have changed.

**What was not done:** I did not relax the 1e-6 tolerance or redefine the recheck after seeing this. As Addendum E2 requires, cached and new outputs were **not combined**, and m = 1024 was not computed.

## 2. Grad-CAM and paired populations (unchanged from Stage 3B; no new computation)

- Under the existing rule, 11/48 Grad-CAM maps are all-zero and ineligible:
  - dermoscopy 3, clinical 8;
  - melanoma 2, non-melanoma 9.
- No substitution was made.
- Complete Grad-CAM pairs: **15/24**, of which melanoma 10 and non-melanoma 5.
- Grad-CAM mismatch contrast: melanoma 9 pairs. For non-melanoma there are **0 pairs, so this cell is NOT ESTIMABLE**. That is not a zero effect.
- Primary map comparisons describe eligible pairs only, and they cannot automatically be generalised to all cases.
- All pilot records are retained for later analysis of missingness.
- There are no IG m = 512 paired counts, because m = 512 was not selected. The m = 256 counts from Stage 3B remain **diagnostic only**.

## 3. Cost

| Kind | Value |
|---|---|
| **Measured** pilot time | Recheck of IG m = 512 on 8 references plus 6 curve pairs: **25.1 s**. Wall-clock 18:12:09Z–18:12:40Z |
| **Measured** GPU memory | 1602 MiB allocated / 1884 MiB reserved (recheck). Stage 3B IG m = 512: 1750 / 2406 MiB |
| **Projected** full-cohort time (996 references, m = 512, from Stage 3B timings; not a measurement) | ≈ 1.34 h. m = 1024 itself would add roughly 2× the m = 512 IG time on the pilot, about 4.5 min for 48 references. This is projected, not measured |
| **Projected disk** (not GPU memory) | ≈ 1.2 GB of float32 maps plus ≈ 0.025 GB of curves |

## 4. Outputs and integrity

**Outputs:**
- `compat_recheck_m512.json`: per-reference differences;
- `summary.json`;
- `receipt.json` with its sidecar.

**New code:** `src/rv1/xai_pilot_ext.py`.

**New documents:**
- `docs/PROTOCOL_ADDENDUM_2026-10-01_E.md`;
- `configs/xai_pilot_stage3b1_v1.json`.

Both were written before the "before" integrity snapshot.

**Integrity:** 516 files before and after, with **0 changed, 0 removed and 0 added**. The cache `.npy` files are unchanged. No `__pycache__` was created in `src` or `tests`.

## 5. Decision needed (not taken here)

One of these, each needing authorisation:

1. **Recompute m = 512 on all 48 references** (about 134 s), in the same batch as m = 1024. Use only these fresh m = 512 and m = 1024 outputs, which avoids mixing cached and new values. This option is recommended.
2. **Amend the recheck tolerance for e_rel**, for example to 1e-4, with this diagnosis as the justification. This would be a post-hoc change and would need to be recorded as such.
3. Make the completeness logits batch-independent, for example by evaluating each image singly. That changes code and would need re-verification.

**Stage 3B.1 stopped safely.** Full XAI and RQ1 were not started.
