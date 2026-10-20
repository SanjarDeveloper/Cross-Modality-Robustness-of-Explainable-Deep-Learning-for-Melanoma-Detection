# Stage 2D2 Report — Insertion/Deletion Curves and Attribution Numerical Sensitivity

Date: 2026-09-27. All work in this stage used `research_v1/.venv_model`.

This is a software and numerics diagnostic, not RQ4. The model is the Stage 2C3 pilot checkpoint, and the images are the same 8 HAM TRAIN images as in Stage 2D1 (the model was trained on them). No IG step count is selected for production, and no inferential statistics were computed.

## Status

| Item | Result |
|---|---|
| Analytical tests (`test_xai_curves.py`, 9 new; `test_xai_attr.py`, 10) | **19 / 19 pass** |
| Full suite | **103 / 103 pass** |
| Diagnostic run | Completed; receipt verifies |
| Repeated identical calls (Grad-CAM, IG, curves) | Bit-identical |
| Numerical sensitivity at **default** precision | **Material.** IG internal batch size alone changes insertion AUC by up to 0.019 |
| TF32 hypothesis for the batch effects | **Largely supported**, but not complete for IG (§4) |
| IG m-convergence criterion (c) | Mixed; not resolved (§5) |
| Protected inputs | 0 changed, 0 removed |

## 1. Changes

**`src/rv1/xai_attr.py`** (the pre-change copy is at `runs/stage2d2/xai_attr_before_2d2.py.txt`)
- Every attribution computation now runs inside `torch.autocast(device_type, enabled=False)`, and model inputs are cast to float32. A test confirms that an enclosing bf16 autocast context — verified to be active — no longer changes the results.
- `precision_settings()` records the effective backend settings: the TF32 flags, `float32_matmul_precision`, cuDNN deterministic/benchmark, and whether deterministic algorithms are on.
- `highest_fp32()` is a context manager that turns TF32 off for matmul and cuDNN and sets matmul precision to "highest", then restores the previous settings. A test checks the restore.
- `eligibility()` keeps separate fields for `map_shape_valid` (finite and non-constant), `ig_completeness_pass`, and `eligible_for_primary_analysis`, which for IG is the conjunction of the two. The reasons are recorded and no record is dropped.
- **Correction to 2D1:** the Stage 2D1 report listed IG maps as "valid" without taking completeness into account. Under this definition, maps that fail completeness are **not** eligible for analysis.

**`src/rv1/xai_curves.py`** (new): the curves contract `curves_v1`.
- Rank pixels by descending map value; ties are broken by ascending flat pixel index (stable argsort).
- All three RGB channels of a selected pixel are replaced together.
- S = 32. Boundaries are e_j = ⌊j·N/S + ½⌋ (**round half up**), computed exactly in integers as (2jN + S) // (2S). For N = 224² this gives exactly 1568 pixels per step.
- Insertion goes from the baseline to the original image; deletion goes from the original to the baseline.
- **Curve values are the melanoma probability** (sigmoid). The **attribution target remains the logit**.
- All S+1 points and the actual fractions e_j/N are stored.
- **AUC v1 is the trapezoidal integral over the actual fractions.** The v0 mean is also returned, for comparison only.
- Checked on every call:
  - the ranking is a permutation;
  - every pixel is replaced exactly once;
  - image₀ and image_S are exactly the baseline and the original (construction checked);
  - the model is in eval mode;
  - FP32 with autocast disabled.
- Forward passes are chunked (`eval_batch`).

**Other files**
- New: `src/rv1/xai_curve_diag.py` and `configs/xai_curve_diag_v1.json`.
- `tests/test_xai_curves.py` is new.
- `test_xai_attr.py` is unchanged.

**Correction to the plan document.** `03_rev2` §14 states E[ins(f)] = g(f·x̄ + (1−f)·b̄) for random orderings. For a non-linear g that equality does **not** hold in general, because E[g(X)] ≠ g(E[X]). That test is not used. Any future test with random orderings must state its assumptions explicitly, for example a model that is exactly linear in pixel values.

## 2. Analytical tests (tolerance 1e-6, fixed beforehand)

| Test | Result |
|---|---|
| **Exact curves and AUC** on an additive model (logit = Σ V·x + c): both curves and both trapezoid AUCs agree with an independent numpy computation (lexsort ranking, floor(·+½) boundaries, hand-coded trapezoid) | Pass |
| **Directions:** with a faithful map on an all-positive additive model, insertion is non-decreasing and deletion is non-increasing; an anti-faithful ordering gives a lower insertion AUC and a higher deletion AUC | Pass |
| **Coverage:** boundaries for N = 224², 300², 49, 63 and 31 with S = 32 have the correct endpoints, match the formula, and differ in size by at most 1; the per-call assertions pass for N = 63 | Pass |
| **Ties:** a constant map gives index order; a map with partial ties gives the expected order | Pass |
| **All three channels together**, and endpoints: a spy model saw no pixel with mixed channels, and the images at steps 0 and 32 were exact | Pass |
| **Immutability and chunk equivalence:** input, baseline, map and model state are unchanged; `eval_batch` 1, 5 and 33 agree to within 1e-6 on the CPU; a model in train mode is rejected | Pass |
| **Autocast:** Grad-CAM, IG and curves give identical float32 results inside an active bf16 autocast context | Pass |
| **Eligibility split** and precision-context restore | Pass |

## 3. Diagnostic set-up

**Fixed choices**
- Curve evaluation: `eval_batch = 11` (3 chunks per curve) for every attribution comparison. It is varied only in its own check (§6).
- IG internal batch: 16. IG batch comparisons use m = 64.
- `cudnn.deterministic = True`, `benchmark = False`, deterministic algorithms on.

**Precision settings actually in effect**

| Setting | Default | "Highest" |
|---|---|---|
| `cuda.matmul.allow_tf32` | False | False |
| **`cudnn.allow_tf32`** | **True** | False |
| `float32_matmul_precision` | "highest" | "highest" |

So under PyTorch's defaults, **convolutions may use TF32** even though every tensor is FP32.

**Outputs:** `runs/xai_curve_diag/20260927T151903Z_650a4a1f_336ba7d4/`, containing:
- `maps.npz`: inputs, baseline, Grad-CAM maps for batch 1, batch 8 and highest precision, and IG attributions and primary maps for every variant;
- `curves.json`: every curve, its fractions and AUCs, and endpoint checks;
- `diagnostics.json`: all per-image measures;
- the receipt.

`verify_receipt` → OK.

## 4. Numerical sensitivity (per-image measures in `diagnostics.json`)

The table gives the median and maximum across the 8 images.

| Comparison | Relative map/attribution difference (to that image's own max) | Minimum Spearman correlation of maps | \|ΔAUC insertion\| | \|ΔAUC deletion\| |
|---|---|---|---|---|
| Repeated identical calls | 0 | 1 | 0 | 0 |
| Grad-CAM batch 1 vs 8, **default** | 0.9%, max 2.7% | 0.9992 | 0.0008, max **0.0042** | 0.0008, max 0.0023 |
| Grad-CAM batch 1 vs 8, **highest** | 4.5e-6, max 2.2e-5 | — | 2e-6, max 3e-5 | 3e-5, max 1.3e-4 |
| IG internal batch 1 vs 16 (m=64), **default** | 5.6%, max **16.7%** of the image's max \|attr\| | 0.958 | 0.0063, max **0.0188** | 0.0044, max 0.0145 |
| IG internal batch 1 vs 16 (m=64), **highest** | about 0.1–0.3% (for example 1.6e-3 on an image whose max is 0.57) | 0.9997 | 0.0005, max 0.0018 | 0.0006, max 0.0042 |
| Default vs highest precision | logits up to 0.0046; Grad-CAM 1.2%, max 4.8% | — | Grad-CAM 0.0008, max 0.0021; **IG 0.0045, max 0.0205** | IG 0.0047, max 0.0070 |

**Interpretation**

- **TF32 hypothesis.** Turning off TF32 in cuDNN removes nearly all of the Grad-CAM batch dependence (differences fall by more than three orders of magnitude). It also reduces the IG batch dependence by about 10–40× in AUC terms. This **supports** TF32 convolutions as the main source of the batch dependence. It does **not** show that TF32 is the only source: under highest precision, IG differences in the batch comparison remain, up to 0.0042 in deletion AUC.
- **Magnitude.** At default precision, IG batch and precision effects (up to about 0.02 in AUC) are **larger than the protocol's 0.005 convergence scale**. For IG they are therefore not negligible. Fixing the batch configuration gives repeatability, but it does not make this sensitivity go away.
- **Provisional configuration (documented, not frozen):** highest FP32 (`cudnn.allow_tf32 = False`), a fixed IG internal batch, and a fixed curve `eval_batch`.
- **Correction to Stage 2D1.** That report explained the batch effects by cuDNN kernel choice. The evidence now points mainly at TF32 at default precision. Kernel choice is not excluded.

## 5. IG step count m = 64 / 128 / 256 (internal batch 16; m=256 explicitly authorised)

| Precision | Completeness failures (of 8), m = 64 / 128 / 256 | p95 of e_rel | Criterion (c): median \|ΔAUC\| insertion / deletion (threshold 0.005) | Minimum Spearman correlation of maps |
|---|---|---|---|---|
| Default | 3 / **0** / **2** | 0.26 / 0.03 / 0.07 | 64 vs 128: 0.0087 / 0.0068; **128 vs 256: 0.0033 / 0.0041** | 0.949; 0.976 |
| Highest | 4 / 0 / 0 | 0.30 / 0.04 / 0.03 | 64 vs 128: 0.0088 / 0.0052; **128 vs 256: 0.0070 / 0.0031** | 0.954; 0.981 |

**Observations**
- At default precision, **m = 256 fails completeness on 2 images while m = 128 passes**. The e_rel values are 0.067 and 0.062. Under highest precision, m = 256 passes on all 8. This is consistent with TF32 round-off accumulating over more integration steps. That is a hypothesis, not established.
- Under highest precision, criterion (c) for 128 vs 256 is met for deletion (0.0031) but **not for insertion (0.0070 > 0.005)**.
- IG primary maps still change between m = 128 and m = 256: minimum Spearman correlation 0.976–0.981.

**Conclusion:** m is not selected. On these 8 training images from a 2-epoch pilot model, no single m meets all three protocol criteria under the provisional highest-precision setting. **m = 128 does not satisfy criterion (c) for insertion.** This must be re-evaluated on the real v1 model and the XAI pilot sample, possibly including m = 512, before any production choice.

## 6. Curve-evaluation batching (Grad-CAM batch-8 map, default precision)

Changing `eval_batch` from 11 to 1 or to 33 changes individual curve points by up to 0.0042–0.0047 in probability. The same effect appears in the endpoint checks: the curve endpoint, evaluated in a batch of 11, differs from a single forward pass by up to 0.0026, while the endpoint *images* are exactly equal.

This was measured **only at default precision**. It is likely that TF32 also drives this effect, but that was not tested; it is a to-do.

## 7. Diagnostic curve values (8 images, not results)

- Grad-CAM (batch 8, default precision):
  - insertion AUC 0.18–0.82;
  - deletion AUC 0.13–0.51.
- IG primary map (m = 128, default precision): insertion AUC 0.11–0.78.
- No control comparison was made, so these values show **nothing** about faithfulness.

## 8. Time and memory (CUDA-synchronised, 8 images)

**Grad-CAM**
- Batch 8: 0.10–0.31 s.
- Batch 1 × 8: 0.16 s.

**IG** (internal batch 16)

| m | Default | Highest |
|---|---|---|
| 64 | 1.69 s | 2.50 s |
| 128 | 3.40 s | 4.95 s |
| 256 | 6.65 s | 9.74 s |

Highest precision takes about **1.46×** as long. IG at m = 64 with internal batch 1: 8.1 s (default) and 11.0 s (highest).

**Curves:** both curves for 8 images with a Grad-CAM map take 1.27 s, i.e. about 0.16 s per image for the 66 forward passes.

**Total diagnostic:** 78.5 s. Peak memory stayed below about 1.9 GiB reserved.

These numbers are **not** a full RQ4 cost estimate.

## 9. Integrity

- 388 files before, 389 after.
- **0 changed, 0 removed.**
- The only file added is `configs/xai_curve_diag_v1.json`.
- The cache `.npy` size and modification times are unchanged.
- No `__pycache__` was created.
- Earlier receipts may report source differences for `xai_attr.py`, which was legitimately edited in this stage.

## 10. Unresolved

1. **Precision policy for RQ4.** The evidence favours highest FP32 (`cudnn.allow_tf32 = False`). It is a decision that affects runtime (about +46% for IG) and is not yet frozen. The training pilot also ran with cuDNN TF32 enabled; whether training should use it is a separate decision.
2. **IG step count.** Undetermined; see §5. It needs the real model and the pilot sample, and possibly m = 512.
3. Curve-evaluation batching sensitivity under highest precision is untested.
4. Whether to add explicit GPU tolerances for batch and precision variation, declared before the RQ4 pilot.
5. Still to come: the control suite (random, centre, Sobel, mismatch), eligibility on real data, other architectures, and full RQ4 cost.
6. The old random-order expectation claim in plan §14 is invalid as written (§1) and must not be used.

**Stage 2D2 is complete.**
