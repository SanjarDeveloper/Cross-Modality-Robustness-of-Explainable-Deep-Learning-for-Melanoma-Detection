# Stage 2D1 Report — Attribution-Map Correctness Foundations (Grad-CAM, IG)

Date: 2026-09-27. All work in this stage used `research_v1/.venv_model`. No new packages were installed: the implementation is custom code, not Captum.

## Status

| Item | Result |
|---|---|
| Analytical tests (`tests/test_xai_attr.py`) | **10 / 10 pass** |
| Full test suite | **94 / 94 pass** |
| Diagnostic run (8 HAM TRAIN images, software diagnostic only) | Completed. Receipt verifies |
| Grad-CAM / IG map validity on the diagnostic images | All maps valid (no constant, all-zero or non-finite maps) |
| IG completeness failures, by step count | m=16: **8/8**, m=32: **6/8**, m=64: **3/8**, m=128: **0/8** |
| Protocol convergence rule for choosing m | **Cannot be evaluated yet** (§4) |
| GPU batch-size invariance | **Not bitwise.** Measurable differences, reported in §5 |
| Protected inputs | 0 changed, 0 removed |

This stage did **not** implement insertion/deletion curves, control maps, localisation statistics, perturbation sweeps or the full RQ4 analysis. No research result was produced.

## 1. Attribution contract `attr_v1` (`src/rv1/xai_attr.py`)

**Common rules for both methods**
- The target is the **melanoma logit**, never the probability.
- The model must be in `eval()` mode; this is checked, and a model in training mode raises an error. Computations use FP32.
- Gradients come from `torch.autograd.grad`, taken with respect to activations or inputs only. As a result, no parameter `.grad` is created and no parameters or buffers are changed.
- The forward hook is removed in a `finally` block after every call.

**Baseline**
- It is identical to the planned formula and to the legacy `code/eval/xai.py:_gauss_blur`. On the real diagnostic batch the maximum absolute difference between the two was **0.0**.
- It is applied to the *normalised* float32 input, shape (B, 3, H, W).
- Kernel size k = 2·⌊min(H,W)/8⌋+1, which is **57** for 224×224; σ = k/6 = **9.5**.
- The 1-D Gaussian is normalised to sum 1. A horizontal pass is followed by a vertical pass, each after `F.pad(mode="reflect")` with 28 pixels (reflection excludes the edge pixel).
- The output has the same shape and dtype as the input.
- This baseline is an attribution reference, not an RQ2 perturbation.

**Grad-CAM**
- Layer: timm ResNet-50 `model.layer4`, i.e. the output of `layer4[-1]`, shape 2048×7×7.
- Channel weights: w_c = spatial mean of ∂logit/∂A_c.
- Raw map: `cam_raw = ReLU(Σ_c w_c A_c)`, 7×7.
- Full-size map: `F.interpolate(mode="bilinear", align_corners=False)` to 224×224.
- Both `cam_raw` and the upsampled map are **stored without normalisation**.

**Integrated Gradients (IG)**
- **Riemann midpoint rule:** α_k = (k+0.5)/m.
- Attribution: attr = (x − b) ⊙ mean_k ∇ logit(b + α_k(x − b)). The signed per-channel attributions are stored.
- **Primary map:** `ReLU(Σ_RGB attr)`, with no smoothing.
- **Secondary map:** `Σ_RGB |attr|`.
- `internal_batch` controls how many α values are evaluated per forward pass.

**Completeness** (02_rev2 §6.6). It is computed on **all signed attributions before ReLU**:
- e_abs = |Σ attr − (logit(x) − logit(b))|.
- e_rel = e_abs / |Δ|, defined only when |Δ| ≥ 0.1.
- An image **fails** if e_rel > 0.05, or, when e_rel is undefined, if e_abs > 0.05.
- Non-finite values also count as a failure.
- Completeness checks only the numerical integration. It says nothing about how useful the explanation is.

**Validity flags**
- `nonfinite`, `all_zero`, and `constant` (max − min < 1e-12).
- Invalid maps are flagged, never replaced, and never dropped here. Deciding eligibility belongs to the later matched-pair analysis.
- `display_normalise` produces a copy for display only.

## 2. Analytical tests (tolerances fixed before running: 1e-5, and 1e-4 for large sums)

1. **Baseline**
   - Kernel size 57 and σ 9.5.
   - A constant image stays constant (reflect padding).
   - An impulse keeps its total mass (sum = 1).
   - The value at offset (5, −3) equals the analytic g(5)·g(−3).
   - Non-float32 input is rejected.
2. **Grad-CAM with a controlled model** (1×1 convolution → BatchNorm in eval mode → global average pooling → weight w):
   - The Grad-CAM map equals the analytic ReLU(w/(h·w)·A).
   - It is positive only on the known bright region.
   - The logit is analytic.
   - With w < 0 the map is **all zero** and is flagged `all_zero` and `constant`.
3. **Upsampling mode:** the full-size map equals bilinear interpolation with `align_corners=False` of a 4×4 `cam_raw`.
4. **Grad-CAM state safety**
   - Parameters and buffers are unchanged (checked by hash).
   - No hooks remain and no parameter gradients are created.
   - Repeated calls are identical.
   - Batched and single-image results agree to within 1e-5 on the CPU.
   - A model in training mode is rejected.
5. **IG on a linear model** at m = 1, 16 and 128:
   - attr = (x − b)·W exactly.
   - The weights include positive R, negative G and partly positive B, so the **signed aggregation** is exercised.
   - The primary map equals ReLU(Σ) and the absolute map equals Σ|·|.
   - Completeness error < 1e-4.
6. **IG on a cubic model** (f = 0.01 Σx³, baseline 0, where the exact IG is 0.01x³): the error falls strictly from m = 16 through 32 and 64 to 128, and is below 1e-3 at m = 128.
7. **IG batching and state:** internal batch 32 and internal batch 5 agree to within 1e-5 on the CPU. Model state is unchanged and no gradients are created.
8. **Completeness rule:** pass and fail cases; an undefined relative error handled by the absolute threshold (not passed silently); non-finite values fail.
9. **Validity flags:** each flag is tested, and a test confirms that display normalisation leaves the stored map unchanged.

## 3. Diagnostic run (software diagnostic only)

**Model:** the Stage 2C3 throughput-pilot checkpoint `runs/train_pilot/P1_20260927T143355Z/ckpt_epoch2.pt`.
- SHA-256 `610307ffd97136a856f6e4df0178c4367fbfa17670b917c9a88bf18d778721ab`, verified against the pointer and the sidecar before loading.
- Architecture: ResNet-50 `resnet50.a1_in1k`, pretrained-weight hash re-verified.
- **This is not a research model.** It was trained for 2 epochs in the pilot, and its training data *includes* these 8 images.

**Records:** 8 HAM TRAIN records, 4 melanoma and 4 non-melanoma, ordered by `stable_seed("xai_diag_v1", record_uid)`, using the evaluation preprocessing.

**Output:** `runs/xai_diag/20260927T144946Z_634c4e89_a4ee98ae/`
- `attributions.npz`: inputs, baseline, raw Grad-CAM (7×7 and 224), and for each m the IG signed attributions, primary map and absolute map;
- `diagnostics.json`;
- `review_sheet.png`;
- receipt: `verify_receipt` → OK.

**Per-image logits** (baseline logits are all negative, from −1.14 to −2.88):
- melanoma: 1.62, 0.61, 2.73, 2.87;
- non-melanoma: −0.05, −0.56, −0.55, −1.01.

**Map validity:** all 8 Grad-CAM maps and all IG primary and absolute maps at every m are valid.

**IG completeness:**

| m | failures | median e_rel | max e_rel |
|---|---|---|---|
| 16 | 8/8 | 0.102 | 0.624 |
| 32 | 6/8 | 0.123 | 0.499 |
| 64 | 3/8 | 0.040 | 0.319 |
| 128 | **0/8** | 0.010 | 0.033 |

The error does **not** always fall as m grows. For example, ISIC_0029453 has e_rel 0.11 at m=16, then 0.50 at m=32, then 0.32 at m=64, and 0.010 at m=128. This is consistent with a sharply non-linear logit along the integration path.

**Change of the IG primary map from m to 2m** (relative L1): 16→32: 0.27–0.53; 32→64: 0.15–0.31; 64→128: 0.09–0.17. So the **maps themselves are still changing by about 10% between m=64 and m=128**, even though completeness passes at m=128.

**Cost** (CUDA-synchronised; 8 images, FP32, RTX 5060 Laptop):

| Step | Time | Peak memory (allocated / reserved) |
|---|---|---|
| Build model and load checkpoint | 1.3 s | — |
| Grad-CAM | 0.25 s | 835 / 858 MiB |
| IG, m=16 | 0.52 s | ~1.55 / 1.82 GiB (similar for every m) |
| IG, m=32 | 0.82 s | |
| IG, m=64 | 1.63 s | |
| IG, m=128 | **3.27 s** (≈0.41 s per image) | |
| Whole diagnostic | 14.1 s | |

**Visual review** of the sheet (image | Grad-CAM | IG primary | IG absolute, at m=128):
- On the 4 melanoma images, Grad-CAM concentrates on the lesion.
- On the non-melanoma images it often highlights blobs away from the lesion. That is plausible for a fixed melanoma-logit target when there is little melanoma evidence, but plausible-looking heatmaps are **not** a correctness test and not clinical validation.
- The IG maps are nearly invisible under min–max display, because a few extreme pixels dominate the scale. This limits display only; the stored values are unaffected.

## 4. Unresolved definitions and decisions

1. **How to choose the IG step count m.** The protocol (02_rev2 §6.6) asks for three things:
   - (a) failure rate ≤ 5%;
   - (b) 95th percentile of e_rel ≤ 0.20;
   - (c) median primary-metric change < 0.005 against 2m.

   On this 8-image sample, (a) and (b) are met only at **m=128**.

   Criterion (c) refers to the insertion/deletion metric, which does not exist yet, so it is **unresolved**. It will be measured once the curves exist, on the real pilot sample rather than this diagnostic set. For m=128, the 2m comparison needs **m=256, which was not run**.

   The ~10% map change between m=64 and m=128 suggests that m=128 may still be too low for rank-based use. This is recorded as an open question, not decided.
2. **Display for the review sheet.** A percentile-clipped display could make the IG maps visible. It would affect display only and has not been adopted.
3. Eight training-set images from a 2-epoch pilot model cannot represent failure rates on the evaluation data, the trained research model, or other architectures.

## 5. Observed GPU numerical behaviour (not bitwise batch-invariant)

- **Repeated identical calls** are bit-identical.
- **Grad-CAM, a batch of 8 vs single images:** maximum absolute difference 0.0011 on the 224 maps, about **1.6%** of the median per-image map maximum (0.069).
- **IG at m=64, internal batch 16 vs 8:** maximum absolute difference 0.0038 in the attributions, about **6%** of the largest attribution among the 2 images compared (0.066).
- On the CPU both are exact to within 1e-5 (tests 4 and 7).

The most likely cause is that cuDNN picks different kernels for different batch shapes. No warning about deterministic algorithms was raised. **No GPU tolerance had been declared beforehand**, so this is reported as an observation, not as a pass.

**Implication:** RQ4 runs should fix one batch configuration (images per batch and IG internal batch) for all conditions. Alternatively, a tolerance must be declared before the RQ4 pilot, and the sensitivity of rank-based metrics to this variation must be assessed.

## 6. Files and integrity

**New files**
- Code: `src/rv1/xai_attr.py` and `src/rv1/xai_diag.py`.
- Tests: `tests/test_xai_attr.py`.
- Config: `configs/xai_diag_v1.json` (purpose `xai_diag`).
- Run: `runs/xai_diag/<run>/`.
- Integrity evidence: `runs/stage2d1/`.

**Changed:** `config.py` gains the purpose `xai_diag`. During this change, an edit briefly made the new purpose require a training block. It was caught by the diagnostic's own config load and fixed before any run. The final file is covered by the 94 passing tests.

**Integrity:** 377 files before, 378 after.
- **0 changed, 0 removed.**
- The only file added is `configs/xai_diag_v1.json`.
- The legacy `code/eval/xai.py` was imported read-only with `-B`; no new `__pycache__` was created.
- The cache `.npy` size and modification times are unchanged.

## 7. Before a full XAI timing estimate is possible

- Insertion/deletion curves v1 (32 steps, full pixel coverage) and their per-image cost: this is the forward-pass-heavy part.
- Control maps (5 random seeds, centre, Sobel, mismatch) and their cost.
- Resolving the IG step count, including a possible m=256 check and criterion (c) once curves exist.
- A fixed batch configuration or declared GPU tolerances (§5).
- The real v1-trained model and the full XAI manifest (249+249 pairs × 2 modalities), plus the simulated arm.
- EfficientNet-B3 and DenseNet-121: layer choice and cost.

The current figure of about 0.41 s per image for IG at m=128 covers IG only and must not be extrapolated to the full RQ4 workload.

**Stage 2D1 is complete.**
