# Stage 2D2.1 Report — XAI Precision Freeze, Curve Guards, Bounded Numerical Diagnostic

Date: 2026-09-27. All work in this stage used `research_v1/.venv_model`.

- **Decision recorded:** `PROTOCOL_ADDENDUM_2026-09-27_B.md`, which defines the policy `xai_precision_v1`.
- **Evidence scope:** software and numerics diagnostic only. The model is the Stage 2C3 pilot checkpoint. The images are the same 8 HAM TRAIN images as before, and that model was trained on them.
- **Not included:** no production m was selected, no control suite, no RQ4, no training. Nothing here says anything about clinical faithfulness.

## Summary

| Area | Result |
|---|---|
| **Software correctness** | 25/25 XAI tests pass (19 earlier + 6 new guard tests). Full suite: **109/109** |
| **Endpoint and batching checks** (highest FP32) | **PASS.** Endpoint predictions agree to ≤ 1e-6 (tolerance 1e-4). The AUC difference for eval_batch 11 or 33 against 1 is ≤ 2.4e-7 (engineering tolerance 1e-3) |
| **IG convergence, diagnostic** | m = 128 vs 256: **does not meet** criterion (c) for insertion. m = 256 vs 512: **meets all criteria on this sample**. m = 512 is **not** shown to be converged |
| **Production decisions** | Unresolved (§5) |
| Protected inputs | 0 changed, 0 removed |

## 1. Software changes and correctness

**`xai_attr.py`**
- A new scoped context, `xai_precision()`, implements policy `xai_precision_v1`:
  - TF32 off for cuDNN and matmul;
  - matmul precision "highest";
  - `cudnn.deterministic` True and `cudnn.benchmark` False;
  - the caller's settings restored on exit.
- It is applied as a decorator to `gaussian_baseline`, `grad_cam` and `integrated_gradients`. Autocast stays disabled inside each computation.
- `integrated_gradients` now rejects a non-positive or non-integer `steps` or `internal_batch`.
- `precision_settings()` also records the name of the policy.

**`xai_curves.py`** (the pre-change copies of both modules are in `runs/stage2d2_1/`)
- **Guards:**
  - rejects a non-finite x, baseline or map;
  - `s` and `eval_batch` must be positive integers;
  - the model must return exactly one logit per image, as (B,1) or (B,);
  - the shapes must match;
  - the model must be in eval mode.
- **Constant maps** raise `IneligibleMap` unless `allow_constant` is set. That flag is set only through the diagnostic override of the analysis wrapper.
- The whole call runs under `xai_precision_v1`.
- **Endpoints**, reported as two separate things:
  - (a) **exact construction of the endpoint images**, which is asserted;
  - (b) **agreement of the endpoint predictions**: the curve values at j = 0 and j = S must match direct single-image predictions on x and on the baseline to within **1e-4**. The result is recorded as `endpoint_status` pass/fail together with the differences. The curve values themselves are never replaced.
- **`analysis_curves()`**, the boundary for primary analysis:
  - requires an explicit eligibility record;
  - for IG, requires the completeness result;
  - refuses ineligible maps;
  - allows a constant or all-zero map only with `diagnostic_override=True`, labelling it `non_primary_diagnostic` and keeping its reasons;
  - never lets an IG map that failed completeness through, override or not;
  - labels a primary curve whose endpoint check fails as `endpoint_failure`.
- **Direct endpoint predictions** are now always computed. This adds 2 forward passes per call, so the channel/endpoint test now counts 68 model inputs (66 curve images plus 2).

**New tests** in `tests/test_xai_guards.py`:

| Test | Result |
|---|---|
| Non-finite values in the input, baseline or map (NaN or Inf) are rejected | Pass |
| Invalid step or batch arguments (0, a float, a bool, a negative number), including IG `steps` and `internal_batch`, are rejected | Pass |
| A model with two outputs is rejected | Pass |
| A constant map needs the override and is then labelled non-primary with reasons `map:all_zero, map:constant` | Pass |
| An IG map that failed completeness is blocked, with or without the override; IG eligibility without a completeness result is refused; a missing record is refused; an IG map that passes completeness is primary | Pass |
| Endpoint mismatch: a model whose output depends on batch size gives status `fail` (above 1e-4), and the analysis wrapper labels it `endpoint_failure` | Pass |
| Inside curves and IG the policy is in effect (checked with a probe model: TF32 off, highest, deterministic, no benchmark); the caller's settings (TF32 on, benchmark, "medium") are restored afterwards | Pass |

`test_xai_curves.py` was updated for the renamed endpoint field and the two extra endpoint forward passes; its analytical expectations are unchanged.

`xai_curve_diag.py` (Stage 2D2) was updated only for the renamed field.

## 2. Endpoint and batching checks

Policy in effect, as recorded: `cudnn_allow_tf32` False, `cuda_matmul_allow_tf32` False, "highest", deterministic, no benchmark, autocast disabled.

Tolerances were **declared before the run**:
- endpoint: 1e-4;
- engineering tolerance for AUC under different evaluation batch sizes: 1e-3, with **eval_batch = 1 as the reference**.

| Map | Endpoint failures (eval_batch 1 / 11 / 33) | Maximum curve-point difference against eval_batch 1 (11 / 33) | Maximum \|ΔAUC\| (11 / 33) | Images outside tolerance |
|---|---|---|---|---|
| Grad-CAM | 0 / 0 / 0 | 1.8e-6 / 1.7e-6 | 2.1e-7 / 2.4e-7 | 0 / 0 |
| IG m = 128 primary | 0 / 0 / 0 | 1.5e-6 / 1.7e-6 | 2.1e-7 / 2.2e-7 | 0 / 0 |

- The largest endpoint prediction difference was ≤ 1e-6.
- Under the frozen policy, curve-evaluation batching is therefore numerically negligible for these 8 images and this model.
- By contrast, at default precision in Stage 2D2 the curve points differed by up to 0.0047.
- This is an engineering check. It is not the protocol's IG convergence criterion.

## 3. IG convergence (internal batch 16, curves at eval_batch = 1, highest FP32)

Protocol criteria (02_rev2 §6.6):
- (a) completeness failure rate ≤ 5%;
- (b) 95th percentile of e_rel ≤ 0.20;
- (c) median |ΔAUC| against 2m < **0.005**, for both insertion and deletion.

**0.005 is the IG metric-convergence threshold, not a general or batch tolerance** (Addendum B2). Earlier wording that could be read otherwise is corrected here.

**Per step count:**

| m | Failures (of 8) | p95 e_rel (defined for 8/8) | Endpoint failures | Eligible (shape valid and completeness pass) |
|---|---|---|---|---|
| 128 | 0 | 0.041 | 0 | 8/8 |
| 256 | 0 | 0.032 | 0 | 8/8 |
| 512 | 0 | 0.017 | 0 | 8/8 |

**Per candidate:**

| Candidate | Median \|ΔAUC\| insertion | Median \|ΔAUC\| deletion | (a) | (b) | (c) insertion | (c) deletion | Minimum Spearman correlation of maps | All criteria met on this sample |
|---|---|---|---|---|---|---|---|---|
| 128 vs 256 | **0.0070** | 0.0031 | ✓ | ✓ | **✗** | ✓ | 0.981 | **No** |
| 256 vs 512 | 0.0027 | 0.0030 | ✓ | ✓ | ✓ | ✓ | 0.993 | Yes (this sample only) |

**Per-image ΔAUC**
- 128 → 256:
  - insertion: 0.0057, 0.0131, −0.0096, −0.0176, −0.0066, 0.0011, 0.0073, −0.0020;
  - deletion: −0.0115, −0.0004, −0.0053, 0.0032, −0.0030, −0.0034, 0.0008, −0.0014.
- 256 → 512:
  - insertion: −0.0022, −0.0003, −0.0082, **0.0125**, 0.0032, 0.0069, −0.0018, 0.0001;
  - deletion: 0.0029, 0.0050, 0.0012, −0.0030, −0.0036, 0.0030, −0.0015, 0.0004.

**Interpretation**
- On this sample, candidate m = 256 meets the criteria. The medians do meet them, but single images still move by up to 0.0125 between 256 and 512.
- m = 512 is **not** called converged, because m = 1024 was not run (out of scope).
- Eight training images from a 2-epoch pilot model are **not** a sufficient basis for selecting the production m.

## 4. Time and memory (CUDA-synchronised, 8 images, highest FP32)

| Operation | Time |
|---|---|
| Grad-CAM, batch 8 | 0.24 s |
| IG m = 128 | 4.79 s |
| IG m = 256 | 9.67 s |
| IG m = 512 | 19.34 s (≈ 2.4 s per image, linear in m) |
| Two curves per image × 8 images, eval_batch = 1 | ≈ 3.8–3.9 s |
| Two curves per image × 8 images, eval_batch = 11 or 33 | ≈ 1.25 s |
| Whole diagnostic | 57.3 s |

**Peak memory**
- IG: about 1.57 GiB allocated / 1.85 GiB reserved.
- Curves at eval_batch 33: 0.55 GiB allocated; up to 2.05 GiB reserved during the run.

These figures are not an estimate of the full RQ4 runtime.

## 5. Unresolved production decisions

1. **Production IG m.** It should be chosen on the real v1 model and the XAI pilot sample, not on this set. Candidates suggested by this diagnostic: m = 256 (with a comparison against 512), possibly with a check against 1024. At about 2× the cost of m = 128, that must be weighed in the RQ4 budget.
2. **Curve evaluation batch size.** Under the frozen policy, eval_batch 11 and 33 match eval_batch 1 within the engineering tolerance on this sample. A provisional choice of 11 or 33 is reasonable for speed. It should be re-checked on the real model and sample, with eval_batch 1 kept as the conservative reference.
3. Training precision is **unchanged** by Addendum B (the training pilot ran with the default `cudnn.allow_tf32`). Whether main training should also turn TF32 off is a separate decision.
4. Still to do: the control suite, the real-model XAI pilot, other architectures, and an estimate of the full RQ4 cost.

## 6. Files and integrity

**New files**
- Code: `src/rv1/xai_curve_diag2.py`.
- Tests: `tests/test_xai_guards.py`.
- Config: `configs/xai_curve_diag2_v1.json`, which records Addendum B as a supporting input.
- The addendum itself.
- Run: `runs/xai_curve_diag2/20260927T153704Z_6bc8db15_e0dff697/` (`maps.npz`, `diagnostics.json`, receipt). `verify_receipt` → OK.
- `runs/stage2d2_1/`: pre-change copies of the two modules, plus integrity evidence.

**Changed:** `xai_attr.py`, `xai_curves.py`, `xai_curve_diag.py` (field name only) and `test_xai_curves.py`.

**Integrity:** 400 files before, 402 after.
- **0 changed, 0 removed.**
- Added: the new config and the addendum.
- The earlier 2D2 outputs and configs, the checkpoints and the caches are unchanged. For the cache `.npy` files, size and modification time were compared.
- Earlier receipts may report source differences for the edited modules. They were not rewritten.

**Stage 2D2.1 is complete.**
