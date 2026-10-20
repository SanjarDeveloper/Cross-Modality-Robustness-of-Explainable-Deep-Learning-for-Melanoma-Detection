# Protocol Addendum B — 2026-09-27: XAI Numerical Precision Policy

**Applies to:** all subsequent XAI diagnostics and the planned RQ4 calculations. These are baseline construction, Grad-CAM, Integrated Gradients, endpoint predictions and insertion/deletion evaluation.

Earlier documents are preserved unchanged. This addendum does **not** change training precision retroactively, and it does not invalidate any earlier pilot or diagnostic record, including Stages 2C3, 2D1 and 2D2.

## B1. Policy `xai_precision_v1`

These settings are applied inside a scoped context, `rv1.xai_attr.xai_precision()`. The caller's settings are restored on exit.

| Setting | Value |
|---|---|
| Tensors | float32 |
| `torch.autocast` | disabled (`enabled=False`) |
| `torch.backends.cudnn.allow_tf32` | **False** |
| `torch.backends.cuda.matmul.allow_tf32` | **False** |
| `torch.set_float32_matmul_precision` | **"highest"** |
| `torch.backends.cudnn.deterministic` | True |
| `torch.backends.cudnn.benchmark` | False |

The effective settings, read back at the time of the computation, are recorded in every receipt.

**Why:** Stage 2D2 showed that PyTorch's default `cudnn.allow_tf32=True` produced batch- and precision-dependent differences in IG insertion/deletion AUC of up to about 0.02. Turning TF32 off reduced those differences substantially.

**Cost:** about 1.46× the IG time measured in Stage 2D2. That measurement is from a diagnostic and is not a guarantee.

## B2. Tolerances (fixed before the Stage 2D2.1 run)

- **Endpoint prediction tolerance: 1e-4** in probability, as in plan §9. The curve's endpoint predictions must agree with direct predictions on the original image and on the baseline to within this tolerance. A mismatch is reported as a failure. It is never replaced or relaxed.
- **Provisional engineering tolerance for curve-evaluation batching: 1e-3** in insertion/deletion AUC, measured against `eval_batch = 1`. This is an engineering tolerance only. It is neither the protocol's IG convergence criterion nor a statistical result.
- **The protocol threshold of 0.005** (02_rev2 §6.6, criterion (c)) is the **IG metric-convergence threshold**: the median |ΔAUC| between m and 2m. It is **not** a general acceptable error, and it is not an acceptable batch error. Earlier wording that could be read as treating 0.005 as a universal tolerance is corrected by this addendum.

## B3. Eligibility at the analysis boundary

Curves for primary analysis may only be computed through `analysis_curves()`, which requires an explicit eligibility record.

For IG, eligibility requires both a valid map shape (finite and non-constant) **and** a completeness pass.

A constant map may be evaluated only through an explicit diagnostic override. The result is labelled non-primary, and the reasons are stored with it.
