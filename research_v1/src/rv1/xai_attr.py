"""Attribution-map foundations (Stage 2D1): blurred baseline, Grad-CAM, Integrated
Gradients, validity and completeness diagnostics. No curves, controls or statistics.

Contract "attr_v1"
------------------
Target      : the melanoma LOGIT (single output unit), never the sigmoid probability.
Model state : caller must pass a model in eval() mode (checked); computations are FP32;
              gradients are taken with torch.autograd.grad w.r.t. activations/inputs only,
              so no parameter .grad is created or accumulated and no parameter/buffer is
              modified. Hooks are forward hooks, removed in a finally block after each call.

Baseline (plan 03 rev1 §9 / rev2 §9, identical to legacy code/eval/xai.py:_gauss_blur)
  applied to the NORMALISED input tensor x (float32, shape B,3,H,W);
  k = 2*floor(min(H,W)/8) + 1  (224 -> 57), sigma = k/6 (224 -> 9.5);
  separable 1-D Gaussian (normalised to sum 1), horizontal then vertical pass,
  each with F.pad(mode="reflect") of k//2 = 28 pixels (reflect excludes the edge pixel);
  output has the input's shape and dtype. It is an attribution reference, not an RQ2
  perturbation.

Grad-CAM
  A = activations of the configured feature layer (timm resnet50: `layer4`, i.e. the
  output of layer4[-1], shape B,2048,7,7); G = d logit / d A;
  w_c = mean_{h,w} G_c;  cam_raw = ReLU(sum_c w_c * A_c)  (B,h,w);
  map = F.interpolate(cam_raw[:,None], size=(H,W), mode="bilinear", align_corners=False).
  Stored: cam_raw (pre-upsampling) and map (upsampled, NOT normalised).

Integrated Gradients (Riemann MIDPOINT rule, m steps)
  alpha_k = (k + 0.5)/m, k = 0..m-1;  x_k = b + alpha_k (x - b);
  attr = (x - b) * mean_k d logit(x_k)/d x_k   (signed, per channel, B,3,H,W).
  primary map   = ReLU(sum_RGB attr)       (no smoothing)
  secondary map = sum_RGB |attr|
Completeness (protocol 02 rev2 §6.6), on ALL signed attributions before ReLU:
  delta = logit(x) - logit(b);  e_abs = |sum(attr) - delta|;
  e_rel = e_abs/|delta| only if |delta| >= 0.1, else undefined (None);
  failure = (e_rel defined and e_rel > 0.05) or (e_rel undefined and e_abs > 0.05).
Completeness is a numerical-integration check only; it says nothing about usefulness.

Numerical policy (Stage 2D2): every attribution computation runs inside
torch.autocast(device_type, enabled=False), so an enclosing autocast context cannot
lower the precision; tensors are float32. Casting inputs alone would NOT override an
enclosing autocast. The effective backend settings (TF32 flags, matmul precision, cuDNN
determinism/benchmark) are NOT fixed here; record them with precision_settings().

Eligibility: map_shape_valid (finite and non-constant) is separate from
ig_completeness_pass; for IG, eligible_for_primary_analysis = both. Records are never
dropped; reasons are kept.

Validity flags (map in the space it will be ranked in): nonfinite (any NaN/Inf),
all_zero (every value == 0), constant (max - min < 1e-12, includes all_zero).
Invalid maps are flagged, never replaced or dropped here.
"""
import math

ATTR_VERSION = "attr_v1"
PRECISION_POLICY = "xai_precision_v1"
CONST_TOL = 1e-12
DELTA_MIN = 0.1
REL_TOL = 0.05
ABS_TOL = 0.05


class xai_precision:
    """Scoped policy xai_precision_v1 (Addendum B): TF32 off (cuDNN + matmul), float32 matmul
    precision 'highest', cudnn.deterministic True, cudnn.benchmark False. Caller settings are
    restored on exit. Autocast is disabled separately inside each computation."""

    def __enter__(self):
        import torch
        b = torch.backends
        self.prev = (b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32, torch.get_float32_matmul_precision(),
                     b.cudnn.deterministic, b.cudnn.benchmark)
        b.cuda.matmul.allow_tf32 = False
        b.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        b.cudnn.deterministic = True
        b.cudnn.benchmark = False
        return self

    def __exit__(self, *exc):
        import torch
        b = torch.backends
        (b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32, p, b.cudnn.deterministic, b.cudnn.benchmark) = self.prev
        torch.set_float32_matmul_precision(p)
        return False


def _policy(fn):
    import functools

    @functools.wraps(fn)
    def wrapped(*a, **k):
        with xai_precision():
            return fn(*a, **k)
    return wrapped


def blur_params(h, w):
    k = max(3, (min(h, w) // 8) * 2 + 1)
    return k, k / 6.0


@_policy
def gaussian_baseline(x):
    """Blurred baseline of a normalised input batch (B,3,H,W) float32."""
    import torch
    import torch.nn.functional as F
    if x.dim() != 4 or x.dtype != torch.float32:
        raise ValueError("expected float32 (B,C,H,W)")
    k, sig = blur_params(x.shape[-2], x.shape[-1])
    c = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(c ** 2) / (2 * sig ** 2))
    g = (g / g.sum()).view(1, 1, 1, k)
    C = x.shape[1]
    y = F.conv2d(F.pad(x, (k // 2,) * 2 + (0, 0), mode="reflect"), g.expand(C, 1, 1, k), groups=C)
    y = F.conv2d(F.pad(y, (0, 0) + (k // 2,) * 2, mode="reflect"),
                 g.transpose(2, 3).expand(C, 1, k, 1), groups=C)
    return y


def _check_eval(model):
    if model.training:
        raise RuntimeError("attribution requires model.eval()")


def _no_autocast(x):
    import torch
    return torch.autocast(device_type=x.device.type, enabled=False)


def _logit(model, x):
    with _no_autocast(x):
        out = model(x.float())
    if out.dim() == 2 and out.shape[1] == 1:
        out = out[:, 0]
    if out.dim() != 1:
        raise ValueError(f"expected a single melanoma logit per image, got {tuple(out.shape)}")
    return out.float()


@_policy
def grad_cam(model, layer, x):
    """Returns dict(cam_raw (B,h,w), map (B,H,W), logit (B,)) — detached float32 tensors."""
    import torch
    import torch.nn.functional as F
    _check_eval(model)
    store = {}

    def fwd(_m, _i, o):
        store["A"] = o
    h = layer.register_forward_hook(fwd)
    try:
        with torch.enable_grad(), _no_autocast(x):
            xi = x.detach().float()
            logit = _logit(model, xi)
            A = store["A"]
            (G,) = torch.autograd.grad(logit.sum(), A)
        with _no_autocast(x):
            w = G.float().mean(dim=(2, 3), keepdim=True)
            cam_raw = torch.relu((w * A.float()).sum(1))
            up = F.interpolate(cam_raw[:, None], size=x.shape[-2:], mode="bilinear",
                               align_corners=False)[:, 0]
        return {"cam_raw": cam_raw.detach(), "map": up.detach(), "logit": logit.detach()}
    finally:
        h.remove()
        store.clear()


@_policy
def integrated_gradients(model, x, baseline, steps, internal_batch=16):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive int")
    if not isinstance(internal_batch, int) or isinstance(internal_batch, bool) or internal_batch < 1:
        raise ValueError("internal_batch must be a positive int")
    """Returns dict(attr (B,3,H,W) signed, primary (B,H,W), absolute (B,H,W),
    logit, logit_baseline, completeness list)."""
    import torch
    _check_eval(model)
    x = x.detach().float()
    b = baseline.detach().float()
    if x.shape != b.shape:
        raise ValueError("baseline shape differs")
    alphas = [(k + 0.5) / steps for k in range(steps)]
    grads = torch.zeros_like(x)
    diff = x - b
    for i in range(x.shape[0]):
        acc = torch.zeros_like(x[i])
        for s in range(0, steps, internal_batch):
            a = torch.tensor(alphas[s:s + internal_batch], device=x.device, dtype=x.dtype)
            xa = (b[i][None] + a.view(-1, 1, 1, 1) * diff[i][None]).requires_grad_(True)
            with torch.enable_grad(), _no_autocast(xa):
                out = _logit(model, xa)
                (g,) = torch.autograd.grad(out.sum(), xa)
            acc += g.sum(0)
        grads[i] = acc / steps
    attr = diff * grads
    with torch.no_grad():
        lx, lb = _logit(model, x), _logit(model, b)
    comp = [completeness(float(attr[i].double().sum()), float(lx[i]), float(lb[i]))
            for i in range(x.shape[0])]
    return {"attr": attr.detach(), "primary": torch.relu(attr.sum(1)).detach(),
            "absolute": attr.abs().sum(1).detach(), "logit": lx, "logit_baseline": lb,
            "completeness": comp, "steps": steps, "rule": "riemann_midpoint"}


def completeness(sum_attr, logit_x, logit_b):
    delta = logit_x - logit_b
    e_abs = abs(sum_attr - delta)
    e_rel = e_abs / abs(delta) if abs(delta) >= DELTA_MIN else None
    fail = (e_rel is not None and e_rel > REL_TOL) or (e_rel is None and e_abs > ABS_TOL)
    finite = all(math.isfinite(v) for v in (sum_attr, logit_x, logit_b))
    return {"sum_attr": sum_attr, "delta_logit": delta, "e_abs": e_abs, "e_rel": e_rel,
            "rel_defined": e_rel is not None, "failure": bool(fail or not finite)}


def validity(m):
    """Flags for one map tensor (any shape)."""
    import torch
    finite = bool(torch.isfinite(m).all())
    flags = []
    if not finite:
        flags.append("nonfinite")
    else:
        if bool((m == 0).all()):
            flags.append("all_zero")
        if float(m.max() - m.min()) < CONST_TOL:
            flags.append("constant")
    return {"valid": not flags, "flags": flags,
            "min": float(m.min()) if finite else None, "max": float(m.max()) if finite else None}


def display_normalise(m):
    """Min-max to [0,1] FOR DISPLAY ONLY (returns a new array; stored maps untouched)."""
    import numpy as np
    a = np.asarray(m, dtype=np.float64)
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi - lo < CONST_TOL else (a - lo) / (hi - lo)


def precision_settings():
    """Actual backend precision/determinism settings at call time (record with results)."""
    import torch
    return {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "autocast_inside_attribution": "disabled", "policy": PRECISION_POLICY}


class highest_fp32:
    """Context manager: TF32 off for matmul and cuDNN, matmul precision 'highest';
    previous settings restored on exit."""

    def __enter__(self):
        import torch
        self.prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                     torch.get_float32_matmul_precision())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        return self

    def __exit__(self, *exc):
        import torch
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, p = self.prev
        torch.set_float32_matmul_precision(p)


def eligibility(map_validity, ig_completeness=None):
    """Separate reasons; IG eligibility requires both a valid map and completeness pass."""
    shape_ok = bool(map_validity["valid"])
    out = {"map_shape_valid": shape_ok, "map_flags": list(map_validity["flags"])}
    if ig_completeness is None:
        out["ig_completeness_pass"] = None
        out["eligible_for_primary_analysis"] = shape_ok
        reasons = [] if shape_ok else [f"map:{f}" for f in map_validity["flags"]]
    else:
        cp = not ig_completeness["failure"]
        out["ig_completeness_pass"] = cp
        out["eligible_for_primary_analysis"] = shape_ok and cp
        reasons = ([] if shape_ok else [f"map:{f}" for f in map_validity["flags"]]) +                   ([] if cp else ["ig_completeness_failure"])
    out["ineligibility_reasons"] = reasons
    return out
