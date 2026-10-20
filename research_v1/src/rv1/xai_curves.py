"""Insertion/deletion curves v1 (Stage 2D2; guards and endpoint status added in 2D2.1).

Contract "curves_v1"
--------------------
* Inputs: normalised image x (3,H,W) float32, baseline b = xai_attr.gaussian_baseline
  (same tensor space), and a primary map s (H,W). All must be finite (checked).
* Ranking: pixels sorted by DESCENDING map value; ties broken by ascending flat pixel
  index (row-major), via a stable argsort of -s.
* Replacement: all three RGB channels of a selected spatial pixel are replaced together.
* Steps: S = 32 (positive int); boundaries e_j = floor(j*N/S + 1/2) (round-half-UP, exact
  integer arithmetic (2*j*N + S) // (2*S)), j = 0..S; e_0 = 0, e_S = N; each pixel changes
  exactly once (asserted). Reported fractions f_j = e_j / N.
* Insertion: image_j = x on the top-e_j pixels, b elsewhere (j=0 -> b, j=S -> x).
  Deletion : image_j = b on the top-e_j pixels, x elsewhere (j=0 -> x, j=S -> b).
* Curve values: melanoma PROBABILITY sigmoid(logit); attribution targets are the LOGIT.
* AUC v1 = trapezoidal integral over the actual fractions; v0 mean returned for comparison.
* Numerics: xai_precision_v1 (Addendum B) scoped for the whole call; autocast disabled;
  no_grad; eval mode required; exactly one output logit per image required.
* Endpoints: (a) endpoint IMAGE construction is exact (asserted: image_0/image_S masks);
  (b) endpoint PREDICTION agreement: curve values at j=0 and j=S versus direct single-image
  predictions on x and b must agree within ENDPOINT_TOL = 1e-4 (probability). A mismatch is
  reported (endpoint_status "fail"), never replaced.
* Eligibility: curves() computes only; primary-analysis use must go through
  analysis_curves(), which requires an eligibility record (xai_attr.eligibility). A
  constant/all-zero map is refused unless diagnostic_override=True, in which case the
  result is labelled non-primary with its reasons.
"""
import math

import numpy as np

CURVES_VERSION = "curves_v1"
S_DEFAULT = 32
ENDPOINT_TOL = 1e-4


class IneligibleMap(ValueError):
    pass


def _trapz(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x)


def _pos_int(v, name):
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        raise ValueError(f"{name} must be a positive int, got {v!r}")


def boundaries(n, s=S_DEFAULT):
    _pos_int(n, "n")
    _pos_int(s, "s")
    e = [(2 * j * n + s) // (2 * s) for j in range(s + 1)]
    assert e[0] == 0 and e[-1] == n and all(b >= a for a, b in zip(e, e[1:]))
    return e


def rank_positions(smap):
    """rank position (0 = most salient) of every flat pixel; stable tie-break by index."""
    import torch
    flat = smap.reshape(-1).double()
    order = torch.argsort(-flat, stable=True)
    pos = torch.empty_like(order)
    pos[order] = torch.arange(order.numel(), device=order.device)
    return order, pos


def _probs(model, imgs):
    import torch
    with torch.no_grad(), torch.autocast(device_type=imgs.device.type, enabled=False):
        o = model(imgs.float())
    if not ((o.dim() == 2 and o.shape == (imgs.shape[0], 1)) or (o.dim() == 1 and o.shape[0] == imgs.shape[0])):
        raise ValueError(f"model must return exactly one logit per image, got shape {tuple(o.shape)}")
    return torch.sigmoid(o.reshape(-1).float())


def curves(model, x, baseline, smap, s=S_DEFAULT, eval_batch=11, check=True, allow_constant=False):
    """Compute both curves. Returns dict with curves, fractions, AUCs and endpoint status."""
    import torch
    from .xai_attr import xai_precision
    if model.training:
        raise RuntimeError("curves require model.eval()")
    _pos_int(s, "s")
    _pos_int(eval_batch, "eval_batch")
    if x.dim() != 3 or x.shape != baseline.shape or smap.dim() != 2 or smap.shape != x.shape[-2:]:
        raise ValueError("expected x,baseline (3,H,W) and map (H,W)")
    for name, t in (("x", x), ("baseline", baseline), ("map", smap)):
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"non-finite values in {name}")
    if float(smap.max() - smap.min()) < 1e-12 and not allow_constant:
        raise IneligibleMap("constant map: only evaluable with an explicit diagnostic override")
    with xai_precision():
        H, W = smap.shape
        n = H * W
        e = boundaries(n, s)
        order, pos = rank_positions(smap)
        pos = pos.view(1, H, W).to(x.device)
        if check:
            assert torch.equal(torch.sort(order).values, torch.arange(n, device=order.device))
            counts = torch.zeros(n, dtype=torch.int64, device=order.device)
            for a, b in zip(e, e[1:]):
                counts[order[a:b]] += 1
            assert bool((counts == 1).all()), "a pixel is replaced zero or multiple times"
            assert not bool((pos < e[0]).any()) and bool((pos < e[s]).all())   # endpoint images exact
        xs, bs = x.float(), baseline.float()
        out = {}
        for kind in ("insertion", "deletion"):
            vals = []
            for c0 in range(0, s + 1, eval_batch):
                ks = list(range(c0, min(c0 + eval_batch, s + 1)))
                sel = torch.stack([(pos < e[k]).expand(3, H, W) for k in ks])     # all channels together
                if kind == "insertion":
                    imgs = torch.where(sel, xs.unsqueeze(0), bs.unsqueeze(0))
                else:
                    imgs = torch.where(sel, bs.unsqueeze(0), xs.unsqueeze(0))
                vals += _probs(model, imgs).cpu().tolist()
            out[kind] = vals
        px = float(_probs(model, xs[None])[0])
        pb = float(_probs(model, bs[None])[0])
    allv = out["insertion"] + out["deletion"] + [px, pb]
    finite = all(math.isfinite(v) for v in allv)
    if not finite:                      # never usable: AUCs invalidated, status recorded
        out.update(curve_status="nonfinite", endpoint_status="fail", auc_insertion_v1=None,
                   auc_deletion_v1=None, mean_insertion_v0=None, mean_deletion_v0=None,
                   p_full=px, p_base=pb, fractions=[k / n for k in e], boundaries=e, steps=s,
                   eval_batch=eval_batch, version=CURVES_VERSION, endpoint_images_exact=bool(check),
                   n_nonfinite_points=sum(not math.isfinite(v) for v in allv))
        return out
    diffs = {"ins0_minus_p_base": out["insertion"][0] - pb, "insS_minus_p_full": out["insertion"][-1] - px,
             "del0_minus_p_full": out["deletion"][0] - px, "delS_minus_p_base": out["deletion"][-1] - pb}
    worst = max(abs(v) for v in diffs.values())
    f = [k / n for k in e]
    out.update(fractions=f, boundaries=e, steps=s, eval_batch=eval_batch, version=CURVES_VERSION,
               p_full=px, p_base=pb, endpoint_images_exact=bool(check), curve_status="ok",
               endpoint_prediction_diffs=diffs, endpoint_prediction_max_abs=worst,
               endpoint_tol=ENDPOINT_TOL,
               endpoint_status="pass" if worst <= ENDPOINT_TOL and math.isfinite(worst) else "fail",
               auc_insertion_v1=float(_trapz(out["insertion"], f)),
               auc_deletion_v1=float(_trapz(out["deletion"], f)),
               mean_insertion_v0=float(np.mean(out["insertion"])),
               mean_deletion_v0=float(np.mean(out["deletion"])))
    return out


def analysis_curves(model, x, baseline, smap, eligibility_record, method, diagnostic_override=False,
                    **kw):
    """Primary-analysis boundary. eligibility_record = xai_attr.eligibility(...)."""
    if method not in ("gradcam", "ig"):
        raise ValueError("method must be 'gradcam' or 'ig'")
    need = ("map_shape_valid", "ig_completeness_pass", "eligible_for_primary_analysis", "ineligibility_reasons")
    if not isinstance(eligibility_record, dict) or any(k not in eligibility_record for k in need):
        raise IneligibleMap("an explicit eligibility record is required")
    if method == "ig" and eligibility_record["ig_completeness_pass"] is None:
        raise IneligibleMap("IG eligibility must include the completeness result")
    if method == "ig" and eligibility_record["ig_completeness_pass"] is not True:
        raise IneligibleMap("IG map did not pass completeness (ig_completeness_failure)")
    from .xai_attr import validity
    v = validity(smap)
    if bool(v["valid"]) != bool(eligibility_record["map_shape_valid"]):
        raise IneligibleMap("inconsistent eligibility metadata: map validity recomputed as "
                            f"{v['valid']} ({v['flags']}) but record says {eligibility_record['map_shape_valid']}")
    expected = v["valid"] and (method == "gradcam" or eligibility_record["ig_completeness_pass"] is True)
    if bool(eligibility_record["eligible_for_primary_analysis"]) != bool(expected):
        raise IneligibleMap("inconsistent eligibility metadata: eligible_for_primary_analysis contradicts "
                            "the recomputed map validity/completeness")
    reasons = list(eligibility_record["ineligibility_reasons"])
    if eligibility_record["eligible_for_primary_analysis"]:
        status = "primary"
    else:
        only_shape = reasons and all(r.startswith("map:") and r[4:] in ("constant", "all_zero") for r in reasons)
        if not (diagnostic_override and only_shape):
            raise IneligibleMap(f"map not eligible for primary analysis: {reasons}")
        status = "non_primary_diagnostic"
    r = curves(model, x, baseline, smap, allow_constant=(status == "non_primary_diagnostic"), **kw)
    r.update(analysis_status=status, ineligibility_reasons=reasons, method=method)
    if r.get("curve_status") != "ok":
        r["analysis_status"] = "invalid_nonfinite"
    elif r["endpoint_status"] != "pass" and status == "primary":
        r["analysis_status"] = "endpoint_failure"
    return r
