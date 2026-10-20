"""RQ2 perturbation simulator, version perturb_msim_v1 (Stage 8A).

Controlled synthetic image changes; NOT a model of real smartphone acquisition.

Pipeline position (as legacy code/data.py CachedDataset and rv1 prep_v1_cached256):
  cached 256x256 uint8 RGB  ->  apply(factor, severity)  ->  256x256 uint8 RGB
  ->  preprocess.to_model_input (cv2 INTER_LINEAR to 224, /255, ImageNet normalisation).
  Masks: 256-grid uint8 {0,1} (augment.mask_to_cache_grid of a mask_interp_v1 mask) -> apply -> 224 via
  INTER_NEAREST_EXACT (preprocess's second mask step).

Factors and parameters (legacy code/msim/simulator.py defaults; 02_rev2 §4: radius_max 7 px, JPEG
quality_min 18, zoom_out_max 2.4):
  blur          Gaussian, r = 7*s px; r < 0.3 -> identity; k = 2*round(3r)+1; cv2.BORDER_REPLICATE
  illumination  sign in {+1,-1} drawn once per image; bright = 0.28*s*sign; contrast = 1 - 0.25*s*sign;
                temp = 0.22*s*sign; x=(img/255-0.5)*contrast+0.5+bright; R*=(1+temp); B*=(1-temp);
                clip [0,1]*255 then astype(uint8) (truncation, as legacy); float32
  compression   s == 0: bypass (no re-encode); else JPEG quality q = round(95 + (18-95)*s) via cv2.imencode
                (OpenCV defaults incl. chroma subsampling), decode
  scale         zoom-out z = 1 + 1.4*s, offset fraction 0.18*s; legacy synthetic low-frequency skin surround
                (noise from rng), INTER_AREA inset, feathered alpha; mask: same geometry, INTER_NEAREST_EXACT
                inset into zeros (scale_v1)
  nuisance      glare ellipse (intensity 0.95*s) + round(16*s) semi-transparent hair strands; photometric
                (mask unchanged; occlusion NOT masked out)

RNG: one local numpy Generator(PCG64(stable_seed(split_seed, image_uid, factor))) per call; independent of
severity, record order, process and PYTHONHASHSEED; global RNG untouched. RETROSPECTIVE decision: every
random quantity of a factor is drawn unconditionally in a fixed order (legacy skipped the glare draw when
intensity < 0.01), so all severities share one realization; hair k uses the same draws at every severity
(higher severity adds strands).
"""
import math

import numpy as np

VERSION = "perturb_msim_v1"
FACTORS = ("blur", "illumination", "compression", "scale", "nuisance")
PHOTOMETRIC = ("blur", "illumination", "compression", "nuisance")
SEVERITY_GRID = tuple(i / 8 for i in range(9))
PARAMS = {"blur": {"radius_max": 7.0}, "illumination": {"brightness_max": 0.28, "contrast_max": 0.25, "temp_max": 0.22},
          "compression": {"quality_s0": 95, "quality_min": 18}, "scale": {"zoom_out_max": 2.4, "offset_max": 0.18},
          "nuisance": {"glare_max": 0.95, "hairs_max": 16}}


class PerturbError(ValueError):
    pass


def _lerp(a, b, s):
    return a + (b - a) * float(s)


def _rng(split_seed, image_uid, factor):
    from .seeds import stable_seed
    seed = stable_seed(split_seed, image_uid, factor)
    return np.random.Generator(np.random.PCG64(seed)), seed


def _check(img, mask, factor, severity):
    if factor not in FACTORS:
        raise PerturbError(f"unknown factor {factor!r}")
    if not isinstance(severity, (int, float)) or isinstance(severity, bool) or not math.isfinite(severity) \
            or not 0.0 <= severity <= 1.0:
        raise PerturbError(f"invalid severity {severity!r}")
    if not (isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.shape == (256, 256, 3)):
        raise PerturbError("image must be uint8 (256,256,3) RGB from the cached256 path")
    if mask is not None:
        if not (isinstance(mask, np.ndarray) and mask.dtype == np.uint8 and mask.shape == (256, 256)):
            raise PerturbError("mask must be uint8 (256,256) on the cache grid")
        if not set(np.unique(mask).tolist()) <= {0, 1}:
            raise PerturbError("mask values must be {0,1}")


def apply(img256, factor, severity, image_uid, split_seed=0, mask256=None):
    """Returns (img_out uint8 256x256x3, mask_out or None, meta). Inputs are never modified."""
    import cv2
    _check(img256, mask256, factor, severity)
    rng, seed = _rng(split_seed, image_uid, factor)
    s = float(severity)
    img = img256.copy()
    meta = {"version": VERSION, "factor": factor, "severity": s, "perturb_seed": seed, "split_seed": split_seed,
            "image_uid": image_uid, "geometry_changed": False, "mask_supported": True, "params": {}}
    mask_out = None if mask256 is None else mask256.copy()
    if factor == "blur":
        r = _lerp(0.0, PARAMS["blur"]["radius_max"], s)
        if r < 0.3:
            out = img
            meta["params"] = {"radius": r, "identity": True}
        else:
            k = int(2 * round(3 * r) + 1)
            out = cv2.GaussianBlur(img, (k, k), r, borderType=cv2.BORDER_REPLICATE)
            meta["params"] = {"radius": r, "ksize": k, "border": "BORDER_REPLICATE"}
    elif factor == "illumination":
        p = PARAMS["illumination"]
        sign = 1.0 if rng.random() < 0.5 else -1.0
        bright = _lerp(0.0, p["brightness_max"], s) * sign
        contrast = 1.0 + _lerp(0.0, p["contrast_max"], s) * (-sign)
        temp = _lerp(0.0, p["temp_max"], s) * sign
        meta["params"] = {"sign": sign, "brightness": bright, "contrast": contrast, "temp": temp}
        if s == 0:
            out = img
        else:
            x = img.astype(np.float32) / 255.0
            x = (x - 0.5) * contrast + 0.5 + bright
            x[..., 0] *= (1.0 + temp)
            x[..., 2] *= (1.0 - temp)
            out = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    elif factor == "compression":
        p = PARAMS["compression"]
        if s == 0:
            out = img
            meta["params"] = {"bypass": True}
        else:
            q = int(round(_lerp(p["quality_s0"], p["quality_min"], s)))
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                raise PerturbError("JPEG encode failed")
            out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            meta["params"] = {"quality": q, "encoded_bytes": int(buf.size), "opencv": cv2.__version__}
    elif factor == "scale":
        out, geo = _scale(img, s, rng)
        meta["params"] = geo
        meta["geometry_changed"] = s > 0
        if mask_out is not None and s > 0:
            small = cv2.resize(mask256, (geo["nw"], geo["nh"]), interpolation=cv2.INTER_NEAREST_EXACT)
            mask_out = np.zeros_like(mask256)
            mask_out[geo["y0"]:geo["y0"] + geo["nh"], geo["x0"]:geo["x0"] + geo["nw"]] = small
    else:
        out, prm = _nuisance(img, s, rng)
        meta["params"] = prm
    if out.dtype != np.uint8 or out.shape != (256, 256, 3):
        raise PerturbError("internal: bad output")
    return out, mask_out, meta


def _scale(img, s, rng):
    """Legacy msim.scale with explicit rng, all draws unconditional, geometry returned (scale_v1)."""
    import cv2
    h, w = img.shape[:2]
    noise = rng.normal(0, 1.0, (h, w, 1)).astype(np.float32)          # drawn first, always
    dy, dx = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))        # always drawn
    if s <= 0:
        return img, {"identity": True, "offset_dir": [dy, dx]}
    p = PARAMS["scale"]
    z = _lerp(1.0, p["zoom_out_max"], s)
    off = _lerp(0.0, p["offset_max"], s)
    nh, nw = max(8, int(h / z)), max(8, int(w / z))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    ring = np.concatenate([
        img[int(h * .18):int(h * .26), int(w * .18):int(w * .82)].reshape(-1, 3),
        img[int(h * .74):int(h * .82), int(w * .18):int(w * .82)].reshape(-1, 3),
        img[int(h * .18):int(h * .82), int(w * .18):int(w * .26)].reshape(-1, 3),
        img[int(h * .18):int(h * .82), int(w * .74):int(w * .82)].reshape(-1, 3)])
    lum = ring.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
    skin = np.median(ring[lum >= np.median(lum)], axis=0)
    ph, pw = max(8, int(h * .10)), max(8, int(w * .10))
    best, best_score = None, -1e9
    for fy in (.20, .35, .50, .65, .78):
        for fx in (.20, .35, .50, .65, .78):
            y, x = int((h - ph) * fy), int((w - pw) * fx)
            cand = img[y:y + ph, x:x + pw]
            tone = -np.abs(cand.reshape(-1, 3).mean(0) - skin).mean()
            tex = float(cv2.Laplacian(cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
            score = tone + min(tex, 40.0) * 0.15
            if score > best_score:
                best, best_score = cand, score
    pm = best.reshape(-1, 3).mean(0).astype(np.float32)
    ps = best.reshape(-1, 3).std(0).astype(np.float32).clip(2.0, 18.0)
    nz = cv2.GaussianBlur(noise, (0, 0), max(h, w) * 0.06)[..., None]
    nz /= max(nz.std(), 1e-6)
    surround = np.clip(pm[None, None, :] + nz * ps[None, None, :] * 0.55, 0, 255)
    y0 = int(np.clip((h - nh) // 2 + dy * off * h, 0, h - nh))
    x0 = int(np.clip((w - nw) // 2 + dx * off * w, 0, w - nw))
    alpha = np.zeros((h, w), np.float32)
    inner = max(1, int(min(nh, nw) * 0.10))
    alpha[y0 + inner:y0 + nh - inner, x0 + inner:x0 + nw - inner] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(3.0, min(nh, nw) * 0.10))[..., None]
    inset = surround.copy()
    inset[y0:y0 + nh, x0:x0 + nw] = small
    out = np.clip(inset * alpha + surround * (1 - alpha), 0, 255).astype(np.uint8)
    return out, {"zoom": z, "offset_frac": off, "offset_dir": [dy, dx], "nh": nh, "nw": nw, "y0": y0, "x0": x0,
                 "inset_interp": "INTER_AREA", "mask_interp": "INTER_NEAREST_EXACT", "mask_fill": 0,
                 "feather_inner_px": inner, "surround": "legacy synthetic low-frequency skin-tone noise"}


def _nuisance(img, s, rng):
    """Legacy msim.nuisance with explicit rng; glare and the maximum number of hair strands are drawn
    unconditionally so every severity shares the same realization."""
    import cv2
    h, w = img.shape[:2]
    p = PARAMS["nuisance"]
    cy, cx = int(rng.integers(h // 4, 3 * h // 4)), int(rng.integers(w // 4, 3 * w // 4))
    ang = float(rng.integers(0, 180))
    hairs = []
    for _ in range(p["hairs_max"]):
        hairs.append((int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(w // 3, int(w * 1.2))),
                      float(rng.random() * 2 * np.pi), float((rng.random() - 0.5) * 1.6), float((rng.random() - 0.5) * 2.2),
                      float(rng.uniform(0.30, 0.65))))
    gi = _lerp(0.0, p["glare_max"], s)
    n_h = int(round(_lerp(0, p["hairs_max"], s)))
    prm = {"glare_intensity": gi, "glare_center": [cy, cx], "glare_angle": ang, "n_hairs": n_h}
    if s <= 0:
        return img, prm
    out = img.astype(np.float32)
    if gi > 0.01:
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (cx, cy), (int(w * 0.14), int(h * 0.10)), ang, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), max(h, w) * 0.035)
        m = m / max(m.max(), 1e-6)
        out += (255.0 - out) * (m[..., None] * gi)
    if n_h:
        layer = np.zeros((h, w), np.float32)
        for (x0, y0, L, a0, c1, c2, op) in hairs[:n_h]:
            pts = np.array([[x0 + L * t * np.cos(a0 + c1 * t + c2 * t * t), y0 + L * t * np.sin(a0 + c1 * t + c2 * t * t)]
                            for t in np.linspace(0, 1, 48)], np.int32).reshape(-1, 1, 2)
            cv2.polylines(layer, [pts], False, op, thickness=1, lineType=cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 0.9)
        a = np.clip(layer, 0, 1)[..., None]
        out = out * (1 - a) + np.array([35.0, 28.0, 24.0])[None, None, :] * a
    return np.clip(out, 0, 255).astype(np.uint8), prm


def to_model_input(img256):
    from . import preprocess as PP
    return PP.to_model_input(img256)


def mask_to_input(mask256):
    import cv2
    from . import preprocess as PP
    return cv2.resize(mask256, (PP.INPUT_SIZE, PP.INPUT_SIZE), interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
