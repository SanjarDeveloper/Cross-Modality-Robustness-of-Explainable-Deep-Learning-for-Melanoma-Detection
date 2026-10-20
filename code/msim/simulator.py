"""Modality-Shift Simulator -- the study's technical contribution.

Converts a clean dermoscopic image toward smartphone capture conditions,
ONE FACTOR AT A TIME, at calibrated severities. This is what real paired
datasets cannot do: Derm7pt tells you *that* modality hurts, the simulator
tells you *which factor* and *how much*.

Five factors, each with a documented dermoscopy/smartphone difference:
  blur         -- hand-held phone without macro focus
  illumination -- ambient light + auto-white-balance vs polarised dermoscopy
  compression  -- lossy storage / messaging-app re-encoding
  scale        -- variable shooting distance, off-centre framing
  nuisance     -- glare, hair, background skin (absent in contact dermoscopy)

Every factor exposes severity in [0,1] where 0 == identity. Severity is
mapped to physical parameters by `calibrate.py`, which fits the endpoints to
measured statistics of the REAL smartphone sets, so severity=1.0 means
"as degraded as a typical real phone photo", not an arbitrary number.

VERIFICATION MODE: each factor ships a monotone image statistic that it must
move in a known direction. `verify()` asserts this, which catches a silently
broken transform -- the failure mode that would invalidate every downstream
threshold.
"""
import numpy as np
import cv2

RNG_DEFAULT = 12345


# --------------------------------------------------------------- statistics
def stat_sharpness(img):
    """Variance of Laplacian. Falls as blur rises."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def stat_colour_cast(img, ref=None):
    """Gray-world deviation: how far channel means are from equal.

    Must be measured RELATIVE TO THE UNSHIFTED IMAGE, because dermoscopic
    images already carry a strong intrinsic cast (commonly 15-25). Warming
    an already-warm image can *reduce* absolute deviation, so the absolute
    value is not monotone in severity. With `ref` supplied this returns the
    distance travelled in (brightness, cast) space, which is monotone.
    """
    m = img.reshape(-1, 3).mean(0).astype(np.float64)
    cast = float(np.abs(m - m.mean()).mean())
    if ref is None:
        return cast
    rm = ref.reshape(-1, 3).mean(0).astype(np.float64)
    # travel distance: per-channel mean shift, which grows with any
    # brightness/contrast/temperature perturbation regardless of sign.
    return float(np.abs(m - rm).mean())


def stat_blockiness(img):
    """JPEG 8x8 blocking energy: difference across block boundaries minus
    difference inside blocks. Rises as JPEG quality falls."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    d = np.abs(np.diff(g, axis=1))
    if d.shape[1] < 16:
        return 0.0
    cols = np.arange(d.shape[1])
    on = d[:, (cols % 8) == 7].mean()
    off = d[:, (cols % 8) != 7].mean()
    return float(on - off)


def stat_lesion_fraction(img, ref=None):
    """How much of the frame the dark lesion fills.

    A percentile threshold computed on the image itself is self-normalising
    (it always returns ~the percentile), so the threshold must come from the
    UNSHIFTED image. With `ref` given, the absolute intensity cut-off is
    taken from `ref`, making the statistic fall as the lesion shrinks.
    """
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    src = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY) if ref is not None else g
    thr = np.percentile(src, 35)
    return float((g < thr).mean())


def stat_specular(img, ref=None):
    """Bright-highlight energy from glare.

    Absolute saturation (>235) is the wrong probe: soft synthetic glare
    brightens large regions without clipping. Measuring the upper tail
    relative to the unshifted image captures it. Falls back to a fixed
    high-percentile mass when no reference is supplied.
    """
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    if ref is None:
        return float((g > 200).mean())
    rg = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float64)
    thr = np.percentile(rg, 97)
    return float((g > thr).mean() - (rg > thr).mean())


# ------------------------------------------------------------------ factors
def _lerp(a, b, s):
    return a + (b - a) * float(s)


def blur(img, s, params=None):
    """Defocus blur. s=0 identity; s=1 ~ typical hand-held phone softness."""
    p = params or {}
    r_max = p.get("radius_max", 7.0)
    r = _lerp(0.0, r_max, s)
    if r < 0.3:
        return img.copy()
    # Gaussian approximates defocus well at these radii and is separable.
    k = int(2 * round(3 * r) + 1)
    return cv2.GaussianBlur(img, (k, k), r, borderType=cv2.BORDER_REPLICATE)


def illumination(img, s, params=None, rng=None):
    """Brightness + contrast + white-balance (colour-temperature) shift."""
    p = params or {}
    rng = rng or np.random.default_rng(RNG_DEFAULT)
    if s <= 0:
        return img.copy()
    # Direction (warm/bright vs cool/dark) is a property of the IMAGE, not of
    # the severity: it must stay fixed while severity sweeps, or the sweep is
    # not monotone and the threshold estimate is meaningless. `sign` is
    # therefore drawn from the per-image generator once, and callers that
    # want a fixed direction can pass params["sign"].
    sign = float(p["sign"]) if "sign" in p else (1.0 if rng.random() < 0.5 else -1.0)
    bright = _lerp(0.0, p.get("brightness_max", 0.28), s) * sign
    contrast = 1.0 + _lerp(0.0, p.get("contrast_max", 0.25), s) * (-sign)
    # colour temperature: push R up / B down (warm) or the reverse (cool)
    temp = _lerp(0.0, p.get("temp_max", 0.22), s) * sign
    x = img.astype(np.float32) / 255.0
    x = (x - 0.5) * contrast + 0.5 + bright
    x[..., 0] *= (1.0 + temp)
    x[..., 2] *= (1.0 - temp)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def compression(img, s, params=None):
    """JPEG re-encode. s=0 -> quality 95; s=1 -> quality_min."""
    p = params or {}
    q = int(round(_lerp(95, p.get("quality_min", 18), s)))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, q))])
    if not ok:
        return img.copy()
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def scale(img, s, params=None, rng=None):
    """Lesion shrinks in frame + off-centre framing (greater shooting
    distance). Implemented as zoom-out with replicate padding, so the lesion
    genuinely occupies fewer pixels."""
    p = params or {}
    rng = rng or np.random.default_rng(RNG_DEFAULT)
    if s <= 0:
        return img.copy()
    h, w = img.shape[:2]
    z = _lerp(1.0, p.get("zoom_out_max", 2.4), s)          # >1 == further away
    off = _lerp(0.0, p.get("offset_max", 0.18), s)          # off-centre fraction
    nh, nw = max(8, int(h / z)), max(8, int(w / z))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    # The surround must be HEALTHY SKIN, which is what a phone photo taken
    # from further away actually contains. cv2.BORDER_REPLICATE is wrong
    # here: dermoscopic images have a dark vignette at the frame edge, so
    # replicating edge pixels floods the canvas with dark pixels and makes
    # the lesion appear to GROW. Instead, fill with the modal colour of a
    # peripheral-but-inside-the-vignette skin ring.
    ring = np.concatenate([
        img[int(h * .18):int(h * .26), int(w * .18):int(w * .82)].reshape(-1, 3),
        img[int(h * .74):int(h * .82), int(w * .18):int(w * .82)].reshape(-1, 3),
        img[int(h * .18):int(h * .82), int(w * .18):int(w * .26)].reshape(-1, 3),
        img[int(h * .18):int(h * .82), int(w * .74):int(w * .82)].reshape(-1, 3),
    ])
    # median over the brighter half of the ring: robust to lesion pixels
    # that intrude into the ring on large lesions.
    lum = ring.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
    skin = np.median(ring[lum >= np.median(lum)], axis=0)
    # Build the surround from REAL SKIN TEXTURE tiled from the ring region.
    # Mirroring the whole frame (BORDER_REFLECT) duplicates the lesion into
    # the periphery, creating kaleidoscope ghosts -- a spurious cue the model
    # could latch onto. A flat fill is equally wrong (obviously synthetic).
    # Instead tile a small lesion-free skin patch, which keeps pore texture
    # and colour without reproducing lesion structure.
    # Choose the tile patch by SEARCHING for a genuinely skin-like window
    # rather than trusting a fixed coordinate: at a fixed offset the window
    # often lands on the dark dermoscopic vignette (or on the lesion), which
    # produces a flat unnatural fill. Score candidate windows by closeness to
    # the measured skin tone and by having real texture.
    ph, pw = max(8, int(h * .10)), max(8, int(w * .10))
    best, best_score = None, -1e9
    for fy in (.20, .35, .50, .65, .78):
        for fx in (.20, .35, .50, .65, .78):
            y, x = int((h - ph) * fy), int((w - pw) * fx)
            cand = img[y:y + ph, x:x + pw]
            if cand.shape[:2] != (ph, pw):
                continue
            tone = -np.abs(cand.reshape(-1, 3).mean(0) - skin).mean()   # near skin
            tex = float(cv2.Laplacian(cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY),
                                      cv2.CV_64F).var())
            score = tone + min(tex, 40.0) * 0.15      # texture, but not lesion edge
            if score > best_score:
                best, best_score = cand, score
    patch = best if best is not None else img[:ph, :pw]
    # SYNTHESISE the surround from the patch's colour statistics rather than
    # tiling it: repeating a small patch imprints a periodic grid that no
    # camera produces and that a CNN can trivially detect. Low-pass filtered
    # noise matched to the patch's mean, per-channel spread and correlation
    # length gives non-repeating skin-like texture.
    pm = patch.reshape(-1, 3).mean(0).astype(np.float32)
    ps = patch.reshape(-1, 3).std(0).astype(np.float32).clip(2.0, 18.0)
    # Low-frequency only: skin at a distance is smooth, and high-frequency
    # speckle both looks wrong and adds sharpness the blur factor is supposed
    # to own. Correlation length is a large fraction of the frame.
    noise = rng.normal(0, 1.0, (h, w, 1)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), max(h, w) * 0.06)[..., None]
    noise /= max(noise.std(), 1e-6)
    surround = pm[None, None, :] + noise * ps[None, None, :] * 0.55
    surround = np.clip(surround, 0, 255)

    y0, x0 = (h - nh) // 2, (w - nw) // 2
    if off > 0:
        y0 = int(np.clip(y0 + rng.integers(-1, 2) * off * h, 0, h - nh))
        x0 = int(np.clip(x0 + rng.integers(-1, 2) * off * w, 0, w - nw))
    # Feathered alpha so the inset lesion blends into the surround. The
    # feather must be wide enough that no straight seam survives -- a visible
    # rectangle is a cue the model can read as "this image was degraded".
    alpha = np.zeros((h, w), np.float32)
    inner = max(1, int(min(nh, nw) * 0.10))
    alpha[y0 + inner:y0 + nh - inner, x0 + inner:x0 + nw - inner] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(3.0, min(nh, nw) * 0.10))[..., None]
    inset = surround.copy()
    inset[y0:y0 + nh, x0:x0 + nw] = small
    return np.clip(inset * alpha + surround * (1 - alpha), 0, 255).astype(np.uint8)


def nuisance(img, s, params=None, rng=None):
    """Capture clutter absent from contact dermoscopy: specular glare plus
    hair-like occlusions."""
    p = params or {}
    rng = rng or np.random.default_rng(RNG_DEFAULT)
    if s <= 0:
        return img.copy()
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    # --- glare: a tight, bright specular highlight.
    # A wide, weak ellipse just raises overall brightness (indistinguishable
    # from the illumination factor) and is barely visible. Real phone glare
    # is a small, near-saturating spot with a soft falloff, so keep it
    # compact and let it actually clip.
    gi = _lerp(0.0, p.get("glare_max", 0.95), s)
    if gi > 0.01:
        cy, cx = rng.integers(h // 4, 3 * h // 4), rng.integers(w // 4, 3 * w // 4)
        ry, rx = int(h * 0.10), int(w * 0.14)
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (int(cx), int(cy)), (rx, ry),
                    float(rng.integers(0, 180)), 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), max(h, w) * 0.035)
        m = m / max(m.max(), 1e-6)
        out += (255.0 - out) * (m[..., None] * gi)
    # --- hair: fine dark strands.
    # Thickness 1-3 px at full frame resolution renders as heavy black bars;
    # real terminal hairs are ~1-2 px at these sizes and are semi-transparent.
    # Draw on a separate layer and alpha-composite so strands look like hair
    # lying ON skin rather than ink drawn INTO the image.
    nh_ = int(round(_lerp(0, p.get("hairs_max", 16), s)))
    if nh_:
        layer = np.zeros((h, w), np.float32)
        for _ in range(nh_):
            x0, y0 = rng.integers(0, w), rng.integers(0, h)
            L = rng.integers(w // 3, int(w * 1.2))
            ang = rng.random() * 2 * np.pi
            # Second-order curvature: a single constant term gives near-
            # straight strokes that read as drawn lines, not hair.
            c1, c2 = (rng.random() - 0.5) * 1.6, (rng.random() - 0.5) * 2.2
            pts = np.array([[x0 + L * t * np.cos(ang + c1 * t + c2 * t * t),
                             y0 + L * t * np.sin(ang + c1 * t + c2 * t * t)]
                            for t in np.linspace(0, 1, 48)], np.int32).reshape(-1, 1, 2)
            # Strands are drawn sub-pixel thin and thickened only by the
            # blur below; thickness>=2 at 224px reads as a black bar.
            cv2.polylines(layer, [pts], False, float(rng.uniform(0.30, 0.65)),
                          thickness=1, lineType=cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 0.9)          # soften aliasing
        hair_rgb = np.array([35.0, 28.0, 24.0])                # dark brown, not black
        a = np.clip(layer, 0, 1)[..., None]
        out = out * (1 - a) + hair_rgb[None, None, :] * a
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ registry
# `relative=True` means the statistic is computed against the unshifted
# image (stat(shifted, ref)) because its absolute value is not monotone in
# severity -- see the docstrings of the individual statistics.
FACTORS = {
    "blur":         dict(fn=blur,         stat=stat_sharpness,
                         direction=-1, relative=False),
    "illumination": dict(fn=illumination, stat=stat_colour_cast,
                         direction=+1, relative=True),
    "compression":  dict(fn=compression,  stat=stat_blockiness,
                         direction=+1, relative=False),
    "scale":        dict(fn=scale,        stat=stat_lesion_fraction,
                         direction=-1, relative=True),
    "nuisance":     dict(fn=nuisance,     stat=stat_specular,
                         direction=+1, relative=True),
}
STOCHASTIC = {"illumination", "scale", "nuisance"}


class ModalityShift:
    """Callable single-factor transform, safe to use inside a DataLoader.

    Seeding: each image gets its own generator derived from (seed, factor,
    severity, image index) so results are identical across runs and across
    worker counts -- a requirement for the paired statistical tests.
    """

    def __init__(self, factor, severity, params=None, seed=RNG_DEFAULT):
        if factor not in FACTORS and factor != "none":
            raise ValueError(f"unknown factor {factor!r}; have {list(FACTORS)}")
        self.factor, self.severity = factor, float(severity)
        self.params = params or {}
        self.seed = seed

    def _rng(self, img):
        """Per-IMAGE generator, independent of severity and of call order.

        Keying on image content (not a call counter) means the same lesion
        gets the same glare placement / illumination direction at every
        severity, so a severity sweep varies only magnitude. It also makes
        results identical regardless of DataLoader worker count.
        """
        h = int(img[::37, ::37].sum())          # cheap content fingerprint
        return np.random.default_rng((hash((self.seed, self.factor, h))) % (2**32))

    def __call__(self, img):
        if self.factor == "none" or self.severity <= 0:
            return img
        spec = FACTORS[self.factor]
        if self.factor in STOCHASTIC:
            return spec["fn"](img, self.severity, self.params, rng=self._rng(img))
        return spec["fn"](img, self.severity, self.params)

    def __repr__(self):
        return f"ModalityShift({self.factor}, s={self.severity:.2f})"


def apply_factor(img, factor, severity, params=None, seed=RNG_DEFAULT):
    return ModalityShift(factor, severity, params, seed)(img)


# -------------------------------------------------------------- verification
def verify(img, factors=None, severities=(0.25, 0.5, 0.75, 1.0), tol=1e-9):
    """Assert each factor moves its statistic monotonically in the expected
    direction. Returns {factor: {"ok":bool, "values":[...]}}.

    This is the guard against a transform that silently does nothing -- which
    would make a threshold look like 'no degradation' rather than 'broken code'.
    """
    results = {}
    for name in (factors or FACTORS):
        spec = FACTORS[name]
        relative = spec.get("relative", False)
        # A relative statistic is evaluated as stat(shifted, ref). Its
        # baseline is stat(ref, ref) -- zero for "travel distance" style
        # statistics, but non-zero for stat_lesion_fraction, whose reference
        # value is the fraction of the frame the lesion occupies unshifted.
        base = spec["stat"](img, img) if relative else spec["stat"](img)
        vals, ok = [base], True
        for s in severities:
            shifted = apply_factor(img, name, s)
            v = spec["stat"](shifted, img) if relative else spec["stat"](shifted)
            vals.append(v)
        # require the endpoint to move the right way by a real margin
        delta = (vals[-1] - vals[0]) * spec["direction"]
        if delta <= tol:
            ok = False
        # require broad monotonicity (allow small non-monotone wobble)
        adj = [(vals[i + 1] - vals[i]) * spec["direction"] for i in range(len(vals) - 1)]
        if sum(1 for a in adj if a < -abs(delta) * 0.25) > 1:
            ok = False
        results[name] = {"ok": ok, "values": vals, "endpoint_delta": delta}
    return results
