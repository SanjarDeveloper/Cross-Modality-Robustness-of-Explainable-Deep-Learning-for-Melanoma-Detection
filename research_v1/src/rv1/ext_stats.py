"""Stage 4C helpers: Wilson interval, PAD role isolation, tone-group mapping, subgroup eligibility."""
import math

Z975 = 1.959963984540054
PAD_ROLES = ("eval", "adapt_train", "adapt_val")
MIN_MEL_SUBGROUP = 5
DDI_TONES = {"12": "FST I-II", "34": "FST III-IV", "56": "FST V-VI"}


def wilson(x, n, z=Z975):
    """Wilson score interval for x successes of n (no continuity correction). None if n == 0."""
    if n == 0:
        return None
    if not (0 <= x <= n):
        raise ValueError("need 0 <= x <= n")
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def check_role_isolation(rows):
    """Every PAD patient (cluster_id) must belong to exactly one frozen role."""
    roles = {}
    for r in rows:
        if r["rq5_role"] not in PAD_ROLES:
            raise ValueError(f"{r['record_uid']}: unknown role {r['rq5_role']!r}")
        roles.setdefault(r["cluster_id"], set()).add(r["rq5_role"])
    bad = sorted(k for k, v in roles.items() if len(v) > 1)
    if bad:
        raise ValueError(f"patients in more than one role: {bad[:5]}")
    return {role: sum(1 for v in roles.values() if role in v) for role in PAD_ROLES}


def pad_tone(raw):
    """PAD Fitzpatrick metadata ('1.0'..'6.0') -> 'FST 1'..'FST 6'; '' -> 'missing' (never imputed)."""
    if raw is None or str(raw).strip() == "":
        return "missing"
    v = float(raw)
    if v != int(v) or not 1 <= v <= 6:
        raise ValueError(f"unexpected Fitzpatrick value {raw!r}")
    return f"FST {int(v)}"


def ddi_tone(raw):
    if raw not in DDI_TONES:
        raise ValueError(f"unexpected DDI tone group {raw!r}")
    return DDI_TONES[raw]


def subgroup_eligible(n_mel):
    return n_mel >= MIN_MEL_SUBGROUP
