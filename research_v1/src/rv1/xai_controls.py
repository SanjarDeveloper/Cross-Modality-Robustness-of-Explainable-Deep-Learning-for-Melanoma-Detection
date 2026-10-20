"""XAI control maps and control-referenced diagnostic scores (Stage 2D3), spec controls_v1
(research_v1/configs/xai_controls_v1_spec.json, frozen before computation).

Controls (maps on the 224 input grid, float64):
  random_maps(image_uid)        5 x Uniform[0,1), local numpy Generator(PCG64) seeded with
                                stable_seed("ctrl", k, image_uid), k = 0..4
  centre_map(H, W)              exp(-((i+.5-H/2)^2+(j+.5-W/2)^2)/(2 sigma^2)), sigma = min(H,W)/4
  sobel_map(x_norm)             rgb = x*STD+MEAN; Y = .299R+.587G+.114B; 3x3 Sobel (cross-
                                correlation), edge-replicate border; sqrt(gx^2+gy^2)
  mismatch_assignment(records)  per (label, modality, explainer, condition) group, cyclic
                                shift over a stable_seed("mismatch_v1", uid) order by the
                                smallest s >= 1 with no self and no same-case donor

Scores: random controls are evaluated map by map and their AUCs averaged WITHIN the image
(maps are never averaged). insertion_advantage = explainer - control; deletion_advantage =
control - explainer (positive = better than that control). No combined score.
No labels, masks or model outputs are used to build random/centre/Sobel maps.
"""
import numpy as np

from . import preprocess as PP
from .seeds import stable_seed

CONTROLS_VERSION = "controls_v1"
N_RANDOM = 5
CONST_TOL = 1e-12


def random_maps(image_uid, h=224, w=224):
    return [np.random.default_rng(stable_seed("ctrl", k, image_uid)).random((h, w), dtype=np.float64)
            for k in range(N_RANDOM)]


def centre_map(h=224, w=224):
    sigma = 0.25 * min(h, w)
    i = np.arange(h, dtype=np.float64)[:, None] + 0.5 - h / 2
    j = np.arange(w, dtype=np.float64)[None, :] + 0.5 - w / 2
    return np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))


GX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
GY = GX.T.copy()


def sobel_map(x_norm):
    """x_norm: normalised (3,H,W) array/tensor on the 224 grid."""
    x = np.asarray(x_norm.detach().cpu().numpy() if hasattr(x_norm, "detach") else x_norm, dtype=np.float64)
    rgb = x * np.asarray(PP.STD, np.float64)[:, None, None] + np.asarray(PP.MEAN, np.float64)[:, None, None]
    y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    p = np.pad(y, 1, mode="edge")
    H, W = y.shape
    gx = sum(GX[a, b] * p[a:a + H, b:b + W] for a in range(3) for b in range(3))
    gy = sum(GY[a, b] * p[a:a + H, b:b + W] for a in range(3) for b in range(3))
    return np.sqrt(gx ** 2 + gy ** 2)


def map_valid(m):
    m = np.asarray(m, dtype=np.float64)
    if not np.isfinite(m).all():
        return False, ["nonfinite"]
    if float(m.max() - m.min()) < CONST_TOL:
        return False, ["constant"]
    return True, []


def assignment_key(record_uid, modality, explainer, condition):
    """Composite identity of one mismatch assignment (record x modality x explainer x condition)."""
    return f"{record_uid}|{modality}|{explainer}|{condition}"


def mismatch_assignment(records):
    """records: dicts with record_uid, case_id, label, modality, explainer, condition.
    Returns {composite_key: {"record_uid", "modality", "explainer", "condition", "donor" (record_uid
    or None), "donor_key", "status", "reason", "group", "shift"}}. Duplicate composite keys raise."""
    keys = [assignment_key(r["record_uid"], r["modality"], r["explainer"], r["condition"]) for r in records]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise ValueError(f"duplicate mismatch assignment keys: {dup[:5]}")
    groups = {}
    for r in records:
        key = (str(r["label"]), r["modality"], r["explainer"], r["condition"])
        groups.setdefault(key, []).append(r)
    out = {}
    for key, rs in groups.items():
        rs = sorted(rs, key=lambda r: (stable_seed("mismatch_v1", r["record_uid"]), r["record_uid"]))
        n = len(rs)
        shift = None
        for s_ in range(1, n):
            if all(rs[i]["case_id"] != rs[(i + s_) % n]["case_id"] and
                   rs[i]["record_uid"] != rs[(i + s_) % n]["record_uid"] for i in range(n)):
                shift = s_
                break
        for i, r in enumerate(rs):
            k = assignment_key(r["record_uid"], r["modality"], r["explainer"], r["condition"])
            base = {"record_uid": r["record_uid"], "modality": r["modality"], "explainer": r["explainer"],
                    "condition": r["condition"], "group": list(key), "shift": shift}
            if shift is None:
                out[k] = dict(base, donor=None, donor_key=None, status="unavailable",
                              reason="group has <2 records" if n < 2 else "no valid derangement shift")
            else:
                d = rs[(i + shift) % n]
                out[k] = dict(base, donor=d["record_uid"], status="assigned", reason=None,
                              donor_key=assignment_key(d["record_uid"], d["modality"], d["explainer"], d["condition"]))
    return out


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def control_scores(expl, ctrl):
    """expl: dict auc_ins/auc_del of the explainer; ctrl: dict with auc_ins/auc_del, or for the
    random control 'auc_ins_each'/'auc_del_each' with EXACTLY N_RANDOM finite values each (mean taken
    within the image). Any missing/None/NaN/Inf value makes the comparison unavailable; the mean is
    never taken over only the successful seeds."""
    def na(reason):
        return {"status": "unavailable", "reason": reason, "insertion_advantage": None,
                "deletion_advantage": None}
    if not (_finite(expl.get("auc_ins")) and _finite(expl.get("auc_del"))):
        return na("explainer AUC missing or non-finite (ineligible or invalid curve)")
    if ctrl.get("status") != "available":
        return na(ctrl.get("reason") or "control unavailable")
    if "auc_ins_each" in ctrl or "auc_del_each" in ctrl:
        ins, dele = ctrl.get("auc_ins_each"), ctrl.get("auc_del_each")
        if not (isinstance(ins, (list, tuple)) and isinstance(dele, (list, tuple)) and
                len(ins) == N_RANDOM and len(dele) == N_RANDOM and all(_finite(v) for v in list(ins) + list(dele))):
            return na(f"random control requires exactly {N_RANDOM} finite AUCs per metric")
        ci, cd = float(np.mean(ins)), float(np.mean(dele))
    else:
        ci, cd = ctrl.get("auc_ins"), ctrl.get("auc_del")
        if not (_finite(ci) and _finite(cd)):
            return na("control AUC missing or non-finite")
    return {"status": "available", "control_auc_ins": ci, "control_auc_del": cd,
            "insertion_advantage": expl["auc_ins"] - ci, "deletion_advantage": cd - expl["auc_del"]}


def resolve_mismatch(recipient_key, assignment, donor_maps, donor_eligible):
    """Map for the recipient (composite key) from its PLANNED donor, or unavailable. Never re-selects
    a donor. donor_maps / donor_eligible: dicts keyed by donor record_uid (same explainer/condition)."""
    a = assignment[recipient_key]
    if a["status"] != "assigned":
        return None, {"status": "unavailable", "reason": a["reason"], "donor": None}
    d = a["donor"]
    if not donor_eligible.get(d, False):
        return None, {"status": "unavailable", "reason": "planned donor map not eligible", "donor": d}
    return donor_maps[d], {"status": "available", "reason": None, "donor": d}
