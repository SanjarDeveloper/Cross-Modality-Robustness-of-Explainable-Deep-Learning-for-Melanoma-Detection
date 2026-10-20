"""Stage 9G1: unperturbed evaluation of the nine aug_heavy_v1 cells and frozen descriptive standard-vs-heavy
contrasts. Phases (separate processes): spec -> threshold -> infer -> analyze -> compare.

spec    : new (heavy registry, verified against file/sidecar/best.json/embedded config/Stage 9F card; standard
          references = the verified Stage 9C1 run). Written in the Stage 9C1 spec format so that the threshold,
          infer and analyze phases are the UNCHANGED, tested rq3_eval phases (same bootstrap seeds -> common draws).
compare : new. Matched (arch, seed) heavy-minus-standard contrasts; within each replicate the SAME clusters
          (HAM lesions / PAD patients / Derm7pt pairs) are resampled for both models; Derm7pt gaps keep both
          modalities and both models together. Intervals condition on fitted models and thresholds.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq3_heavy_eval --config ../configs/rq3_heavy_eval_9g1_v1.json --phase spec --out-root ../runs/rq3_heavy_eval_9g1
  ... --phase threshold|infer|analyze|compare --run-dir <dir>
"""
import argparse
import json
import os
import time

import numpy as np

from . import rq3_eval as E

ARCHES = (("resnet50", "r50"), ("efficientnet_b3", "effb3"), ("densenet121", "dn121"))
STD_9C1 = "research_v1/runs/rq3_eval_9c1/20261004T102700Z_031b3a03_940d15b8"
CARD_9F = "research_v1/runs/stage9f/stage9f_summary.json"
KNOWN_9D_SOURCE_CHANGE = "source: hash mismatch research_v1/src/rv1/augment.py"  # Stage 9D aug_heavy_v1 addition (prior copy runs/stage9d/augment_before.py.txt)
CONTRAST_SEED_OFFSET = 100      # contrast seeds = bootstrap_seed + 100 + k (frozen in spec)


def heavy_models():
    return [{"id": f"{arch}_heavy_s{s}", "arch": arch, "seed": s, "strategy": "aug_heavy_v1",
             "run": f"research_v1/runs/main_train/rq3h_{sh}_s{s}_9f", "card_key": f"{sh}_s{s}",
             "standard_id": f"{arch}_s{s}", "threshold_src": None, "reuse": {}}
            for arch, sh in ARCHES for s in (0, 1, 2)]


def match(heavy, standard):
    """Map each heavy entry to the standard entry with the same arch and seed (exactly one)."""
    out = {}
    for h in heavy:
        m = [s for s in standard if s["arch"] == h["arch"] and s["seed"] == h["seed"]]
        if len(m) != 1 or m[0]["id"] != h["standard_id"]:
            raise RuntimeError(f"{h['id']}: no unique standard match")
        out[h["id"]] = m[0]
    return out


def check_strategy(ck, strategy):
    if ck["config"]["versions"]["aug_version"] != strategy:
        raise RuntimeError("checkpoint strategy mismatch")
    return True


# ------------------------------------------------------------------ paired statistics (pure)
def _bin_metrics(y, p, t):
    from . import rq2_stats as R2
    y, p = np.asarray(y, bool), np.asarray(p, float)
    if y.all() or not y.any():
        return None
    return float((p[y] >= t).mean()), float((p[~y] < t).mean()), R2.auc_ranks(p, y)


def binary_contrast(cases, ts, th):
    """cases: list of {y, ps, ph}. Returns (sens_s, sens_h, d_sens, spec_s, spec_h, d_spec, auc_s, auc_h, d_auc) or None."""
    y = [c["y"] for c in cases]
    a = _bin_metrics(y, [c["ps"] for c in cases], ts)
    b = _bin_metrics(y, [c["ph"] for c in cases], th)
    if a is None or b is None:
        return None
    return a[0], b[0], b[0] - a[0], a[1], b[1], b[1] - a[1], a[2], b[2], b[2] - a[2]


BIN_NAMES = ("sens_std", "sens_heavy", "d_sens", "spec_std", "spec_heavy", "d_spec", "auroc_std", "auroc_heavy", "d_auroc")
GAP_NAMES = ("sens_derm_std", "sens_clin_std", "gap_sens_std", "sens_derm_heavy", "sens_clin_heavy", "gap_sens_heavy",
             "d_gap_sens", "d_sens_derm", "d_sens_clin",
             "spec_derm_std", "spec_clin_std", "spec_derm_heavy", "spec_clin_heavy",
             "auroc_derm_std", "auroc_clin_std", "gap_auroc_std", "auroc_derm_heavy", "auroc_clin_heavy", "gap_auroc_heavy",
             "d_gap_auroc", "d_auroc_derm", "d_auroc_clin")


def gap_contrast(pairs):
    """pairs: list of {label, s: {pred_d, pred_c, prob_d, prob_c}, h: {...}}; both modalities and both models stay
    together. gap = dermoscopy - clinical; d_gap = gap_heavy - gap_standard. None if a class is absent."""
    from . import rq2_stats as R2
    pos = [q for q in pairs if q["label"] == 1]
    neg = [q for q in pairs if q["label"] == 0]
    if not pos or not neg:
        return None
    y = np.array([q["label"] == 1 for q in pairs])

    def side(k):
        sd = sum(q[k]["pred_d"] for q in pos) / len(pos)
        sc = sum(q[k]["pred_c"] for q in pos) / len(pos)
        pd_ = sum(1 - q[k]["pred_d"] for q in neg) / len(neg)
        pc = sum(1 - q[k]["pred_c"] for q in neg) / len(neg)
        ad = R2.auc_ranks([q[k]["prob_d"] for q in pairs], y)
        ac = R2.auc_ranks([q[k]["prob_c"] for q in pairs], y)
        return sd, sc, pd_, pc, ad, ac
    s, h = side("s"), side("h")
    gs, gh = s[0] - s[1], h[0] - h[1]
    ags, agh = s[4] - s[5], h[4] - h[5]
    return (s[0], s[1], gs, h[0], h[1], gh, gh - gs, h[0] - s[0], h[1] - s[1],
            s[2], s[3], h[2], h[3],
            s[4], s[5], ags, h[4], h[5], agh, agh - ags, h[4] - s[4], h[5] - s[5])


def join_binary(std_rows, heavy_rows, key):
    """Identifier join; every record must appear once in both with identical label and cluster."""
    hs = {r["record_uid"]: r for r in heavy_rows}
    if len(hs) != len(heavy_rows) or len({r["record_uid"] for r in std_rows}) != len(std_rows) or set(hs) != {r["record_uid"] for r in std_rows}:
        raise RuntimeError("incomplete or duplicated join")
    cl = {}
    for r in std_rows:
        h = hs[r["record_uid"]]
        if h["label_mel"] != r["label_mel"] or h[key] != r[key]:
            raise RuntimeError(f"{r['record_uid']}: label/cluster mismatch")
        cl.setdefault(r[key], []).append({"y": r["label_mel"] == "1", "ps": float(r["prob"]), "ph": float(h["prob"])})
    return cl


def join_pairs(std_pairs, heavy_pairs):
    if set(std_pairs) != set(heavy_pairs):
        raise RuntimeError("Derm7pt pair sets differ")
    out = {}
    for k, s in std_pairs.items():
        h = heavy_pairs[k]
        if s["label"] != h["label"] or s["primary"] != h["primary"] or (s["uid_d"], s["uid_c"]) != (h["uid_d"], h["uid_c"]):
            raise RuntimeError(f"pair {k}: mismatch")
        out[k] = {"label": s["label"], "primary": s["primary"], "s": s, "h": h}
    return out


def _summ(names, point, boot):
    ci = boot["ci"]
    return {n: {"point": point[i], "ci95": None if ci is None else ci[i]} for i, n in enumerate(names)} | {
        "_bootstrap": {"B": boot["B"], "seed": boot["seed"], "n_undefined": boot["n_undefined"], "unreliable": boot["unreliable"]}}


# ------------------------------------------------------------------ phases
def phase_spec(cfg, a):
    from . import provenance as P
    import torch
    with open(E._abs(cfg, CARD_9F), encoding="utf-8") as f:
        card = json.load(f)
    std_dir = E._abs(cfg, STD_9C1)
    receipt_notes = {}
    for n in ("receipt_spec.json", "receipt_threshold.json", "receipt_infer.json", "receipt_analyze.json"):
        probs = P.verify_receipt(os.path.join(std_dir, n), cfg["project_root"]) or []
        if [x for x in probs if x != KNOWN_9D_SOURCE_CHANGE]:
            raise RuntimeError(f"Stage 9C1 {n} does not verify: {probs}")
        with open(os.path.join(std_dir, n), encoding="utf-8") as f:
            for o, hsh in json.load(f).get("outputs", {}).items():
                if P.sha256_file(os.path.join(std_dir, o)) != hsh:
                    raise RuntimeError(f"Stage 9C1 output changed: {o}")
        receipt_notes[n] = probs
    std_spec = E._load_spec(cfg, std_dir)
    std_reg = E._hashed_json(std_dir, "threshold_registry.json")
    reg = []
    for m in heavy_models():
        run = E._abs(cfg, m["run"])
        with open(os.path.join(run, "best.json"), encoding="utf-8") as f:
            best = json.load(f)
        ck_path = os.path.join(run, best["checkpoint"])
        sha = P.sha256_file(ck_path)
        with open(ck_path + ".sha256", encoding="utf-8") as f:
            side = f.read().split()[0]
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        E.check_selected(best, sha, ck, m["arch"], m["seed"])
        check_strategy(ck, "aug_heavy_v1")
        c = card[m["card_key"]]
        if not (sha == side == c["selected_sha256"] and best["epoch"] == c["selected_epoch"] and c["ok"]):
            raise RuntimeError(f"{m['id']}: does not match Stage 9F card")
        reg.append(dict(m, checkpoint=best["checkpoint"], epoch=best["epoch"], sha256=sha, best_val_auroc=best["val_auroc"]))
    pairs = match(reg, std_spec["models"])
    s = cfg["seeds"]["bootstrap_seed"]
    spec = {"spec_version": "rq3_heavy_eval_9g1_v1", "created_utc": E._now(), "models": reg,
            "standard_reference": {"run_dir": STD_9C1, "spec_sha256": P.sha256_file(os.path.join(std_dir, "spec.json")),
                                   "threshold_registry_sha256": P.sha256_file(os.path.join(std_dir, "threshold_registry.json")),
                                   "receipt_verification": receipt_notes, "receipt_note": "only the documented Stage 9D augment.py change is tolerated; all receipt output hashes re-checked",
                                   "matches": {h: {"id": v["id"], "sha256": v["sha256"], "epoch": v["epoch"],
                                                   "threshold": std_reg["thresholds"][v["id"]]["threshold"]} for h, v in pairs.items()}},
            "manifests": {k: {"path": cfg["manifests"][k], "sha256": P.sha256_file(E._abs(cfg, cfg["manifests"][k]))}
                          for k in ("ham10000", "derm7pt", "pad_ufes_20")},
            "cohorts": {"ham_test": "HAM10000 split=test (1548; lesion clusters 1155)",
                        "derm7pt": "Derm7pt ALL 2022 rows / 1011 pairs; PRIMARY flag retained (1002 pairs)",
                        "pad_all": "PAD-UFES-20 all 2298 images (external evaluation of HAM-trained models; none is the PAD-adapted RQ5 model); rq5_role retained; patient clusters",
                        "excluded": "DDI, RQ2 sweeps, XAI (out of scope)"},
            "inference": {"prep": "prep_v1_cached256", "precision": "xai_precision_v1", "batch": 16, "order": "manifest order per cache"},
            "threshold_rule": "v1_highest_t_img_sens95 on full HAM VAL (val-list order), one new threshold per heavy checkpoint; standard thresholds not reused",
            "analyses": std_spec["analyses"],
            "bootstrap": dict(std_spec["bootstrap"], note="identical seeds to Stage 9C1 -> identical cluster draws for standard and heavy per-model intervals"),
            "contrasts": {"frozen_before_analysis": True,
                          "definition": "heavy minus standard, matched on (arch, seed); each model at its own threshold",
                          "ham_test": {"unit": "lesion cluster", "metrics": BIN_NAMES, "seed": s + CONTRAST_SEED_OFFSET},
                          "pad_all": {"unit": "patient cluster", "metrics": BIN_NAMES, "seed": s + CONTRAST_SEED_OFFSET + 1},
                          "derm7pt_PRIMARY": {"unit": "pair (both modalities, both models kept together)", "metrics": GAP_NAMES,
                                              "seed": s + CONTRAST_SEED_OFFSET + 2},
                          "derm7pt_ALL": {"unit": "pair", "metrics": GAP_NAMES, "seed": s + CONTRAST_SEED_OFFSET + 3,
                                          "role": "prespecified sensitivity analysis"},
                          "B": E.B, "ci": "percentile 2.5/97.5 (numpy linear)", "undefined": "counted, never redrawn",
                          "interpretation": "descriptive; conditional on the fitted models and thresholds; no training variability; "
                                            "not significance tests; no between-architecture tests; no combined p-values"},
            "no_tests_between_architectures": True, "primary_model": "resnet50_s0 standard (unchanged)"}
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    receipt["outputs"] = {"spec.json": E._hashed_json(out, "spec.json", spec)}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out)
    for e in reg:
        print(e["id"], e["checkpoint"], e["sha256"][:12], "->", pairs[e["id"]]["id"])


def phase_compare(cfg, a):
    from . import provenance as P, rq1_stats as RS
    from .desc_4b3 import publish_csv, publish_json
    t0 = time.perf_counter()
    out = os.path.abspath(a.run_dir)
    spec = E._load_spec(cfg, out)
    reg = E._hashed_json(out, "threshold_registry.json")
    with open(os.path.join(out, "receipt_analyze.json"), encoding="utf-8") as f:
        for n, h in json.load(f)["outputs"].items():
            if P.sha256_file(os.path.join(out, n)) != h:
                raise RuntimeError(f"changed {n}")
    std_dir = E._abs(cfg, STD_9C1)
    std_spec = E._load_spec(cfg, std_dir)
    std_reg = E._hashed_json(std_dir, "threshold_registry.json")
    if P.sha256_file(os.path.join(std_dir, "spec.json")) != spec["standard_reference"]["spec_sha256"]:
        raise RuntimeError("standard reference changed")
    std_by = {e["id"]: e for e in std_spec["models"]}
    reg_sha = P.sha256_file(os.path.join(out, "threshold_registry.json"))
    std_reg_sha = P.sha256_file(os.path.join(std_dir, "threshold_registry.json"))
    C = spec["contrasts"]
    results, rows = {}, []
    for h in spec["models"]:
        s = std_by[spec["standard_reference"]["matches"][h["id"]]["id"]]
        th, ts = reg["thresholds"][h["id"]]["threshold"], std_reg["thresholds"][s["id"]]["threshold"]
        r = {"heavy": h["id"], "standard": s["id"], "arch": h["arch"], "seed": h["seed"], "threshold_heavy": th, "threshold_std": ts,
             "heavy_sha256": h["sha256"], "std_sha256": s["sha256"]}
        for coh, key in (("ham_test", "cluster_id"), ("pad_all", "cluster_id")):
            cl = join_binary(E._rows_for(cfg, std_dir, s, coh, None), E._rows_for(cfg, out, h, coh, None), key)
            allc = [c for v in cl.values() for c in v]
            point = binary_contrast(allc, ts, th)
            b = RS.pair_bootstrap(cl, sorted(cl), lambda cs: binary_contrast([x for c in cs for x in c], ts, th), E.B, C[coh]["seed"])
            r[coh] = _summ(BIN_NAMES, point, b)
        ds = E._rows_for(cfg, std_dir, s, "derm7pt", None)
        dh = E._rows_for(cfg, out, h, "derm7pt", None)
        jp = join_pairs(RS.build_pairs(ds, ts, ds[0]["threshold_sha256"], s["sha256"]),
                        RS.build_pairs(dh, th, dh[0]["threshold_sha256"], h["sha256"]))
        for name in ("PRIMARY", "ALL"):
            coh = jp if name == "ALL" else {k: v for k, v in jp.items() if v["primary"]}
            point = gap_contrast(list(coh.values()))
            b = RS.pair_bootstrap(coh, sorted(coh), gap_contrast, E.B, C[f"derm7pt_{name}"]["seed"])
            r[f"derm7pt_{name}"] = _summ(GAP_NAMES, point, b) | {"_n_pairs": len(coh)}
        results[h["id"]] = r
        for coh in ("ham_test", "pad_all"):
            for m in ("sens", "spec", "auroc"):
                rows.append({"arch": h["arch"], "seed": h["seed"], "cohort": coh, "metric": m, "threshold_std": ts, "threshold_heavy": th,
                             "std": r[coh][f"{m}_std"]["point"], "heavy": r[coh][f"{m}_heavy"]["point"],
                             "diff_heavy_minus_std": r[coh][f"d_{m}"]["point"], "diff_ci95": json.dumps(r[coh][f"d_{m}"]["ci95"]),
                             "n_undefined": r[coh]["_bootstrap"]["n_undefined"]})
        for name in ("PRIMARY", "ALL"):
            blk = r[f"derm7pt_{name}"]
            for m in ("sens", "auroc"):
                rows.append({"arch": h["arch"], "seed": h["seed"], "cohort": f"derm7pt_{name}", "metric": f"{m}_dermoscopy",
                             "threshold_std": ts, "threshold_heavy": th, "std": blk[f"{m}_derm_std"]["point"], "heavy": blk[f"{m}_derm_heavy"]["point"],
                             "diff_heavy_minus_std": blk[f"d_{m}_derm"]["point"], "diff_ci95": json.dumps(blk[f"d_{m}_derm"]["ci95"]),
                             "n_undefined": blk["_bootstrap"]["n_undefined"]})
                rows.append({"arch": h["arch"], "seed": h["seed"], "cohort": f"derm7pt_{name}", "metric": f"{m}_clinical",
                             "threshold_std": ts, "threshold_heavy": th, "std": blk[f"{m}_clin_std"]["point"], "heavy": blk[f"{m}_clin_heavy"]["point"],
                             "diff_heavy_minus_std": blk[f"d_{m}_clin"]["point"], "diff_ci95": json.dumps(blk[f"d_{m}_clin"]["ci95"]),
                             "n_undefined": blk["_bootstrap"]["n_undefined"]})
                rows.append({"arch": h["arch"], "seed": h["seed"], "cohort": f"derm7pt_{name}", "metric": f"gap_{m}_derm_minus_clin",
                             "threshold_std": ts, "threshold_heavy": th, "std": blk[f"gap_{m}_std"]["point"], "heavy": blk[f"gap_{m}_heavy"]["point"],
                             "diff_heavy_minus_std": blk[f"d_gap_{m}"]["point"], "diff_ci95": json.dumps(blk[f"d_gap_{m}"]["ci95"]),
                             "n_undefined": blk["_bootstrap"]["n_undefined"]})
            for m, k in (("spec_dermoscopy", "derm"), ("spec_clinical", "clin")):
                rows.append({"arch": h["arch"], "seed": h["seed"], "cohort": f"derm7pt_{name}", "metric": m, "threshold_std": ts,
                             "threshold_heavy": th, "std": blk[f"spec_{k}_std"]["point"], "heavy": blk[f"spec_{k}_heavy"]["point"],
                             "diff_heavy_minus_std": blk[f"spec_{k}_heavy"]["point"] - blk[f"spec_{k}_std"]["point"], "diff_ci95": "",
                             "n_undefined": blk["_bootstrap"]["n_undefined"]})
        print(h["id"], {c: round(r[c]["d_auroc"]["point"], 3) for c in ("ham_test", "pad_all")},
              round(r["derm7pt_PRIMARY"]["d_gap_sens"]["point"], 3), flush=True)
    ranges = []
    keys = sorted({(x["cohort"], x["metric"]) for x in rows})
    for arch, _ in ARCHES:
        for coh, m in keys:
            v = sorted((x for x in rows if x["arch"] == arch and x["cohort"] == coh and x["metric"] == m), key=lambda x: x["seed"])
            for fld in ("std", "heavy", "diff_heavy_minus_std"):
                vals = [x[fld] for x in v]
                ranges.append({"arch": arch, "cohort": coh, "metric": m, "quantity": fld, "s0": vals[0], "s1": vals[1], "s2": vals[2],
                               "min": min(vals), "max": max(vals), "n_seeds_positive": sum(x > 0 for x in vals),
                               "note": "range across 3 seeds; not a CI"})
    publish_json(out, "contrast_results.json", {"contrasts": results, "spec_contrasts": C, "runtime_s": time.perf_counter() - t0,
                                                 "standard_threshold_registry_sha256": std_reg_sha, "heavy_threshold_registry_sha256": reg_sha})
    publish_csv(out, "matched_table.csv", rows, list(rows[0]))
    publish_csv(out, "matched_seed_ranges.csv", ranges, list(ranges[0]))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "compare"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("contrast_results.json", "matched_table.csv",
                                                                           "matched_seed_ranges.csv", "spec.json")}
    rp = P.write_receipt(receipt, out, name="receipt_compare.json")
    print("runtime", round(time.perf_counter() - t0, 1), "verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "threshold", "infer", "analyze", "compare"))
    ap.add_argument("--out-root")
    ap.add_argument("--run-dir")
    a = ap.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from . import config as C
    from .seeds import seed_everything
    cfg = C.load(a.config)
    if cfg["purpose"] != "predict_v1":
        raise C.ConfigError("requires purpose 'predict_v1'")
    seed_everything(0, cfg["determinism"])
    {"spec": phase_spec, "threshold": E.phase_threshold, "infer": E.phase_infer, "analyze": E.phase_analyze,
     "compare": phase_compare}[a.phase](cfg, a)


if __name__ == "__main__":
    main()
