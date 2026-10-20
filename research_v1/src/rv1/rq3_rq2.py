"""Stage 9C2: RQ2-style perturbation evaluation for the nine RQ3 standard models (HAM TEST, 5 factors x 9
severities, perturb_msim_v1). ResNet-50 s0 reuses the Stage 8B sweep; the other eight models are swept
here. Unit-major execution: each (factor, severity) unit is transformed ONCE from the original cached images
(seeds depend only on image_uid and factor) and scored by all eight models; per-model unit directories are
published atomically and verified by hash on resume.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq3_rq2 --config ../configs/rq3_rq2_9c2_v1.json --phase spec --out-root ../runs/rq3_rq2_9c2
  ... --phase sweep|analyze --run-dir <dir>
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
import time
import uuid

import numpy as np

from .rq2_predict import BATCH, FACTORS, SEVERITIES, batches, check_baseline, composite_key, perturbed_inputs, plan_units

B = 2000
AUDIT_N = 4
BASELINE_TOL = 1e-6
R9C1 = "research_v1/runs/rq3_eval_9c1/20261004T102700Z_031b3a03_940d15b8/"
R8B = "research_v1/runs/rq2_predict_8b/20261004T080538Z_1d94708a_3474fb69/"
R8C = "research_v1/runs/rq2_stats_8c/20261004T082234Z_cbc0aad2_2529e34b/"
REUSED = "resnet50_s0"


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _abs(cfg, rel):
    from . import config as C
    return C.resolve(cfg, rel)


def _read(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def digest(img):
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()


def bind(model_entry, registry):
    """Threshold for a model, bound to its checkpoint hash."""
    t = registry["thresholds"][model_entry["id"]]
    if t["model_sha256"] != model_entry["sha256"]:
        raise ValueError(f"{model_entry['id']}: threshold bound to another model")
    return t["threshold"]


def reference_path(model_entry):
    """Unperturbed HAM TEST predictions used as the model's own baseline reference."""
    if "ham_test" in model_entry["reuse"]:
        return model_entry["reuse"]["ham_test"]["path"]
    return R9C1 + f"pred_{model_entry['id']}_ham_test.csv"


def phase_spec(cfg, a):
    from . import provenance as P
    from .desc_4b3 import publish_bytes, publish_json
    s9 = _json(_abs(cfg, R9C1 + "spec.json"))
    reg = _json(_abs(cfg, R9C1 + "threshold_registry.json"))
    rec9 = _json(_abs(cfg, R9C1 + "receipt_analyze.json"))
    for n, h in rec9["outputs"].items():
        if P.sha256_file(_abs(cfg, R9C1 + n)) != h:
            raise RuntimeError(f"9C1 changed {n}")
    rec8 = _json(_abs(cfg, R8B + "receipt_verify.json"))
    for n, h in rec8["outputs"].items():
        if P.sha256_file(_abs(cfg, R8B + n)) != h:
            raise RuntimeError(f"8B changed {n}")
    s8 = _json(_abs(cfg, R8B + "spec.json"))
    models = []
    for e in s9["models"]:
        t = bind(e, reg)
        ref = reference_path(e)
        models.append({k: e[k] for k in ("id", "arch", "seed", "run", "checkpoint", "epoch", "sha256")} |
                      {"threshold": t, "reference_ham_test": {"path": ref, "sha256": P.sha256_file(_abs(cfg, ref))},
                       "sweep": "reused Stage 8B" if e["id"] == REUSED else "new (Stage 9C2)"})
    if models[0]["threshold"] != s8["threshold"]["value"]:
        raise RuntimeError("ResNet s0 threshold differs from Stage 8B")
    srcs = {n: P.sha256_file(os.path.join(os.path.dirname(__file__), n)) for n in ("perturb.py", "rq2_predict.py", "rq3_rq2.py",
                                                                                     "rq2_stats.py", "preprocess.py", "xai_prod.py")}
    spec = {"spec_version": "rq3_rq2_9c2_v1", "created_utc": _now(), "models": models,
            "threshold_registry": {"path": R9C1 + "threshold_registry.json", "sha256": P.sha256_file(_abs(cfg, R9C1 + "threshold_registry.json"))},
            "stage8b": {"run": R8B, "spec_sha256": P.sha256_file(_abs(cfg, R8B + "spec.json"))},
            "simulator": s8["simulator"], "grid": s8["grid"], "pipeline": s8["pipeline"], "sources": srcs,
            "manifest": {"path": cfg["manifests"]["ham10000"], "sha256": P.sha256_file(_abs(cfg, cfg["manifests"]["ham10000"]))},
            "baseline_tolerance": BASELINE_TOL,
            "perturbation_identity": "seed = stable_seed(0, image_uid, factor): independent of model; each unit transformed once "
                                     "and scored by all models; audit digests of the first 4 records per unit compared with Stage 8B",
            "coverage": {"per_model_rows": 69660, "new_models": 8, "new_rows": 8 * 69660, "all_rows_incl_reuse": 9 * 69660},
            "bootstrap": {"B": B, "seed": 20261050, "draws": "identical to Stage 8C (same seed, lesion order, procedure); shared "
                                                             "across all models, factors and levels"},
            "statistics": "Stage 8C definitions (rq2_stats): per-level metrics, worst-case loss incl. s=0 recomputed per "
                          "replicate, censored 10-pp crossing, within-model ranking shares with split ties"}
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "spec.json", spec)
    publish_bytes(out, "spec.json.sha256", f"{P.sha256_file(os.path.join(out, 'spec.json'))}  spec.json\n".encode())
    receipt["outputs"] = {"spec.json": P.sha256_file(os.path.join(out, "spec.json"))}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out)


def _load_spec(cfg, out):
    from . import provenance as P
    sp = os.path.join(out, "spec.json")
    with open(sp + ".sha256", encoding="utf-8") as f:
        if f.read().split()[0] != P.sha256_file(sp):
            raise RuntimeError("spec hash mismatch")
    spec = _json(sp)
    for n, h in spec["sources"].items():
        if P.sha256_file(os.path.join(os.path.dirname(__file__), n)) != h:
            raise RuntimeError(f"source changed {n} (resume refused)")
    for m in spec["models"]:
        if P.sha256_file(_abs(cfg, m["reference_ham_test"]["path"])) != m["reference_ham_test"]["sha256"]:
            raise RuntimeError(f"reference changed {m['id']}")
        if P.sha256_file(os.path.join(_abs(cfg, m["run"]), m["checkpoint"])) != m["sha256"]:
            raise RuntimeError(f"checkpoint changed {m['id']}")
    return spec, P.sha256_file(sp)


def _load_model(cfg, m):
    import timm
    from . import trainer as TR
    ck = TR._load_verified(os.path.join(_abs(cfg, m["run"]), m["checkpoint"]), m["sha256"])
    if ck["epoch_completed"] != m["epoch"] or ck["config"]["training"]["arch"] != m["arch"]:
        raise RuntimeError(f"{m['id']}: checkpoint identity")
    net = timm.create_model(m["arch"], pretrained=False, num_classes=1)
    net.load_state_dict(ck["model"], strict=True)
    return net.to("cuda").eval()


def phase_sweep(cfg, a):
    import torch
    from . import decode as DC, preprocess as PP, threshold as TH, xai_attr as XA
    from .xai_prod import publish_dir, quarantine_tmp, verify_dir
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load_spec(cfg, out)
    quarantine_tmp(out)
    rows = DC.read_manifest(_abs(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    test = [r for r in rows if r["split"] == "test"]
    reader = PP.EvalReader(rows, "ham_test", os.path.join(cfg["project_root"], "cache"))
    origs = [np.asarray(reader.img[int(r["cache_index"])]).copy() for r in test]
    new = [m for m in spec["models"] if m["id"] != REUSED]
    nets = {m["id"]: _load_model(cfg, m) for m in new}
    refs = {m["id"]: _read(_abs(cfg, m["reference_ham_test"]["path"])) for m in new}
    log = os.path.join(out, "progress.jsonl")
    for u in plan_units():
        dirs = {m["id"]: os.path.join(out, f"u_{m['id']}__{u['unit']}") for m in new}
        if all(verify_dir(d, ssha) for d in dirs.values()):
            continue
        t0 = time.perf_counter()
        a8 = np.load(_abs(cfg, R8B + f"u_{u['unit']}/audit_images.npz"))
        from . import perturb as PT
        audit = {}
        for i in range(AUDIT_N):
            img = PT.apply(origs[i], u["factor"], u["severity"], test[i]["image_uid"])[0]
            audit[test[i]["record_uid"]] = digest(img)
            if digest(a8["images"][i]) != audit[test[i]["record_uid"]] or str(a8["record_uids"][i]) != test[i]["record_uid"]:
                raise RuntimeError(f"perturbation differs from Stage 8B at {u['unit']}")
        recs = {m["id"]: [] for m in new}
        tt = {"transform": 0.0, "inference": 0.0}
        for (b0, b1) in batches(len(test)):
            ts = time.perf_counter()
            x, metas = perturbed_inputs(origs[b0:b1], [r["image_uid"] for r in test[b0:b1]], u["factor"], u["severity"])
            tt["transform"] += time.perf_counter() - ts
            xt = torch.from_numpy(x).to("cuda")
            ts = time.perf_counter()
            for m in new:
                with XA.xai_precision(), torch.no_grad():
                    lg = nets[m["id"]](xt).squeeze(1).float()
                    pr = torch.sigmoid(lg).cpu().tolist()
                    lg = lg.cpu().tolist()
                for r, l, p, me in zip(test[b0:b1], lg, pr, metas):
                    if not (math.isfinite(l) and 0.0 <= p <= 1.0):
                        raise RuntimeError(f"invalid output {m['id']} {r['record_uid']}")
                    recs[m["id"]].append({"record_uid": r["record_uid"], "cluster_id": r["cluster_id"], "label_mel": r["label_mel"],
                                          "model_id": m["id"], "factor": u["factor"], "severity": f"{u['severity']:.3f}",
                                          "key": composite_key(r["record_uid"], u["factor"], u["severity"]),
                                          "perturb_seed": me["perturb_seed"], "params": json.dumps(me["params"], sort_keys=True),
                                          "logit": repr(l), "prob": repr(p), "pred": TH.predict(p, m["threshold"]),
                                          "threshold": repr(m["threshold"]), "checkpoint_sha256": m["sha256"],
                                          "simulator_sha256": spec["simulator"]["module_sha256"], "batch_index": b0 // BATCH})
            torch.cuda.synchronize()
            tt["inference"] += time.perf_counter() - ts
        for m in new:
            if verify_dir(dirs[m["id"]], ssha):
                continue
            base = None
            if u["severity"] == 0.0:
                base = check_baseline(refs[m["id"]], recs[m["id"]], m["threshold"], BASELINE_TOL)
            tmp = f"{dirs[m['id']]}.tmp-{uuid.uuid4().hex[:8]}"
            os.makedirs(tmp)
            with open(os.path.join(tmp, "predictions.csv"), "x", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(recs[m["id"]][0]), lineterminator="\n")
                w.writeheader()
                w.writerows(recs[m["id"]])
            publish_dir(tmp, dirs[m["id"]], {"spec_sha256": ssha, "model": m["id"], "unit": u, "n": len(recs[m["id"]]),
                                             "audit_digests": audit, "baseline_max_abs_prob_diff": base})
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"unit": u["unit"], "seconds": round(time.perf_counter() - t0, 1), "timing": tt, "utc": _now()}) + "\n")
        print(f"unit {u['unit']} done {time.perf_counter() - t0:.1f}s", flush=True)
    done = sum(verify_dir(os.path.join(out, f"u_{m['id']}__{u['unit']}"), ssha) for m in new for u in plan_units())
    print("SWEEP STATUS", done, "/", len(new) * 45, flush=True)


def _model_units(cfg, out, m, ssha):
    from .xai_prod import verify_dir
    units = {}
    for u in plan_units():
        if m["id"] == REUSED:
            p = _abs(cfg, R8B + f"u_{u['unit']}/predictions.csv")
        else:
            d = os.path.join(out, f"u_{m['id']}__{u['unit']}")
            if not verify_dir(d, ssha):
                raise RuntimeError(f"incomplete {d}")
            p = os.path.join(d, "predictions.csv")
        units[(u["factor"], u["severity"])] = _read(p)
    return units


def phase_analyze(cfg, a):
    from . import provenance as P, rq2_stats as R2
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    t0 = time.perf_counter()
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load_spec(cfg, out)
    data, ref_ids = {}, None
    for m in spec["models"]:
        units = _model_units(cfg, out, m, ssha)
        arr = {}
        for k, rr in units.items():
            ident = [(r["record_uid"], r["label_mel"], r["cluster_id"]) for r in rr]
            if ref_ids is None:
                ref_ids = ident
            if ident != ref_ids or len({r["key"] for r in rr}) != len(rr):
                raise RuntimeError(f"{m['id']} {k}: identity/coverage")
            if any(r["checkpoint_sha256"] != m["sha256"] for r in rr):
                raise RuntimeError(f"{m['id']} {k}: checkpoint identity")
            p = np.array([float(r["prob"]) for r in rr])
            if any(int(r["pred"]) != int(float(r["prob"]) >= m["threshold"]) for r in rr):
                raise RuntimeError(f"{m['id']} {k}: pred rule")
            arr[k] = p
        base = [arr[(f, 0.0)] for f in FACTORS]
        if not all(np.array_equal(base[0], b) for b in base[1:]):
            raise RuntimeError(f"{m['id']}: severity-0 copies differ")
        data[m["id"]] = arr
    y = np.array([r[1] == "1" for r in ref_ids])
    clusters = [r[2] for r in ref_ids]
    ids, groups = R2.lesion_index(clusters)
    rng = np.random.Generator(np.random.PCG64(spec["bootstrap"]["seed"]))
    rep = {mid: {k: {"sens": np.full(B, np.nan), "auc": np.full(B, np.nan)} for k in arr} for mid, arr in data.items()}
    for b in range(B):
        idx = R2.draw_rows(groups, rng)
        yb = y[idx]
        for mid, arr in data.items():
            th = next(m["threshold"] for m in spec["models"] if m["id"] == mid)
            for k, p in arr.items():
                if k[1] == 0.0 and k[0] != "blur":
                    continue
                pb = p[idx]
                rep[mid][k]["sens"][b] = (pb[yb] >= th).mean() if yb.any() else np.nan
                au = R2.auc_ranks(pb, yb)
                rep[mid][k]["auc"][b] = np.nan if au is None else au
        if b == 0:
            pass
    for mid in rep:
        for f in FACTORS[1:]:
            rep[mid][(f, 0.0)] = rep[mid][("blur", 0.0)]
    q = lambda v: np.quantile(v[np.isfinite(v)], [0.025, 0.975], method="linear").tolist()
    level_rows, wc_rows, cr_rows, rk_rows = [], [], [], []
    for m in spec["models"]:
        mid, th = m["id"], m["threshold"]
        pt = {}
        for k, p in data[mid].items():
            pt[k] = {"TP": int((p[y] >= th).sum()), "FN": int((p[y] < th).sum()), "TN": int((p[~y] < th).sum()),
                     "FP": int((p[~y] >= th).sum()), "sens": float((p[y] >= th).mean()), "spec": float((p[~y] < th).mean()),
                     "auc": R2.auc_ranks(p, y)}
        for f in FACTORS:
            for s in SEVERITIES:
                k = (f, s)
                ds = (rep[mid][k]["sens"] - rep[mid][(f, 0.0)]["sens"]) * 100
                da = rep[mid][k]["auc"] - rep[mid][(f, 0.0)]["auc"]
                level_rows.append({"model": mid, "arch": m["arch"], "seed": m["seed"], "factor": f, "severity": s,
                                   **{kk: pt[k][kk] for kk in ("TP", "FN", "TN", "FP")}, "sensitivity": pt[k]["sens"],
                                   "specificity": pt[k]["spec"], "auroc": pt[k]["auc"],
                                   "d_sens_pp": (pt[k]["sens"] - pt[(f, 0.0)]["sens"]) * 100, "d_sens_ci": json.dumps(q(ds)),
                                   "d_auroc": pt[k]["auc"] - pt[(f, 0.0)]["auc"], "d_auroc_ci": json.dumps(q(da)),
                                   "undefined": int(np.isnan(da).sum() + np.isnan(ds).sum())})
        wcr = {"sens": {}, "auc": {}}
        for f in FACTORS:
            for met in ("sens", "auc"):
                obs, lvl = R2.worst_case([pt[(f, s)][met] for s in SEVERITIES])
                arrw = np.array([(lambda w: np.nan if w is None else w)(R2.worst_case(
                    [None if np.isnan(rep[mid][(f, s)][met][b]) else float(rep[mid][(f, s)][met][b]) for s in SEVERITIES])[0])
                    for b in range(B)])
                wcr[met][f] = arrw
                sc = 100 if met == "sens" else 1
                wc_rows.append({"model": mid, "arch": m["arch"], "seed": m["seed"], "factor": f, "metric": met,
                                "worst_loss": obs * sc, "worst_level": lvl, "ci_lo": q(arrw)[0] * sc, "ci_hi": q(arrw)[1] * sc,
                                "undefined": int(np.isnan(arrw).sum())})
            cv = []
            und = 0
            for b in range(B):
                v = R2.crossing([None if np.isnan(rep[mid][(f, s)]["sens"][b]) else float(rep[mid][(f, s)]["sens"][b]) for s in SEVERITIES])
                if v is None:
                    und += 1
                else:
                    cv.append(v)
            cq = R2.censored_quantiles(cv)
            cr_rows.append({"model": mid, "arch": m["arch"], "seed": m["seed"], "factor": f,
                            "observed": R2.crossing([pt[(f, s)]["sens"] for s in SEVERITIES]), "valid": len(cv), "undefined": und,
                            "no_crossing_share": sum(v == R2.CENSORED for v in cv) / len(cv), "q025": cq[0.025], "q50": cq[0.5],
                            "q975": cq[0.975]})
        for met in ("sens", "auc"):
            rs = R2.ranking_shares([{f: (None if np.isnan(wcr[met][f][b]) else float(wcr[met][f][b])) for f in FACTORS} for b in range(B)])
            for f in FACTORS:
                rk_rows.append({"model": mid, "metric": met, "factor": f, "share": rs["shares"][f], "valid": rs["valid"],
                                "ties": rs["tie_replicates"]})
    # reproduce Stage 8C for ResNet s0
    c8 = {(r["factor"], float(r["severity"])): r for r in _read(_abs(cfg, R8C + "per_level_table.csv"))}
    w8 = {(r["factor"], r["metric"]): r for r in _read(_abs(cfg, R8C + "worst_case_table.csv"))}
    chk = {"max_abs_point_diff": 0.0, "max_abs_ci_diff": 0.0}
    for r in level_rows:
        if r["model"] != REUSED:
            continue
        o = c8[(r["factor"], r["severity"])]
        chk["max_abs_point_diff"] = max(chk["max_abs_point_diff"], abs(float(o["d_sens_pp"]) - r["d_sens_pp"]),
                                        abs(float(o["d_auroc"]) - r["d_auroc"]))
        lo, hi = json.loads(r["d_sens_ci"])
        chk["max_abs_ci_diff"] = max(chk["max_abs_ci_diff"], abs(float(o["d_sens_ci_lo"]) - lo), abs(float(o["d_sens_ci_hi"]) - hi))
    for r in wc_rows:
        if r["model"] == REUSED:
            o = w8[(r["factor"], "sensitivity_pp" if r["metric"] == "sens" else "auroc")]
            chk["max_abs_point_diff"] = max(chk["max_abs_point_diff"], abs(float(o["worst_loss"]) - r["worst_loss"]))
            chk["max_abs_ci_diff"] = max(chk["max_abs_ci_diff"], abs(float(o["ci_lo"]) - r["ci_lo"]), abs(float(o["ci_hi"]) - r["ci_hi"]))
    if chk["max_abs_point_diff"] > 1e-9:
        raise RuntimeError(f"ResNet s0 does not reproduce Stage 8C: {chk}")
    ranges = []
    for arch in ("resnet50", "efficientnet_b3", "densenet121"):
        for f in FACTORS:
            for met in ("sens", "auc"):
                v = [r["worst_loss"] for r in wc_rows if r["arch"] == arch and r["factor"] == f and r["metric"] == met]
                ranges.append({"arch": arch, "factor": f, "metric": "worst_loss_" + ("sens_pp" if met == "sens" else "auroc"),
                               "s0": v[0], "s1": v[1], "s2": v[2], "min": min(v), "max": max(v), "note": "seed range, not a CI"})
    publish_json(out, "results.json", {"stage8c_reproduction": chk, "runtime_s": time.perf_counter() - t0})
    publish_csv(out, "per_level_table.csv", level_rows, list(level_rows[0]))
    publish_csv(out, "worst_case_table.csv", wc_rows, list(wc_rows[0]))
    publish_csv(out, "crossing_table.csv", cr_rows, list(cr_rows[0]))
    publish_csv(out, "ranking_table.csv", rk_rows, list(rk_rows[0]))
    publish_csv(out, "seed_ranges.csv", ranges, list(ranges[0]))
    for f in FACTORS:
        for key, lab in (("d_sens_pp", "sensitivity change (pp)"), ("d_auroc", "AUROC change")):
            publish_bytes(out, f"plot_{f}_{key}.png", _plot(level_rows, f, key, f"{f}: {lab} from each model's own baseline"))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))
                          if os.path.isfile(os.path.join(out, n)) and not n.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    print(chk, "runtime", round(time.perf_counter() - t0, 1), "verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def _plot(rows, factor, key, title):
    from PIL import Image, ImageDraw
    rr = [r for r in rows if r["factor"] == factor]
    W, H, L, R, T, Bm = 900, 480, 70, 190, 40, 50
    vals = [r[key] for r in rr] + [0.0]
    ymin, ymax = min(vals), max(vals)
    pad = (ymax - ymin) * 0.08 or 1
    ymin, ymax = ymin - pad, ymax + pad
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    X = lambda s: L + s * (W - L - R)
    Y = lambda v: T + (ymax - v) / (ymax - ymin) * (H - T - Bm)
    d.rectangle([L, T, W - R, H - Bm], outline="black")
    d.line([L, Y(0), W - R, Y(0)], fill=(160, 160, 160))
    for s in SEVERITIES:
        d.text((X(s) - 8, H - Bm + 4), f"{s:g}", fill="black")
    for k in range(6):
        v = ymin + k * (ymax - ymin) / 5
        d.text((4, Y(v) - 6), f"{v:.3g}", fill="black")
    d.text((L, 12), title + " (point estimates; seeds shown separately)", fill="black")
    cols = {"resnet50": (31, 119, 180), "efficientnet_b3": (214, 39, 40), "densenet121": (44, 160, 44)}
    dashes = {0: 0, 1: 1, 2: 2}
    j = 0
    for mid in sorted({r["model"] for r in rr}):
        mr = [r for r in rr if r["model"] == mid]
        col = cols[mr[0]["arch"]]
        shade = tuple(min(255, c + 60 * mr[0]["seed"]) for c in col)
        pts = [(X(r["severity"]), Y(r[key])) for r in mr]
        d.line(pts, fill=shade, width=2)
        for x, yv in pts:
            d.ellipse([x - 2, yv - 2, x + 2, yv + 2], fill=shade)
        d.rectangle([W - R + 10, T + 18 * j, W - R + 24, T + 18 * j + 10], fill=shade)
        d.text((W - R + 30, T + 18 * j - 1), mid, fill="black")
        j += 1
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "sweep", "analyze"))
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
    {"spec": phase_spec, "sweep": phase_sweep, "analyze": phase_analyze}[a.phase](cfg, a)


if __name__ == "__main__":
    main()
