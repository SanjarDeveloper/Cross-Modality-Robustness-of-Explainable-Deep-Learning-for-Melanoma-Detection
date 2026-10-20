"""Stage 8B: RQ2 prediction collection (HAM TEST; 5 factors x 9 severities; seed-0 model; Stage 4A
threshold; perturb_msim_v1). Phases (separate processes): spec -> pilot -> sweep -> verify.

Each (factor, severity) unit: all 1548 HAM TEST images in manifest order, each perturbed from the ORIGINAL
cached 256 image, batch 16 (frozen membership = Stage 4A batches), xai_precision_v1. Units are published
atomically (tmp dir + manifest + rename) and verified by hash on resume.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq2_predict --config ../configs/rq2_predict_8b_v1.json --phase spec --out-root ../runs/rq2_predict_8b
  ... --phase pilot|sweep|verify --run-dir <dir>
"""
import argparse
import csv
import datetime
import io
import json
import math
import os
import time
import uuid

FACTORS = ("blur", "illumination", "compression", "scale", "nuisance")
SEVERITIES = tuple(i / 8 for i in range(9))
BATCH = 16
MODEL_SHA = "303d21e71986dc8f26dec84c6822a301f696060e5f08f613c90fc33c41a003dd"
BASELINE_TOL = 1e-6
AUDIT_N = 4


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def plan_units():
    return [{"unit": f"{f}__{s:.3f}", "factor": f, "severity": s} for f in FACTORS for s in SEVERITIES]


def composite_key(record_uid, factor, severity):
    return f"{record_uid}|{factor}|{severity:.3f}"


def batches(n, size=BATCH):
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def perturbed_inputs(originals, uids, factor, severity):
    """Perturb each ORIGINAL image independently (never chained); returns model inputs and metadata."""
    import numpy as np
    from . import perturb as PT
    xs, metas = [], []
    for img, u in zip(originals, uids):
        out, _, meta = PT.apply(img, factor, severity, u)
        xs.append(PT.to_model_input(out))
        metas.append(meta)
    return np.stack(xs), metas


def check_baseline(rows4a, rows_new, t, tol=BASELINE_TOL):
    """rows4a: Stage 4A prediction rows; rows_new: dicts with record_uid, label_mel, prob, pred."""
    a = {r["record_uid"]: r for r in rows4a}
    if len(a) != len(rows4a) or [r["record_uid"] for r in rows4a] != [r["record_uid"] for r in rows_new]:
        raise ValueError("record order/identity differs from Stage 4A")
    d = 0.0
    for r in rows_new:
        o = a[r["record_uid"]]
        if o["label_mel"] != str(r["label_mel"]):
            raise ValueError(f"label mismatch {r['record_uid']}")
        d = max(d, abs(float(o["prob"]) - float(r["prob"])))
        if int(r["pred"]) != int(float(r["prob"]) >= t) or int(o["pred"]) != int(r["pred"]):
            raise ValueError(f"prediction mismatch {r['record_uid']}")
    if d > tol:
        raise ValueError(f"probability difference {d} > {tol}")
    return d


def bind_threshold(thr, model_sha=MODEL_SHA):
    """The Stage 4A threshold may only be used with the model it was estimated for."""
    if thr["model"]["sha256"] != model_sha or thr.get("status") != "ESTIMATED":
        raise ValueError("threshold not bound to this model")
    return thr["threshold"]


def _paths(cfg):
    from . import config as C
    return {k: C.resolve(cfg, v) for k, v in cfg["supporting_inputs"].items()}


def phase_spec(cfg, a):
    from . import config as C, decode as DC, provenance as P
    from .desc_4b3 import publish_bytes, publish_json
    p = _paths(cfg)
    spec8a = json.load(open(p["perturb_spec"], encoding="utf-8"))
    if P.sha256_file(os.path.join(os.path.dirname(__file__), "perturb.py")) != spec8a["module_sha256"]:
        raise RuntimeError("perturb.py differs from the frozen Stage 8A specification")
    thr = json.load(open(p["threshold"], encoding="utf-8"))
    rows = [r for r in DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0] if r["split"] == "test"]
    counts = {"images": len(rows), "melanoma": sum(r["label_mel"] == "1" for r in rows), "lesions": len({r["cluster_id"] for r in rows})}
    if counts != {"images": 1548, "melanoma": 170, "lesions": 1155}:
        raise RuntimeError(f"HAM TEST counts {counts}")
    srcs = {n: P.sha256_file(os.path.join(os.path.dirname(__file__), n)) for n in
            ("perturb.py", "rq2_predict.py", "preprocess.py", "seeds.py", "xai_attr.py", "xai_prod.py")}
    spec = {"spec_version": "rq2_predict_8b_v1", "created_utc": _now(),
            "inputs": {k: {"path": v, "sha256": P.sha256_file(C.resolve(cfg, v))} for k, v in cfg["supporting_inputs"].items()},
            "manifest": {"path": cfg["manifests"]["ham10000"], "sha256": P.sha256_file(C.resolve(cfg, cfg["manifests"]["ham10000"]))},
            "sources": srcs, "model_sha256": MODEL_SHA, "threshold": {"value": bind_threshold(thr), "repr": thr["threshold_repr"]},
            "simulator": {"version": spec8a["version"], "module_sha256": spec8a["module_sha256"]},
            "cohort": dict(counts, order="manifest order of split=='test'", cluster="cluster_id (lesion)"),
            "grid": {"factors": FACTORS, "severities": SEVERITIES, "seed_rule": "stable_seed(0, image_uid, factor)",
                     "units": 45, "logical_rows": 45 * 1548,
                     "baseline": "severity 0 computed and stored for every factor (5 explicit copies), each checked against Stage 4A"},
            "pipeline": "cache ham_test 256 uint8 -> perturb.apply(original image; never chained) -> preprocess.to_model_input -> "
                        "seed-0 model, xai_precision_v1, batch 16 in manifest order (97 batches, final partial batch 12 = Stage 4A)",
            "baseline_tolerance": BASELINE_TOL, "expected_baseline_counts": {"TP": 157, "FN": 13, "TN": 911, "FP": 467},
            "outputs": "unit dirs u_<factor>__<sev>/ with predictions.csv, audit_images.npz (first 4 records), chunk_manifest.json; "
                       "published by atomic rename; verified by hash on resume; .tmp dirs never complete",
            "bundled_factors": {"scale": "zoom-out also changes position, synthetic background and inset seams",
                                "illumination": "brightness, contrast and colour temperature together",
                                "nuisance": "glare and hair together",
                                "note": "results cannot isolate individual components"},
            "no_changes_after_predictions": "simulator, threshold, model and grid are frozen"}
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "spec.json", spec)
    publish_bytes(out, "spec.json.sha256", f"{P.sha256_file(os.path.join(out, 'spec.json'))}  spec.json\n".encode())
    receipt["outputs"] = {"spec.json": P.sha256_file(os.path.join(out, "spec.json"))}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out)


def _load_spec(cfg, out):
    from . import config as C, provenance as P
    sp = os.path.join(out, "spec.json")
    if open(sp + ".sha256", encoding="utf-8").read().split()[0] != P.sha256_file(sp):
        raise RuntimeError("spec hash mismatch")
    spec = json.load(open(sp, encoding="utf-8"))
    for k, v in spec["inputs"].items():
        if P.sha256_file(C.resolve(cfg, v["path"])) != v["sha256"]:
            raise RuntimeError(f"input changed {k}")
    for n, h in spec["sources"].items():
        if P.sha256_file(os.path.join(os.path.dirname(__file__), n)) != h:
            raise RuntimeError(f"source changed {n} (resume refused)")
    return spec, P.sha256_file(sp)


def _model(cfg):
    import timm
    from . import trainer as TR
    st = TR._load_verified(_paths(cfg)["model_checkpoint"], MODEL_SHA)
    m = timm.create_model("resnet50", pretrained=False, num_classes=1)
    m.load_state_dict(st["model"], strict=True)
    return m.to("cuda").eval()


def _infer(model, x):
    import torch
    from . import xai_attr as XA
    with XA.xai_precision(), torch.no_grad():
        lg = model(torch.from_numpy(x).to("cuda")).squeeze(1).float()
        return lg.cpu().tolist(), torch.sigmoid(lg).cpu().tolist()


def phase_pilot(cfg, a):
    import numpy as np
    import torch
    from . import config as C, decode as DC, preprocess as PP
    from .desc_4b3 import publish_json
    from .seeds import stable_seed
    out = os.path.abspath(a.run_dir)
    _load_spec(cfg, out)
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    tr = sorted((r for r in rows if r["split"] == "train"), key=lambda r: (stable_seed("rq2_pilot_v1", r["record_uid"]), r["record_uid"]))
    sample = [r for r in tr if r["label_mel"] == "1"][:8] + [r for r in tr if r["label_mel"] == "0"][:8]
    reader = PP.EvalReader(rows, "ham_train", os.path.join(cfg["project_root"], "cache"))
    origs = [np.asarray(reader.img[int(r["cache_index"])]).copy() for r in sample]
    uids = [r["image_uid"] for r in sample]
    model = _model(cfg)
    torch.cuda.reset_peak_memory_stats()
    T = {"transform": {}, "inference": {}, "write": {}}
    for f in FACTORS:
        for s in (0.5, 1.0):
            t0 = time.perf_counter()
            x, metas = perturbed_inputs(origs, uids, f, s)
            T["transform"][f"{f}@{s}"] = time.perf_counter() - t0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _infer(model, x)
            torch.cuda.synchronize()
            T["inference"][f"{f}@{s}"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            buf = io.StringIO()
            csv.writer(buf).writerows([[u, json.dumps(m["params"])] for u, m in zip(uids, metas)])
            T["write"][f"{f}@{s}"] = time.perf_counter() - t0
    n = len(sample)
    per_img = {f: (sum(T["transform"][f"{f}@{s}"] for s in (0.5, 1.0)) / (2 * n)) for f in FACTORS}
    inf_per_img = sum(T["inference"].values()) / (10 * n)
    est = sum(per_img[f] * 1548 * 9 for f in FACTORS) + inf_per_img * 1548 * 45
    import shutil
    pilot = {"label": "runtime pilot on 16 HAM TRAIN images (8 mel, 8 non-mel; stable-seed order); no metrics computed",
             "sample": [r["record_uid"] for r in sample], "timing_s": T, "transform_s_per_image": per_img,
             "inference_s_per_image_batch16": inf_per_img,
             "peak_cuda_mib": [round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)],
             "ESTIMATE_full_sweep_s": est, "estimate_note": "estimate from pilot work (transform + inference); excludes CSV I/O and "
                                                             "model load; not a guarantee",
             "disk_free_bytes": shutil.disk_usage(out).free, "estimated_output_bytes": 45 * 1548 * 600}
    publish_json(out, "pilot.json", pilot)
    print(json.dumps({k: pilot[k] for k in ("transform_s_per_image", "inference_s_per_image_batch16", "peak_cuda_mib",
                                             "ESTIMATE_full_sweep_s", "disk_free_bytes")}, indent=1))


def phase_sweep(cfg, a):
    import numpy as np
    import torch
    from . import config as C, decode as DC, preprocess as PP, threshold as TH
    from .xai_prod import publish_dir, quarantine_tmp, verify_dir
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load_spec(cfg, out)
    q = quarantine_tmp(out)
    if q:
        print("quarantined", q, flush=True)
    p = _paths(cfg)
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    test = [r for r in rows if r["split"] == "test"]
    reader = PP.EvalReader(rows, "ham_test", os.path.join(cfg["project_root"], "cache"))
    origs = [np.asarray(reader.img[int(r["cache_index"])]).copy() for r in test]
    orig_digest = [__import__("hashlib").sha256(o.tobytes()).hexdigest() for o in origs]
    t = spec["threshold"]["value"]
    th_sha = spec["inputs"]["threshold"]["sha256"]
    rows4a = list(csv.DictReader(open(p["stage4a_ham_test_predictions"], encoding="utf-8", newline="")))
    model = _model(cfg)
    log = os.path.join(out, "progress.jsonl")
    for u in plan_units():
        final = os.path.join(out, f"u_{u['unit']}")
        if verify_dir(final, ssha):
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        T = {"transform": 0.0, "inference": 0.0, "write": 0.0}
        recs, audit = [], {}
        for (b0, b1) in batches(len(test)):
            tt = time.perf_counter()
            x, metas = perturbed_inputs(origs[b0:b1], [r["image_uid"] for r in test[b0:b1]], u["factor"], u["severity"])
            T["transform"] += time.perf_counter() - tt
            torch.cuda.synchronize()
            tt = time.perf_counter()
            lg, pr = _infer(model, x)
            torch.cuda.synchronize()
            T["inference"] += time.perf_counter() - tt
            for r, l, pp_, m in zip(test[b0:b1], lg, pr, metas):
                if not (math.isfinite(l) and 0.0 <= pp_ <= 1.0):
                    raise RuntimeError(f"invalid output {r['record_uid']}")
                recs.append({"record_uid": r["record_uid"], "image_uid": r["image_uid"], "cluster_id": r["cluster_id"],
                             "label_mel": r["label_mel"], "factor": u["factor"], "severity": f"{u['severity']:.3f}",
                             "key": composite_key(r["record_uid"], u["factor"], u["severity"]),
                             "perturb_seed": m["perturb_seed"], "params": json.dumps(m["params"], sort_keys=True),
                             "geometry_changed": m["geometry_changed"], "logit": repr(l), "prob": repr(pp_),
                             "pred": TH.predict(pp_, t), "threshold": repr(t), "threshold_sha256": th_sha,
                             "checkpoint_sha256": MODEL_SHA, "simulator": spec["simulator"]["version"],
                             "simulator_sha256": spec["simulator"]["module_sha256"], "prep": "prep_v1_cached256",
                             "precision": "xai_precision_v1", "batch_index": b0 // BATCH})
            if b0 == 0:
                from . import perturb as PT
                for i in range(AUDIT_N):
                    audit[test[i]["record_uid"]] = PT.apply(origs[i], u["factor"], u["severity"], test[i]["image_uid"])[0]
        if [__import__("hashlib").sha256(o.tobytes()).hexdigest() for o in origs[:AUDIT_N]] != orig_digest[:AUDIT_N]:
            raise RuntimeError("original image mutated")
        base = None
        if u["severity"] == 0.0:
            base = check_baseline(rows4a, recs, t)
            c = TH.counts([float(r["prob"]) for r in recs], [int(r["label_mel"]) for r in recs], t)
            if (c["TP"], c["FN"], c["TN"], c["FP"]) != (157, 13, 911, 467):
                raise RuntimeError(f"baseline counts {c}")
        tt = time.perf_counter()
        tmp = f"{final}.tmp-{uuid.uuid4().hex[:8]}"
        os.makedirs(tmp)
        with open(os.path.join(tmp, "predictions.csv"), "x", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(recs)
        buf = io.BytesIO()
        np.savez_compressed(buf, record_uids=np.array(list(audit)), images=np.stack(list(audit.values())))
        with open(os.path.join(tmp, "audit_images.npz"), "xb") as f:
            f.write(buf.getvalue())
        T["write"] = time.perf_counter() - tt
        meta = {"spec_sha256": ssha, "unit": u, "n": len(recs), "timing_s": T, "seconds": time.perf_counter() - t0,
                "baseline_max_abs_prob_diff_vs_stage4a": base,
                "peak_cuda_mib": [round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)]}
        publish_dir(tmp, final, meta)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"unit": u["unit"], "seconds": round(meta["seconds"], 1), "utc": _now()}) + "\n")
        print(f"unit {u['unit']} done {meta['seconds']:.1f}s baseline_diff={base}", flush=True)
    print("SWEEP STATUS", sum(verify_dir(os.path.join(out, f"u_{u['unit']}"), ssha) for u in plan_units()), "/ 45", flush=True)


def phase_verify(cfg, a):
    from . import config as C, decode as DC, provenance as P
    from .desc_4b3 import publish_json
    from .xai_prod import verify_dir
    out = os.path.abspath(a.run_dir)
    spec, ssha = _load_spec(cfg, out)
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    test = {r["record_uid"]: r for r in rows if r["split"] == "test"}
    t = spec["threshold"]["value"]
    keys, n, tim, peak, base_probs = set(), 0, {"transform": 0, "inference": 0, "write": 0, "seconds": 0}, [0, 0], {}
    problems = []
    for u in plan_units():
        d = os.path.join(out, f"u_{u['unit']}")
        if not verify_dir(d, ssha):
            problems.append(f"missing {u['unit']}")
            continue
        m = json.load(open(os.path.join(d, "chunk_manifest.json"), encoding="utf-8"))
        for k in ("transform", "inference", "write"):
            tim[k] += m["timing_s"][k]
        tim["seconds"] += m["seconds"]
        peak = [max(peak[0], m["peak_cuda_mib"][0]), max(peak[1], m["peak_cuda_mib"][1])]
        rr = list(csv.DictReader(open(os.path.join(d, "predictions.csv"), encoding="utf-8", newline="")))
        if [r["record_uid"] for r in rr] != list(test):
            problems.append(f"order/coverage {u['unit']}")
        for r in rr:
            if r["key"] in keys:
                problems.append(f"duplicate {r['key']}")
            keys.add(r["key"])
            o = test[r["record_uid"]]
            pr = float(r["prob"])
            if r["label_mel"] != o["label_mel"] or r["cluster_id"] != o["cluster_id"] or not (0 <= pr <= 1) \
                    or not math.isfinite(float(r["logit"])) or int(r["pred"]) != int(pr >= t):
                problems.append(f"row check {r['key']}")
            n += 1
        if u["severity"] == 0.0:
            base_probs[u["factor"]] = [r["prob"] for r in rr]
    same = len({tuple(v) for v in base_probs.values()}) == 1
    res = {"logical_rows": n, "expected": 45 * 1548, "unique_keys": len(keys), "severity0_copies_identical": same,
           "problems": problems[:50], "n_problems": len(problems), "measured": dict(tim, peak_cuda_mib=peak),
           "output_bytes": sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs),
           "pass": not problems and n == 45 * 1548 and len(keys) == n and same}
    publish_json(out, "verification.json", res)
    receipt = P.build_receipt(cfg, a.config, "completed" if res["pass"] else "failed", details={"phase": "verify"})
    receipt["outputs"] = {f"u_{u['unit']}/chunk_manifest.json": P.sha256_file(os.path.join(out, f"u_{u['unit']}", "chunk_manifest.json"))
                          for u in plan_units()}
    receipt["outputs"].update({n_: P.sha256_file(os.path.join(out, n_)) for n_ in ("spec.json", "pilot.json", "verification.json")})
    rp = P.write_receipt(receipt, out, name="receipt_verify.json")
    print(json.dumps({k: v for k, v in res.items() if k != "problems"}, indent=1), "\nverify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "pilot", "sweep", "verify"))
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
    {"spec": phase_spec, "pilot": phase_pilot, "sweep": phase_sweep, "verify": phase_verify}[a.phase](cfg, a)


if __name__ == "__main__":
    main()
