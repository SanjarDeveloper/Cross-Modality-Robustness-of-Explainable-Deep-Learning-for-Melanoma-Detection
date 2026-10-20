"""Stage 10B.1: bounded IG readiness extension for the simulated-HAM arm (retrospective; motivated by the failed
Stage 10B pilot). Candidate m=2048, reference m=4096, on the identical Stage 10B pilot (20 images x 11 conditions,
same order, chunks, checkpoint, preprocessing, baseline, precision, internal batch). Grad-CAM, controls and
localisation are NOT recomputed (reused from Stage 10B).

Staged, gated computation:
  spec   freeze the extension (Stage 10B spec/run hashes, criteria and rules unchanged)
  gate1  IG m=2048 for all 220 references -> criteria (a) failure rate <= 5% and (b) p95 e_rel <= 0.20, pooled AND
         every condition. Any failure -> NOT_READY; stop (m=4096 is never computed).
  gate2  ONLY if gate1 passed: IG m=4096 on identical references/batches; curves for the m=2048 and m=4096 maps;
         full xai_pairs.assess_candidate(2048 vs 4096) pooled and per condition; Stage 10B readiness rule unchanged.
No automatic escalation beyond m=4096; m=4096 is a reference, not claimed converged.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_sim_ext --config ../configs/xai_sim_ext_10b1_v1.json --phase spec --out-root ../runs/xai_sim_ext_10b1
  ... --phase gate1|gate2 --run-dir <dir>
"""
import argparse
import io
import json
import os
import time
import uuid

import numpy as np

from . import xai_sim as XS

R10B = "research_v1/runs/xai_sim_10b/20261004T180422Z_ad7eef76_cdce001b"
M_CAND, M_REF = 2048, 4096


# ------------------------------------------------------------------ pure helpers (tested)
def assess_ab(at_m, refs):
    """Necessary criteria (a) completeness failure rate and (b) p95 e_rel only, with the frozen xai_pairs rules."""
    from . import xai_pairs as XPa
    A = XPa.assess_candidate(at_m, {}, refs)
    st = [A["a"]["status"], A["b"]["status"]]
    overall = "fail" if "fail" in st else ("UNRESOLVED" if "UNRESOLVED" in st else "pass")
    return {"n_refs": A["n_refs"], "a": A["a"], "b": A["b"], "overall": overall}


def gate1_decision(pooled, per_condition):
    """Same structure as the frozen Stage 10B rule: pooled AND every condition must pass; otherwise NOT_READY."""
    return XS.readiness(pooled, per_condition)


# ------------------------------------------------------------------ io
def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _abs(cfg, rel):
    from . import config as C
    return C.resolve(cfg, rel)


def phase_spec(cfg, a):
    from . import provenance as P, xai_pairs as XPa, xai_prod as XP
    from .desc_4b3 import publish_bytes, publish_json
    run = _abs(cfg, R10B)
    s10, s10sha = XP._load(run, "spec.json")
    if P.verify_receipt(os.path.join(run, "receipt_analyze.json"), cfg["project_root"]):
        raise RuntimeError("Stage 10B analyze receipt does not verify")
    chunk_manifests = {}
    for ch in s10["pilot"]["chunks"]:
        d = f"chunk_{ch['chunk']:02d}"
        if not XP.verify_dir(os.path.join(run, d), s10sha):
            raise RuntimeError(f"{d} does not verify")
        chunk_manifests[d] = P.sha256_file(os.path.join(run, d, "chunk_manifest.json"))
    res10 = XP._read_json(os.path.join(run, "results.json"))
    srcs = {n: P.sha256_file(os.path.join(os.path.dirname(__file__), n)) for n in
            ("xai_sim_ext.py", *s10["sources"].keys())}
    for n, h in s10["sources"].items():
        if srcs[n] != h:
            raise RuntimeError(f"Stage 10B source changed {n}")
    spec = {"spec_version": "xai_sim_ext_10b1_v1", "created_utc": _now(),
            "status": "retrospective extension motivated by the failed Stage 10B pilot (m=512 NOT_READY; m=1024 cannot qualify "
                      "because nuisance 0.5 and 1.0 each have 2/20 completeness failures)",
            "stage10b": {"run": R10B, "spec_sha256": s10sha, "chunk_manifest_sha256": chunk_manifests,
                         "readiness_10b": res10["readiness"]},
            "design": {"images": "identical 20 Stage 10B pilot images (no replacement)", "conditions": "identical 11",
                       "order_and_chunks": "identical (one chunk per condition, frozen image order)",
                       "checkpoint": s10["model"], "preprocessing_baseline_precision": "identical; baseline SHA-256 must equal the Stage 10B records",
                       "ig": {"candidate_m": M_CAND, "reference_m": M_REF, "rule": "riemann_midpoint", "internal_batch": XS.IG_IB},
                       "reused_not_recomputed": ["Grad-CAM", "controls (random/centre/Sobel/mismatch)", "localisation", "IG m=512/1024"]},
            "criteria": {"fail_rate_max": XPa.FAIL_RATE_MAX, "p95_rel_max": XPa.P95_REL_MAX, "median_abs_dauc_max": XPa.MEDIAN_DAUC_MAX,
                         "unchanged_from": "Stage 10B / xai_pairs"},
            "gates": {"gate1": "m=2048 for all 220 refs; criteria (a) and (b) must pass pooled AND in every condition; otherwise "
                               "NOT_READY and stop (no m=4096)",
                      "gate2": "only after gate1 pass: m=4096 on identical refs/batches; curves for both maps via the existing "
                               "eligibility (analysis_curves); assess_candidate(2048 vs 4096) pooled and per condition with eligible "
                               "comparison counts; Stage 10B readiness rule (pooled AND every condition) unchanged",
                      "no_escalation": "no m > 4096; m=4096 not claimed converged"},
            "units_note": "220 references = 20 images x 11 conditions (repeated images), not 220 independent cases",
            "sources": srcs}
    from . import provenance as Pv
    receipt = Pv.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "spec.json", spec)
    publish_bytes(out, "spec.json.sha256", f"{P.sha256_file(os.path.join(out, 'spec.json'))}  spec.json\n".encode())
    receipt["outputs"] = {"spec.json": P.sha256_file(os.path.join(out, "spec.json"))}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out)


def _load(cfg, out):
    from . import provenance as P, xai_prod as XP
    spec, ssha = XP._load(out, "spec.json")
    for n, h in spec["sources"].items():
        if P.sha256_file(os.path.join(os.path.dirname(__file__), n)) != h:
            raise RuntimeError(f"source changed {n} (resume refused)")
    run = _abs(cfg, R10B)
    s10, s10sha = XP._load(run, "spec.json")
    if s10sha != spec["stage10b"]["spec_sha256"]:
        raise RuntimeError("Stage 10B spec changed")
    for d, h in spec["stage10b"]["chunk_manifest_sha256"].items():
        if P.sha256_file(os.path.join(run, d, "chunk_manifest.json")) != h or not XP.verify_dir(os.path.join(run, d), s10sha):
            raise RuntimeError(f"Stage 10B {d} changed")
    return spec, ssha, s10, run


def _ig_phase(cfg, out, spec, ssha, s10, run, m, prefix, with_curves_for=()):
    import torch
    from . import xai_attr as XA, xai_curves as XC, xai_prod as XP
    full_rows = {r["record_uid"]: r for r in s10["full_arm"]["records"]}
    origs = XS._origs(cfg, s10["pilot"]["records"])
    masks = XS._masks256(cfg, s10["pilot"]["records"])
    model = XS._model(cfg)
    XP.quarantine_tmp(out)
    for ch in s10["pilot"]["chunks"]:
        final = os.path.join(out, f"{prefix}_{ch['chunk']:02d}")
        if XP.verify_dir(final, ssha):
            continue
        rec10 = XP._read_json(os.path.join(run, f"chunk_{ch['chunk']:02d}", "records.json"))
        rows = [full_rows[u] for u in ch["record_uids"]]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        xn, _, _ = XS._condition_inputs(origs, masks, rows, ch["factor"], ch["severity"])
        x = torch.from_numpy(xn).to("cuda")
        base = XA.gaussian_baseline(x)
        for i, r in enumerate(rec10):
            if XP._sha_bytes(base[i].detach().cpu().numpy().tobytes()) != r["baseline_sha256"]:
                raise RuntimeError(f"baseline differs from Stage 10B {r['ref_key']}")
        torch.cuda.synchronize()
        ti = time.perf_counter()
        ig = XA.integrated_gradients(model, x, base, m, XS.IG_IB)
        torch.cuda.synchronize()
        t_ig = time.perf_counter() - ti
        maps = {f"ig{m}": ig["primary"]}
        if with_curves_for:
            for mm_ in with_curves_for:
                if mm_ != m:
                    z = np.load(os.path.join(out, f"ig{mm_}_{ch['chunk']:02d}", "maps.npz"))
                    maps[f"ig{mm_}"] = torch.as_tensor(z["primary"], device="cuda")
        recs, tc = [], 0.0
        for i, r in enumerate(rec10):
            comp = ig["completeness"][i]
            el = XA.eligibility(XA.validity(ig["primary"][i]), comp)
            rec = {"ref_key": r["ref_key"], "record_uid": r["record_uid"], "label_mel": r["label_mel"], "condition": r["condition"],
                   "baseline_sha256": r["baseline_sha256"], "logit_x": float(ig["logit"][i]), "logit_baseline": float(ig["logit_baseline"][i]),
                   "completeness": comp, "eligibility": el, "curves": {}}
            for name in (sorted(maps) if with_curves_for else []):
                mp = maps[name][i]
                c_ = comp if name == f"ig{m}" else None
                if c_ is None:
                    prev = XP._read_json(os.path.join(out, f"{name}_{ch['chunk']:02d}", "records.json"))[i]
                    c_, elx = prev["completeness"], prev["eligibility"]
                else:
                    elx = el
                ts = time.perf_counter()
                try:
                    cv = XC.analysis_curves(model, x[i], base[i], mp, elx, "ig", eval_batch=XS.EVAL_BATCH, s=XS.S)
                    rec["curves"][name] = dict(XS._strip(cv), status=cv["analysis_status"])
                except XC.IneligibleMap as ex:
                    rec["curves"][name] = {"status": "ineligible", "reason": str(ex)}
                tc += time.perf_counter() - ts
            recs.append(rec)
        tmp = f"{final}.tmp-{uuid.uuid4().hex[:8]}"
        os.makedirs(tmp)
        buf = io.BytesIO()
        np.savez_compressed(buf, ref_keys=np.array([r["ref_key"] for r in recs]), primary=ig["primary"].cpu().numpy().astype(np.float32))
        with open(os.path.join(tmp, "maps.npz"), "xb") as f:
            f.write(buf.getvalue())
        with open(os.path.join(tmp, "records.json"), "x", encoding="utf-8") as f:
            json.dump(recs, f)
        meta = {"spec_sha256": ssha, "chunk": ch["chunk"], "condition": ch["condition"], "m": m, "ig_seconds": t_ig, "curve_seconds": tc,
                "chunk_seconds": time.perf_counter() - t0,
                "peak_cuda_mib": [round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)]}
        XP.publish_dir(tmp, final, meta)
        print(f"{prefix} {ch['chunk']:02d} {ch['condition']} ig {t_ig:.1f}s curves {tc:.1f}s", flush=True)


def _collect(out, spec, ssha, s10, prefix):
    from . import xai_prod as XP
    recs, man = [], []
    for ch in s10["pilot"]["chunks"]:
        d = os.path.join(out, f"{prefix}_{ch['chunk']:02d}")
        if not XP.verify_dir(d, ssha):
            raise RuntimeError(f"{d} incomplete")
        recs += XP._read_json(os.path.join(d, "records.json"))
        man.append(XP._read_json(os.path.join(d, "chunk_manifest.json")))
    if len(recs) != 220 or len({r["ref_key"] for r in recs}) != 220:
        raise RuntimeError("coverage")
    return recs, man


def _at(recs, name=None):
    out = {}
    for r in recs:
        c = r["completeness"]
        cv = r["curves"].get(name, {}) if name else {}
        out[r["ref_key"]] = {"completeness_failure": c["failure"], "e_rel": c["e_rel"], "eligible": r["eligibility"]["eligible_for_primary_analysis"],
                             "curve_status": cv.get("curve_status"), "endpoint_status": cv.get("endpoint_status"),
                             "auc_ins": cv.get("auc_insertion_v1"), "auc_del": cv.get("auc_deletion_v1"),
                             "map_flags": r["eligibility"]["map_flags"], "reasons": r["eligibility"]["ineligibility_reasons"]}
    return out


def phase_gate1(cfg, a):
    from . import provenance as P
    from .desc_4b3 import publish_csv, publish_json
    out = os.path.abspath(a.run_dir)
    spec, ssha, s10, run = _load(cfg, out)
    _ig_phase(cfg, out, spec, ssha, s10, run, M_CAND, f"ig{M_CAND}")
    recs, man = _collect(out, spec, ssha, s10, f"ig{M_CAND}")
    at = _at(recs)
    keys = [r["ref_key"] for r in recs]
    pooled = assess_ab(at, keys)
    per = {c["key"]: assess_ab(at, [r["ref_key"] for r in recs if r["condition"] == c["key"]]) for c in s10["conditions"]}
    dec = gate1_decision(pooled, per)
    rows = []
    for name, A in [("pooled", pooled)] + list(per.items()):
        sel = [r for r in recs if name == "pooled" or r["condition"] == name]
        rows.append({"group": name, "n_refs": A["n_refs"], "fail_2048": A["a"]["failures"], "fail_rate": A["a"]["rate"],
                     "fail_mel": sum(r["completeness"]["failure"] for r in sel if r["label_mel"] == "1"),
                     "fail_nonmel": sum(r["completeness"]["failure"] for r in sel if r["label_mel"] == "0"),
                     "p95_e_rel": A["b"].get("p95"), "e_rel_undefined": A["b"]["n_undefined"],
                     "status_a": A["a"]["status"], "status_b": A["b"]["status"], "gate1": A["overall"]})
    meas = {"ig_seconds_total": sum(m_["ig_seconds"] for m_ in man), "chunk_seconds_total": sum(m_["chunk_seconds"] for m_ in man),
            "ig_seconds_per_ref": sum(m_["ig_seconds"] for m_ in man) / 220,
            "peak_cuda_mib_max": [max(m_["peak_cuda_mib"][0] for m_ in man), max(m_["peak_cuda_mib"][1] for m_ in man)],
            "disk_bytes": sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)}
    res = {"gate1": dec, "pooled": pooled, "per_condition": per, "measured": meas,
           "next": "gate2 authorised by the frozen spec" if dec["status"] == "READY" else "STOP: NOT_READY; m=4096 not computed"}
    publish_json(out, "gate1_results.json", res)
    publish_csv(out, "gate1_table.csv", rows, list(rows[0]))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "gate1", "decision": dec["status"]})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("gate1_results.json", "gate1_table.csv", "spec.json")}
    rp = P.write_receipt(receipt, out, name="receipt_gate1.json")
    print(json.dumps({"gate1": dec, "measured": meas}, indent=1), "verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")
    if dec["status"] != "READY":
        raise SystemExit(3)                    # distinct exit code: gate not passed (not a crash)


def phase_gate2(cfg, a):
    from . import provenance as P, xai_pairs as XPa
    from .desc_4b3 import publish_csv, publish_json
    out = os.path.abspath(a.run_dir)
    spec, ssha, s10, run = _load(cfg, out)
    g1 = XS_json(os.path.join(out, "gate1_results.json"))
    if g1["gate1"]["status"] != "READY":
        raise RuntimeError("gate1 not passed; gate2 refused")
    _ig_phase(cfg, out, spec, ssha, s10, run, M_REF, f"ig{M_REF}", with_curves_for=(M_CAND, M_REF))
    recs, man = _collect(out, spec, ssha, s10, f"ig{M_REF}")
    a2048 = _at([dict(r, completeness=c["completeness"], eligibility=c["eligibility"]) for r, c in
                 zip(recs, _collect(out, spec, ssha, s10, f"ig{M_CAND}")[0])], f"ig{M_CAND}")
    a4096 = _at(recs, f"ig{M_REF}")
    keys = [r["ref_key"] for r in recs]
    pooled = XPa.assess_candidate(a2048, a4096, keys)
    per = {c["key"]: XPa.assess_candidate(a2048, a4096, [r["ref_key"] for r in recs if r["condition"] == c["key"]]) for c in s10["conditions"]}
    dec = XS.readiness(pooled, per)
    rows = [{"group": n, "n_refs": A["n_refs"], "fail_2048": A["a"]["failures"], "p95_e_rel": A["b"].get("p95"),
             "n_eligible_both": A["c"]["n_eligible_both"], "median_abs_dauc_ins": A["c"]["ins"]["median_abs"],
             "median_abs_dauc_del": A["c"]["del"]["median_abs"], "status_a": A["a"]["status"], "status_b": A["b"]["status"],
             "status_c_ins": A["c"]["ins"]["status"], "status_c_del": A["c"]["del"]["status"], "overall": A["overall"]}
            for n, A in [("pooled", pooled)] + list(per.items())]
    meas = {"ig_seconds_total": sum(m_["ig_seconds"] for m_ in man), "curve_seconds_total": sum(m_["curve_seconds"] for m_ in man)}
    publish_json(out, "gate2_results.json", {"readiness_2048_vs_4096": dec, "pooled": pooled, "per_condition": per, "measured": meas,
                                             "note": "m=4096 is the reference; it is not claimed converged"})
    publish_csv(out, "gate2_table.csv", rows, list(rows[0]))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "gate2", "decision": dec["status"]})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("gate2_results.json", "gate2_table.csv", "spec.json")}
    rp = P.write_receipt(receipt, out, name="receipt_gate2.json")
    print(json.dumps({"readiness": dec, "measured": meas}, indent=1), "verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def XS_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "gate1", "gate2"))
    ap.add_argument("--out-root")
    ap.add_argument("--run-dir")
    a = ap.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from . import config as C
    from .seeds import seed_everything
    cfg = C.load(a.config)
    if cfg["purpose"] != "xai_diag":
        raise C.ConfigError("requires purpose 'xai_diag'")
    seed_everything(0, cfg["determinism"])
    {"spec": phase_spec, "gate1": phase_gate1, "gate2": phase_gate2}[a.phase](cfg, a)


if __name__ == "__main__":
    main()
