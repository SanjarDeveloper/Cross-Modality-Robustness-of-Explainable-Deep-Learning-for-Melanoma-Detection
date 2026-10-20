"""Stage 2D2 diagnostic: insertion/deletion curves + attribution numerical sensitivity.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_curve_diag --config ../configs/xai_curve_diag_v1.json --out-root ../runs/xai_curve_diag

SOFTWARE DIAGNOSTIC ONLY: Stage 2C3 pilot checkpoint, the same 8 HAM TRAIN images as
Stage 2D1. No production IG step count is selected here. Curve-evaluation batching is
FIXED (CURVE_EB) for all attribution comparisons and varied only in its own check.
"""
import argparse
import json
import os
import time

CURVE_EB = 11
IG_IB = 16
M_FIXED = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from . import config as C, decode as DC, provenance as P, torch_data as TD, trainer as TR
    from . import xai_attr as XA, xai_curves as XC
    from .seeds import seed_everything, stable_seed
    from .smoke_train import build_resnet50
    cfg = C.load(a.config)
    if cfg["purpose"] != "xai_diag":
        raise C.ConfigError("requires purpose 'xai_diag'")
    t_all = time.perf_counter()
    dev = "cuda"
    seed_everything(0, cfg["determinism"])
    ck_path = C.resolve(cfg, cfg["supporting_inputs"]["pilot_checkpoint"])
    with open(os.path.join(os.path.dirname(ck_path), "latest.json"), encoding="utf-8") as f:
        ptr = json.load(f)
    ck = TR._load_verified(ck_path, ptr["sha256"])
    pcfg = dict(ck["config"], project_root=cfg["project_root"])
    model, _, _, _, _ = build_resnet50(pcfg, dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    layer = model.layer4
    root = cfg["project_root"]
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    sel = []
    for lab in ("1", "0"):
        rr = sorted((r for r in rows if r["split"] == "train" and r["label_mel"] == lab),
                    key=lambda r: (stable_seed("xai_diag_v1", r["record_uid"]), r["record_uid"]))
        sel += [r["record_uid"] for r in rr[:4]]
    ds = TD.ManifestDataset(rows, "ham_train", os.path.join(root, "cache"), os.path.join(root, "DATASETS"),
                            sel, mode="eval")
    x = torch.stack([ds[i]["x"] for i in range(len(ds))]).to(dev)
    base = XA.gaussian_baseline(x)
    N = len(sel)
    timings, mem, arrays, curves_store = {}, {}, {}, {}

    def timed(name, fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        timings[name] = round(time.perf_counter() - t0, 3)
        mem[name] = [round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                     round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)]
        return out

    def cam_b1():
        return torch.cat([XA.grad_cam(model, layer, x[i:i + 1])["map"] for i in range(N)])

    def ig(m, ib):
        return XA.integrated_gradients(model, x, base, m, internal_batch=ib)

    def curve_set(maps, name, eb=CURVE_EB):
        res = [XC.curves(model, x[i], base[i], maps[i], eval_batch=eb) for i in range(N)]
        curves_store[name] = {"insertion": [r["insertion"] for r in res], "deletion": [r["deletion"] for r in res],
                              "fractions": res[0]["fractions"], "eval_batch": eb,
                              "auc_ins": [r["auc_insertion_v1"] for r in res],
                              "auc_del": [r["auc_deletion_v1"] for r in res],
                              "endpoint_max_abs": [r["endpoint_prediction_max_abs"] for r in res]}
        return curves_store[name]

    def top_overlap(A, B, frac=0.10):
        k = int(round(frac * A.numel()))
        oa, _ = XC.rank_positions(A)
        ob, _ = XC.rank_positions(B)
        return len(set(oa[:k].tolist()) & set(ob[:k].tolist())) / k

    def compare_maps(A, B):
        out = []
        for i in range(N):
            a_, b_ = A[i].double().cpu(), B[i].double().cpu()
            denom = float(b_.abs().max())
            out.append({"max_abs": float((a_ - b_).abs().max()),
                        "rel_to_own_max": float((a_ - b_).abs().max()) / denom if denom > 0 else None,
                        "rel_L1": float((a_ - b_).abs().sum() / max(float(b_.abs().sum()), 1e-30)),
                        "spearman": float(spearmanr(a_.flatten().numpy(), b_.flatten().numpy())[0]),
                        "top10_overlap": top_overlap(A[i].cpu(), B[i].cpu()),
                        "identical_ranking": bool(torch.equal(XC.rank_positions(A[i].cpu())[0],
                                                              XC.rank_positions(B[i].cpu())[0]))})
        return out

    def compare_auc(ca, cb):
        return [{"d_auc_ins": ca["auc_ins"][i] - cb["auc_ins"][i], "d_auc_del": ca["auc_del"][i] - cb["auc_del"][i]}
                for i in range(N)]

    def summarise(lst, key):
        v = [abs(d[key]) for d in lst if d.get(key) is not None]
        return {"median_abs": float(np.median(v)), "max_abs": float(np.max(v))} if v else None

    R = {"precision_default": XA.precision_settings()}
    # ---------------- default precision
    cam8 = timed("gradcam_b8", lambda: XA.grad_cam(model, layer, x))
    cam8b = XA.grad_cam(model, layer, x)
    cam1 = timed("gradcam_b1_x8", cam_b1)
    igs = {ib: timed(f"ig_m{M_FIXED}_ib{ib}", lambda ib=ib: ig(M_FIXED, ib)) for ib in (1, 8, IG_IB)}
    ig16b = ig(M_FIXED, IG_IB)
    c_cam8 = timed("curves_gradcam_b8", lambda: curve_set(cam8["map"], "gradcam_b8"))
    c_cam8_rep = curve_set(cam8["map"], "gradcam_b8_repeat")
    c_cam1 = curve_set(cam1, "gradcam_b1")
    c_ig = {ib: curve_set(igs[ib]["primary"], f"ig_m{M_FIXED}_ib{ib}") for ib in (1, 8, IG_IB)}
    R["repeat"] = {"gradcam_identical": bool(torch.equal(cam8["map"], cam8b["map"])),
                   "ig_identical": bool(torch.equal(igs[IG_IB]["attr"], ig16b["attr"])),
                   "curves_identical": c_cam8["insertion"] == c_cam8_rep["insertion"]
                   and c_cam8["deletion"] == c_cam8_rep["deletion"]}
    R["gradcam_b1_vs_b8"] = {"logit_abs_diff": [abs(float(cam1_l) - float(l8)) for cam1_l, l8 in
                                                zip(torch.cat([XA.grad_cam(model, layer, x[i:i + 1])["logit"]
                                                               for i in range(N)]), cam8["logit"])],
                             "maps": compare_maps(cam1, cam8["map"]), "auc": compare_auc(c_cam1, c_cam8)}
    R["ig_internal_batch"] = {f"ib{ib}_vs_ib{IG_IB}": {"attr": [{"attr_max_abs": float((igs[ib]["attr"][i] - igs[IG_IB]["attr"][i]).abs().max()),
                                                        "attr_rel_to_own_max": float((igs[ib]["attr"][i] - igs[IG_IB]["attr"][i]).abs().max()
                                                                                     / igs[IG_IB]["attr"][i].abs().max())}
                                                       for i in range(N)],
                                                      "primary_maps": compare_maps(igs[ib]["primary"], igs[IG_IB]["primary"]),
                                                      "auc": compare_auc(c_ig[ib], c_ig[IG_IB]),
                                                      "e_abs": [c["e_abs"] for c in igs[ib]["completeness"]]}
                            for ib in (1, 8)}
    # ---------------- highest FP32
    with XA.highest_fp32():
        R["precision_highest"] = XA.precision_settings()
        camH = timed("gradcam_b8_highest", lambda: XA.grad_cam(model, layer, x))
        camH1 = cam_b1()
        igH = {ib: timed(f"ig_m{M_FIXED}_ib{ib}_highest", lambda ib=ib: ig(M_FIXED, ib)) for ib in (1, IG_IB)}
        c_camH = curve_set(camH["map"], "gradcam_b8_highest")
        c_camH1 = curve_set(camH1, "gradcam_b1_highest")
        c_igH = {ib: curve_set(igH[ib]["primary"], f"ig_m{M_FIXED}_ib{ib}_highest") for ib in (1, IG_IB)}
    R["default_vs_highest"] = {"logit_abs_diff": [abs(float(u) - float(v)) for u, v in zip(cam8["logit"], camH["logit"])],
                               "gradcam_maps": compare_maps(cam8["map"], camH["map"]),
                               "gradcam_auc": compare_auc(c_cam8, c_camH),
                               "ig_primary_maps": compare_maps(igs[IG_IB]["primary"], igH[IG_IB]["primary"]),
                               "ig_auc": compare_auc(c_ig[IG_IB], c_igH[IG_IB])}
    R["highest_batch_checks"] = {"gradcam_b1_vs_b8_maps": compare_maps(camH1, camH["map"]),
                                 "gradcam_b1_vs_b8_auc": compare_auc(c_camH1, c_camH),
                                 "ig_ib1_vs_ib16_attr_max_abs": [float((igH[1]["attr"][i] - igH[IG_IB]["attr"][i]).abs().max())
                                                                 for i in range(N)],
                                 "ig_ib1_vs_ib16_primary_maps": compare_maps(igH[1]["primary"], igH[IG_IB]["primary"]),
                                 "ig_ib1_vs_ib16_auc": compare_auc(c_igH[1], c_igH[IG_IB])}
    # ---------------- m = 64, 128, 256 (fixed internal batch), both precisions
    R["m_convergence"] = {}
    for prec in ("default", "highest"):
        ctx = XA.highest_fp32() if prec == "highest" else _Null()
        with ctx:
            res = {m: (igs[IG_IB] if (m == M_FIXED and prec == "default") else
                       igH[IG_IB] if (m == M_FIXED and prec == "highest") else
                       timed(f"ig_m{m}_ib{IG_IB}_{prec}", lambda m=m: ig(m, IG_IB))) for m in (64, 128, 256)}
            cv = {m: (c_ig[IG_IB] if (m == M_FIXED and prec == "default") else
                      c_igH[IG_IB] if (m == M_FIXED and prec == "highest") else
                      curve_set(res[m]["primary"], f"ig_m{m}_ib{IG_IB}_{prec}")) for m in (64, 128, 256)}
        block = {}
        for m in (64, 128, 256):
            er = [c["e_rel"] for c in res[m]["completeness"] if c["e_rel"] is not None]
            block[f"m{m}"] = {"completeness": res[m]["completeness"],
                              "failures": sum(c["failure"] for c in res[m]["completeness"]),
                              "failure_rate": sum(c["failure"] for c in res[m]["completeness"]) / N,
                              "p95_e_rel": float(np.percentile(er, 95)) if er else None,
                              "eligibility": [XA.eligibility(XA.validity(res[m]["primary"][i]), res[m]["completeness"][i])
                                              for i in range(N)]}
            arrays[f"ig_primary_m{m}_{prec}"] = res[m]["primary"].cpu().numpy()
            arrays[f"ig_attr_m{m}_{prec}"] = res[m]["attr"].cpu().numpy()
        for m in (64, 128):
            d = compare_auc(cv[m], cv[2 * m])
            block[f"m{m}_vs_m{2 * m}"] = {"maps": compare_maps(res[m]["primary"], res[2 * m]["primary"]),
                                          "auc": d,
                                          "criterion_c_median_abs_d_auc_ins": float(np.median([abs(v["d_auc_ins"]) for v in d])),
                                          "criterion_c_median_abs_d_auc_del": float(np.median([abs(v["d_auc_del"]) for v in d])),
                                          "criterion_c_threshold": 0.005}
        R["m_convergence"][prec] = block
    # ---------------- curve-evaluation batching (fixed map)
    c_eb = {eb: curve_set(cam8["map"], f"gradcam_b8_eb{eb}", eb) for eb in (1, 33)}
    c_eb[CURVE_EB] = c_cam8
    R["curve_eval_batch"] = {f"eb{eb}_vs_eb{CURVE_EB}": {"auc": compare_auc(c_eb[eb], c_cam8),
                                                         "max_abs_point_diff": max(abs(p - q) for i in range(N)
                                                                                   for kk in ("insertion", "deletion")
                                                                                   for p, q in zip(c_eb[eb][kk][i], c_cam8[kk][i]))}
                             for eb in (1, 33)}
    R["gradcam_eligibility"] = [XA.eligibility(XA.validity(cam8["map"][i])) for i in range(N)]
    R["endpoint_max_abs_all_curves"] = max(max(v["endpoint_max_abs"]) for v in curves_store.values())
    # summaries
    R["summary"] = {
        "gradcam_b1_vs_b8_default": {"rel_to_own_max": summarise(R["gradcam_b1_vs_b8"]["maps"], "rel_to_own_max"),
                                     "d_auc_ins": summarise(R["gradcam_b1_vs_b8"]["auc"], "d_auc_ins"),
                                     "d_auc_del": summarise(R["gradcam_b1_vs_b8"]["auc"], "d_auc_del"),
                                     "min_spearman": min(d["spearman"] for d in R["gradcam_b1_vs_b8"]["maps"])},
        "gradcam_b1_vs_b8_highest": {"rel_to_own_max": summarise(R["highest_batch_checks"]["gradcam_b1_vs_b8_maps"], "rel_to_own_max"),
                                     "d_auc_ins": summarise(R["highest_batch_checks"]["gradcam_b1_vs_b8_auc"], "d_auc_ins"),
                                     "d_auc_del": summarise(R["highest_batch_checks"]["gradcam_b1_vs_b8_auc"], "d_auc_del")},
        "ig_ib1_vs_ib16_default": {"attr_rel_to_own_max": summarise(R["ig_internal_batch"]["ib1_vs_ib16"]["attr"], "attr_rel_to_own_max"),
                                   "d_auc_ins": summarise(R["ig_internal_batch"]["ib1_vs_ib16"]["auc"], "d_auc_ins"),
                                   "d_auc_del": summarise(R["ig_internal_batch"]["ib1_vs_ib16"]["auc"], "d_auc_del"),
                                   "min_spearman": min(d["spearman"] for d in R["ig_internal_batch"]["ib1_vs_ib16"]["primary_maps"])},
        "ig_ib1_vs_ib16_highest": {"attr_max_abs_max": max(R["highest_batch_checks"]["ig_ib1_vs_ib16_attr_max_abs"]),
                                   "d_auc_ins": summarise(R["highest_batch_checks"]["ig_ib1_vs_ib16_auc"], "d_auc_ins"),
                                   "d_auc_del": summarise(R["highest_batch_checks"]["ig_ib1_vs_ib16_auc"], "d_auc_del")},
        "default_vs_highest": {"logit_max_abs": max(R["default_vs_highest"]["logit_abs_diff"]),
                               "gradcam_rel_to_own_max": summarise(R["default_vs_highest"]["gradcam_maps"], "rel_to_own_max"),
                               "gradcam_d_auc_ins": summarise(R["default_vs_highest"]["gradcam_auc"], "d_auc_ins"),
                               "ig_d_auc_ins": summarise(R["default_vs_highest"]["ig_auc"], "d_auc_ins"),
                               "ig_d_auc_del": summarise(R["default_vs_highest"]["ig_auc"], "d_auc_del")},
    }
    R["timings_s"], R["peak_memory_alloc_reserved_mib"] = timings, mem
    R["total_elapsed_s"] = round(time.perf_counter() - t_all, 2)
    R["records"] = sel
    R["model"] = {"checkpoint": cfg["supporting_inputs"]["pilot_checkpoint"], "sha256": ptr["sha256"],
                  "note": "Stage 2C3 pilot checkpoint; software diagnostic only; trained on these images"}
    R["fixed"] = {"curve_eval_batch": CURVE_EB, "ig_internal_batch": IG_IB, "ig_m_for_batch_checks": M_FIXED,
                  "S": 32, "curves_version": XC.CURVES_VERSION, "attr_version": XA.ATTR_VERSION}
    arrays.update(x=x.cpu().numpy(), baseline=base.cpu().numpy(), gradcam_b8=cam8["map"].cpu().numpy(),
                  gradcam_b1=cam1.cpu().numpy(), gradcam_b8_highest=camH["map"].cpu().numpy(),
                  **{f"ig_primary_m64_ib{ib}_default": igs[ib]["primary"].cpu().numpy() for ib in (1, 8, 16)})
    receipt = P.build_receipt(cfg, a.config, "completed", details={"curves_version": XC.CURVES_VERSION,
                                                                  "attr_version": XA.ATTR_VERSION})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    np.savez_compressed(os.path.join(out, "maps.npz"), record_uids=np.array(sel), **arrays)
    with open(os.path.join(out, "curves.json"), "x", encoding="utf-8") as f:
        json.dump({"record_uids": sel, "curves": curves_store}, f)
    with open(os.path.join(out, "diagnostics.json"), "x", encoding="utf-8") as f:
        json.dump(R, f, indent=1)
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in ("maps.npz", "curves.json", "diagnostics.json")}
    rp = P.write_receipt(receipt, out)
    print(rp)
    print(json.dumps({k: R[k] for k in ("repeat", "summary", "timings_s", "total_elapsed_s",
                                        "endpoint_max_abs_all_curves", "precision_default", "precision_highest")}, indent=1))
    for prec in ("default", "highest"):
        b = R["m_convergence"][prec]
        print(prec, {k: (v["failures"], round(v["p95_e_rel"], 4)) for k, v in b.items() if k.startswith("m") and "_vs_" not in k},
              {k: (round(v["criterion_c_median_abs_d_auc_ins"], 5), round(v["criterion_c_median_abs_d_auc_del"], 5),
                   round(min(d["spearman"] for d in v["maps"]), 4)) for k, v in b.items() if "_vs_" in k})
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


if __name__ == "__main__":
    main()
