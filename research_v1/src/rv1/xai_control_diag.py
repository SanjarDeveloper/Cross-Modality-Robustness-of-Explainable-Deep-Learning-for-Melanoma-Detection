"""Stage 2D3 bounded control diagnostic (SOFTWARE DIAGNOSTIC ONLY).

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_control_diag --config ../configs/xai_control_diag_v1.json --out-root ../runs/xai_control_diag

Same 8 HAM TRAIN images and Stage 2C3 pilot checkpoint as Stages 2D1-2D2.1. Diagnostic
settings: IG m=256, internal_batch=16; curves S=32, eval_batch=11; xai_precision_v1.
Controls per spec controls_v1 (configs/xai_controls_v1_spec.json). The recipient->donor
mapping is written to disk BEFORE any curve is evaluated. Scores are not findings.
"""
import argparse
import csv
import json
import os
import time

IG_M, IG_IB, EVAL_BATCH = 256, 16, 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from . import config as C, decode as DC, preprocess as PP, provenance as P, torch_data as TD, trainer as TR
    from . import xai_attr as XA, xai_controls as XK, xai_curves as XC
    from .seeds import seed_everything, stable_seed
    from .smoke_train import build_resnet50
    cfg = C.load(a.config)
    if cfg["purpose"] != "xai_diag":
        raise C.ConfigError("requires purpose 'xai_diag'")
    t_all = time.perf_counter()
    dev = "cuda"
    seed_everything(0, cfg["determinism"])
    T, M = {}, {}

    def timed(name, fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        T[name] = round(time.perf_counter() - t0, 3)
        M[name] = [round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                   round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)]
        return out

    ck_path = C.resolve(cfg, cfg["supporting_inputs"]["pilot_checkpoint"])
    with open(os.path.join(os.path.dirname(ck_path), "latest.json"), encoding="utf-8") as f:
        ptr = json.load(f)
    ck = TR._load_verified(ck_path, ptr["sha256"])
    model, _, _, _, _ = build_resnet50(dict(ck["config"], project_root=cfg["project_root"]), dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    root = cfg["project_root"]
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    sel = []
    for lab in ("1", "0"):
        rr = sorted((r for r in rows if r["split"] == "train" and r["label_mel"] == lab),
                    key=lambda r: (stable_seed("xai_diag_v1", r["record_uid"]), r["record_uid"]))
        sel += [r["record_uid"] for r in rr[:4]]
    ds = TD.ManifestDataset(rows, "ham_train", os.path.join(root, "cache"), os.path.join(root, "DATASETS"),
                            sel, mode="eval")
    recs = ds.rows
    x = torch.stack([ds[i]["x"] for i in range(len(ds))]).to(dev)
    N = len(sel)
    with XA.xai_precision():
        policy = XA.precision_settings()
    # ---------------- attribution
    def attribute():
        base = XA.gaussian_baseline(x)
        cam = XA.grad_cam(model, model.layer4, x)
        ig = XA.integrated_gradients(model, x, base, IG_M, IG_IB)
        return base, cam, ig
    base, cam, ig = timed("attribution", attribute)
    expl_maps = {"gradcam": cam["map"].cpu(), "ig": ig["primary"].cpu()}
    elig = {"gradcam": [XA.eligibility(XA.validity(cam["map"][i])) for i in range(N)],
            "ig": [XA.eligibility(XA.validity(ig["primary"][i]), ig["completeness"][i]) for i in range(N)]}
    # ---------------- mismatch assignment, persisted BEFORE metrics
    receipt = P.build_receipt(cfg, a.config, "completed",
                              details={"policy": policy, "controls_version": XK.CONTROLS_VERSION})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    mm_recs = [{"record_uid": r["record_uid"], "case_id": r["cluster_id"], "label": r["label_mel"],
                "modality": "dermoscopy", "explainer": e, "condition": "none"}
               for e in ("gradcam", "ig") for r in recs]
    assign = {e: XK.mismatch_assignment([m for m in mm_recs if m["explainer"] == e]) for e in ("gradcam", "ig")}
    with open(os.path.join(out, "mismatch_assignment.json"), "x", encoding="utf-8") as f:
        json.dump({"written_before_metrics": True, "assignment": assign}, f, indent=1)
    # ---------------- control generation
    def controls():
        c = {}
        for i, r in enumerate(recs):
            c[r["record_uid"]] = {"random": XK.random_maps(r["image_uid"]), "centre": XK.centre_map(),
                                  "sobel": XK.sobel_map(x[i])}
        return c
    ctrl = timed("control_generation", controls)
    uid_idx = {r["record_uid"]: i for i, r in enumerate(recs)}
    # ---------------- curves
    def to_t(m):
        return torch.as_tensor(np.asarray(m), dtype=torch.float64)

    def run_curves():
        res = {}
        for i, r in enumerate(recs):
            u = r["record_uid"]
            R = res[u] = {"explainers": {}, "controls": {}}
            for e in ("gradcam", "ig"):
                try:
                    cv = XC.analysis_curves(model, x[i], base[i], expl_maps[e][i], elig[e][i], e,
                                            eval_batch=EVAL_BATCH)
                    ok = cv["analysis_status"] == "primary"
                    R["explainers"][e] = {"status": cv["analysis_status"], "curve": cv,
                                          "auc_ins": cv["auc_insertion_v1"] if ok else None,
                                          "auc_del": cv["auc_deletion_v1"] if ok else None}
                except XC.IneligibleMap as ex:
                    R["explainers"][e] = {"status": "ineligible", "reason": str(ex), "auc_ins": None, "auc_del": None}
            rc = [XC.curves(model, x[i], base[i], to_t(m), eval_batch=EVAL_BATCH) for m in ctrl[u]["random"]]
            R["controls"]["random"] = {"status": "available" if all(c["curve_status"] == "ok" for c in rc) else "unavailable",
                                       "reason": None, "curves": rc,
                                       "auc_ins_each": [c["auc_insertion_v1"] for c in rc],
                                       "auc_del_each": [c["auc_deletion_v1"] for c in rc]}
            for name in ("centre", "sobel"):
                ok, flags = XK.map_valid(ctrl[u][name])
                if not ok:
                    R["controls"][name] = {"status": "unavailable", "reason": f"{name} map invalid: {flags}"}
                    continue
                c = XC.curves(model, x[i], base[i], to_t(ctrl[u][name]), eval_batch=EVAL_BATCH)
                R["controls"][name] = {"status": "available" if c["curve_status"] == "ok" else "unavailable",
                                       "reason": None, "curve": c, "auc_ins": c["auc_insertion_v1"],
                                       "auc_del": c["auc_deletion_v1"]}
            for e in ("gradcam", "ig"):
                donor_maps = {v: expl_maps[e][uid_idx[v]] for v in uid_idx}
                donor_ok = {v: elig[e][uid_idx[v]]["eligible_for_primary_analysis"] for v in uid_idx}
                dm, st = XK.resolve_mismatch(XK.assignment_key(u, "dermoscopy", e, "none"), assign[e], donor_maps, donor_ok)
                key = f"mismatch_{e}"
                if dm is None:
                    R["controls"][key] = dict(st)
                    continue
                c = XC.curves(model, x[i], base[i], dm, eval_batch=EVAL_BATCH)
                R["controls"][key] = dict(st, status="available" if c["curve_status"] == "ok" else "unavailable",
                                          curve=c, auc_ins=c["auc_insertion_v1"], auc_del=c["auc_deletion_v1"])
        return res
    res = timed("curve_evaluation", run_curves)
    # ---------------- scores
    rows_out = []
    for u, R in res.items():
        for e in ("gradcam", "ig"):
            ex = R["explainers"][e]
            for cname in ("random", "centre", "sobel", f"mismatch_{e}"):
                sc = XK.control_scores(ex, R["controls"][cname])
                rows_out.append({"record_uid": u, "label_mel": recs[uid_idx[u]]["label_mel"], "explainer": e,
                                 "explainer_status": ex["status"], "control": cname.split("_")[0],
                                 "control_status": sc["status"], "reason": sc.get("reason"),
                                 "explainer_auc_ins": ex["auc_ins"], "explainer_auc_del": ex["auc_del"],
                                 "control_auc_ins": sc.get("control_auc_ins"), "control_auc_del": sc.get("control_auc_del"),
                                 "insertion_advantage": sc["insertion_advantage"],
                                 "deletion_advantage": sc["deletion_advantage"]})
    # ---------------- persistence
    def persist():
        np.savez_compressed(os.path.join(out, "maps.npz"), record_uids=np.array(sel), x=x.cpu().numpy(),
                            baseline=base.cpu().numpy(), gradcam=cam["map"].cpu().numpy(),
                            ig_attr=ig["attr"].cpu().numpy(), ig_primary=ig["primary"].cpu().numpy(),
                            random=np.stack([np.stack(ctrl[u]["random"]) for u in sel]),
                            centre=XK.centre_map(), sobel=np.stack([ctrl[u]["sobel"] for u in sel]))

        def strip(c):
            return {k: c[k] for k in ("insertion", "deletion", "fractions", "auc_insertion_v1", "auc_deletion_v1",
                                      "endpoint_status", "endpoint_prediction_max_abs", "curve_status", "eval_batch")
                    if k in c}
        curves_out = {}
        for u, R in res.items():
            curves_out[u] = {"explainers": {e: dict({k: v for k, v in d.items() if k != "curve"},
                                                    curve=strip(d["curve"]) if "curve" in d else None)
                                            for e, d in R["explainers"].items()},
                             "controls": {}}
            for cn, d in R["controls"].items():
                dd = {k: v for k, v in d.items() if k not in ("curve", "curves")}
                if "curves" in d:
                    dd["curves"] = [strip(c) for c in d["curves"]]
                if "curve" in d:
                    dd["curve"] = strip(d["curve"])
                curves_out[u]["controls"][cn] = dd
        with open(os.path.join(out, "curves.json"), "x", encoding="utf-8") as f:
            json.dump(curves_out, f)
        with open(os.path.join(out, "scores.csv"), "x", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows_out)
        with open(os.path.join(out, "validity.json"), "x", encoding="utf-8") as f:
            json.dump({"explainers": {e: dict(zip(sel, elig[e])) for e in elig},
                       "ig_completeness": dict(zip(sel, ig["completeness"])),
                       "controls": {u: {c: XK.map_valid(ctrl[u][c]) for c in ("centre", "sobel")} for u in sel}},
                      f, indent=1)
    timed("persistence", persist)
    # ---------------- review sheet (display normalisation only)
    tiles = []
    for i, u in enumerate(sel):
        img = np.clip((x[i].cpu().numpy().transpose(1, 2, 0) * np.array(PP.STD) + np.array(PP.MEAN)) * 255,
                      0, 255).astype(np.uint8)

        def heat(mp, clip99=False):
            a_ = np.asarray(mp, dtype=np.float64)
            if clip99:
                a_ = np.minimum(a_, np.percentile(a_, 99.5))
            n = XA.display_normalise(a_)
            return (0.45 * img + 0.55 * (np.stack([n, np.zeros_like(n), 1 - n], -1) * 255)).astype(np.uint8)
        d_g = assign["gradcam"][XK.assignment_key(u, "dermoscopy", "gradcam", "none")]["donor"]
        panels = [img, heat(cam["map"][i].cpu().numpy()), heat(ig["primary"][i].cpu().numpy(), True),
                  heat(ctrl[u]["random"][0]), heat(ctrl[u]["centre"]), heat(ctrl[u]["sobel"], True),
                  heat(cam["map"][uid_idx[d_g]].cpu().numpy()) if d_g else np.zeros_like(img)]
        row = np.concatenate(panels, 1)
        tile = Image.new("RGB", (row.shape[1], row.shape[0] + 16), "white")
        tile.paste(Image.fromarray(row), (0, 16))
        ImageDraw.Draw(tile).text((2, 2), f"{u} y={recs[i]['label_mel']} | image | Grad-CAM | IG m256 (display clip 99.5%) | "
                                          f"random k0 | centre | Sobel (clip 99.5%) | mismatch Grad-CAM from {d_g}",
                                  fill="black")
        tiles.append(tile)
    sheet = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)), "white")
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height
    sheet.save(os.path.join(out, "review_sheet.png"))
    summary = {"policy": policy, "settings": {"ig_m": IG_M, "ig_internal_batch": IG_IB, "eval_batch": EVAL_BATCH,
                                              "S": 32, "controls_version": XK.CONTROLS_VERSION},
               "explainer_status": {e: [res[u]["explainers"][e]["status"] for u in sel] for e in ("gradcam", "ig")},
               "control_status": {c: [res[u]["controls"][c]["status"] for u in sel]
                                  for c in ("random", "centre", "sobel", "mismatch_gradcam", "mismatch_ig")},
               "unavailable_reasons": {f"{u}|{c}": res[u]["controls"][c].get("reason") for u in sel
                                       for c in res[u]["controls"] if res[u]["controls"][c]["status"] != "available"},
               "endpoint_failures": sum(1 for u in sel for c in res[u]["controls"].values()
                                        for cc in (c.get("curves") or ([c["curve"]] if "curve" in c else []))
                                        if cc["endpoint_status"] != "pass") +
                                    sum(1 for u in sel for e in res[u]["explainers"].values()
                                        if "curve" in e and e["curve"]["endpoint_status"] != "pass"),
               "n_curves_evaluated": sum(2 * (len(res[u]["controls"]["random"]["curves"]) +
                                              sum(1 for c in res[u]["controls"].values() if "curve" in c) +
                                              sum(1 for e in res[u]["explainers"].values() if "curve" in e))
                                         for u in sel),
               "timings_s": T, "peak_memory_alloc_reserved_mib": M,
               "model": {"checkpoint": cfg["supporting_inputs"]["pilot_checkpoint"], "sha256": ptr["sha256"],
                         "note": "pilot checkpoint; software diagnostic only; trained on these images"},
               "total_elapsed_s": round(time.perf_counter() - t_all, 2)}
    with open(os.path.join(out, "summary.json"), "x", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in
                          ("mismatch_assignment.json", "maps.npz", "curves.json", "scores.csv", "validity.json",
                           "summary.json", "review_sheet.png")}
    rp = P.write_receipt(receipt, out)
    print(rp)
    print(json.dumps(summary, indent=1))
    for r in rows_out:
        print(r["record_uid"][-12:], r["label_mel"], r["explainer"], r["control"], r["control_status"],
              None if r["insertion_advantage"] is None else round(r["insertion_advantage"], 3),
              None if r["deletion_advantage"] is None else round(r["deletion_advantage"], 3))
    print("assignment gradcam:", {v["record_uid"][-12:]: (v["donor"] or "")[-12:] for v in assign["gradcam"].values()})
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


if __name__ == "__main__":
    main()
