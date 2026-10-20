"""Stage 2D1 small attribution diagnostic (SOFTWARE DIAGNOSTIC ONLY).

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_diag --config ../configs/xai_diag_v1.json --out-root ../runs/xai_diag

Model: the Stage 2C3 throughput-pilot checkpoint (verified SHA-256), used only to exercise
the code; it is not a research model and nothing here is a result. Records: 8 HAM TRAIN
images (4 melanoma, 4 non-melanoma) ordered by stable_seed("xai_diag_v1", record_uid).
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

STEPS = (16, 32, 64, 128)


def _state_hash(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    raw = json.load(open(a.config, encoding="utf-8"))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from . import config as C, decode as DC, preprocess as PP, provenance as P, torch_data as TD
    from . import trainer as TR, xai_attr as XA
    from .seeds import seed_everything, stable_seed
    from .smoke_train import build_resnet50
    cfg = C.load(a.config)
    if cfg["purpose"] != "xai_diag":
        raise C.ConfigError("xai_diag requires purpose 'xai_diag'")
    t_all = time.perf_counter()
    dev = "cuda"
    seed_everything(0, cfg["determinism"])
    # --- model: pilot checkpoint, verified
    ck_rel = cfg["supporting_inputs"]["pilot_checkpoint"]
    ck_path = C.resolve(cfg, ck_rel)
    ck_sha = P.sha256_file(ck_path)
    with open(os.path.join(os.path.dirname(ck_path), "latest.json"), encoding="utf-8") as f:
        ptr = json.load(f)
    ck = TR._load_verified(ck_path, ptr["sha256"])
    pilot_cfg = ck["config"]
    pilot_cfg["project_root"] = cfg["project_root"]
    torch.cuda.synchronize()
    tb = time.perf_counter()
    model, _, _, _, info = build_resnet50(pilot_cfg, dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    torch.cuda.synchronize()
    build_s = time.perf_counter() - tb
    layer = model.layer4
    h0 = _state_hash(model)
    # --- records
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
    labels = [int(r["label_mel"]) for r in ds.rows]
    # --- baseline (and equality with the legacy implementation)
    base = XA.gaussian_baseline(x)
    sys.path.insert(0, os.path.join(root, "code"))
    from eval.xai import _gauss_blur as legacy_blur          # read-only import (-B)
    legacy_diff = float((legacy_blur(x, 57) - base).abs().max())
    timings, mem = {}, {}

    def timed(name, fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        timings[name] = round(time.perf_counter() - t0, 4)
        mem[name] = {"alloc_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                     "reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)}
        return out
    cam = timed("gradcam_8_images", lambda: XA.grad_cam(model, layer, x))
    cam2 = XA.grad_cam(model, layer, x)
    cam_single = torch.cat([XA.grad_cam(model, layer, x[i:i + 1])["map"] for i in range(len(sel))])
    ig = {m: timed(f"ig_m{m}_8_images", lambda m=m: XA.integrated_gradients(model, x, base, m, 16)) for m in STEPS}
    ig64_b8 = XA.integrated_gradients(model, x[:2], base[:2], 64, internal_batch=8)
    with torch.no_grad():
        lb = XA._logit(model, base)
    h1 = _state_hash(model)
    # --- per-image diagnostics
    per = []
    for i, uid in enumerate(sel):
        d = {"record_uid": uid, "label_mel": labels[i], "logit": float(cam["logit"][i]),
             "logit_baseline": float(lb[i]),
             "gradcam": {"validity_map224": XA.validity(cam["map"][i]),
                         "validity_raw7": XA.validity(cam["cam_raw"][i])},
             "ig": {}}
        for m in STEPS:
            d["ig"][str(m)] = {"completeness": ig[m]["completeness"][i],
                               "validity_primary": XA.validity(ig[m]["primary"][i]),
                               "validity_absolute": XA.validity(ig[m]["absolute"][i])}
        conv = {}
        for m in (16, 32, 64):
            p, q = ig[m]["primary"][i], ig[2 * m]["primary"][i]
            conv[f"{m}_vs_{2 * m}"] = float((p - q).abs().sum() / max(float(q.abs().sum()), 1e-12))
        d["ig_primary_rel_L1_change_vs_2m"] = conv
        per.append(d)
    fail = {m: sum(ig[m]["completeness"][i]["failure"] for i in range(len(sel))) for m in STEPS}
    summary = {
        "status_note": "software diagnostic; not a research result",
        "model": {"checkpoint": ck_rel, "sha256": ck_sha, "pointer_sha256": ptr["sha256"],
                  "provenance": "Stage 2C3 throughput pilot (ResNet-50 resnet50.a1_in1k, 2 epochs FP32 on full HAM "
                                "TRAIN); its training data includes these 8 images",
                  "target_layer": "model.layer4 (output of layer4[-1], 2048x7x7)",
                  "build_and_load_s": round(build_s, 3)},
        "attr_version": XA.ATTR_VERSION, "baseline": {"k": 57, "sigma": 9.5, "space": "normalised input",
                                                      "padding": "reflect", "max_abs_diff_vs_legacy_gauss_blur": legacy_diff},
        "records": per, "ig_completeness_failures_by_steps": fail,
        "state_unchanged": h0 == h1, "hooks_left_on_layer": len(layer._forward_hooks),
        "param_grads_created": any(p.grad is not None for p in model.parameters()),
        "gradcam_repeat_identical": bool(torch.equal(cam["map"], cam2["map"])),
        "gradcam_batch_vs_single_max_abs_diff": float((cam_single - cam["map"]).abs().max()),
        "ig64_internal_batch_16_vs_8_max_abs_diff": float((ig64_b8["attr"] - ig[64]["attr"][:2]).abs().max()),
        "timings_s": timings, "peak_memory": mem,
        "convergence_note": ("Protocol criterion 'median primary metric change < 0.005 vs 2m' needs "
                             "insertion/deletion curves (not implemented yet): UNRESOLVED here. The 2m "
                             "comparison for m=128 would need m=256, which was NOT run. Shown instead: "
                             "relative L1 change of the IG primary map vs 2m (diagnostic only)."),
    }
    receipt = P.build_receipt(cfg, a.config, "completed", details={"attr_version": XA.ATTR_VERSION})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    arrays = {"x": x.cpu().numpy(), "baseline": base.cpu().numpy(),
              "gradcam_raw7": cam["cam_raw"].cpu().numpy(), "gradcam_map224": cam["map"].cpu().numpy()}
    for m in STEPS:
        arrays[f"ig_attr_m{m}"] = ig[m]["attr"].cpu().numpy()
        arrays[f"ig_primary_m{m}"] = ig[m]["primary"].cpu().numpy()
        arrays[f"ig_absolute_m{m}"] = ig[m]["absolute"].cpu().numpy()
    np.savez_compressed(os.path.join(out, "attributions.npz"), record_uids=np.array(sel), **arrays)
    # review sheet (display normalisation only)
    tiles = []
    for i, uid in enumerate(sel):
        img = np.clip((x[i].cpu().numpy().transpose(1, 2, 0) * np.array(PP.STD) + np.array(PP.MEAN)) * 255,
                      0, 255).astype(np.uint8)

        def heat(mp):
            n = XA.display_normalise(mp)
            return (np.stack([n, np.zeros_like(n), 1 - n], -1) * 255).astype(np.uint8)
        over = lambda h: (0.5 * img + 0.5 * h).astype(np.uint8)
        panels = [img, over(heat(cam["map"][i].cpu().numpy())), over(heat(ig[128]["primary"][i].cpu().numpy())),
                  over(heat(ig[128]["absolute"][i].cpu().numpy()))]
        row = np.concatenate(panels, 1)
        tile = Image.new("RGB", (row.shape[1], row.shape[0] + 16), "white")
        tile.paste(Image.fromarray(row), (0, 16))
        c = ig[128]["completeness"][i]
        ImageDraw.Draw(tile).text((2, 2), f"{uid} y={labels[i]} logit={per[i]['logit']:.2f} base={per[i]['logit_baseline']:.2f} | "
                                          f"image | Grad-CAM | IG primary m128 | IG |abs| m128 | IG e_abs={c['e_abs']:.3f}",
                                  fill="black")
        tiles.append(tile)
    sheet = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)), "white")
    yy = 0
    for tl in tiles:
        sheet.paste(tl, (0, yy))
        yy += tl.height
    sheet.save(os.path.join(out, "review_sheet.png"))
    summary["total_elapsed_s"] = round(time.perf_counter() - t_all, 2)
    with open(os.path.join(out, "diagnostics.json"), "x", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n))
                          for n in ("diagnostics.json", "attributions.npz", "review_sheet.png")}
    rp = P.write_receipt(receipt, out)
    print(rp)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=1))
    for d in per:
        print(d["record_uid"], d["label_mel"], round(d["logit"], 3), round(d["logit_baseline"], 3),
              d["gradcam"]["validity_map224"]["flags"],
              {m: (round(v["completeness"]["e_abs"], 4), v["completeness"]["e_rel"] and round(v["completeness"]["e_rel"], 4),
                   v["completeness"]["failure"], v["validity_primary"]["flags"]) for m, v in d["ig"].items()},
              {k: round(v, 4) for k, v in d["ig_primary_rel_L1_change_vs_2m"].items()})
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


if __name__ == "__main__":
    main()
