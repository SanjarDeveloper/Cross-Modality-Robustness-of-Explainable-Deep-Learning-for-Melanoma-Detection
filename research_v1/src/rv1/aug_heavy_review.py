"""Stage 9D bounded review and throughput check for aug_heavy_v1 (HAM TRAIN only; no model).

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.aug_heavy_review --config ../configs/aug_heavy_review_9d_v1.json --out-root ../runs/aug_heavy_review_9d

Sample (fixed before viewing): HAM TRAIN, first record per (label, mask mode) in
(stable_seed('aug_heavy_review_v1', record_uid), record_uid) order -> 4 images; sheets show the original
and 6 heavy samples (train_seed 0, epochs 0..5) with mask boundary overlays.
Throughput: aug_std_v1 vs aug_heavy_v1 through torch_data.ManifestDataset + make_epoch_loader on the same
fixed 512-image HAM TRAIN sample, batch 16, workers 0, 2 repeats each, wall-clock with perf_counter.
"""
import argparse
import io
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    import numpy as np
    from PIL import Image, ImageDraw
    from . import augment as AU, config as C, decode as DC, preprocess as PP, provenance as P, torch_data as TD
    from .desc_4b3 import publish_bytes, publish_json
    from .seeds import stable_seed
    cfg = C.load(a.config)
    spec = json.load(open(C.resolve(cfg, cfg["supporting_inputs"]["heavy_spec"]), encoding="utf-8"))
    if P.sha256_file(os.path.join(os.path.dirname(__file__), "augment.py")) != spec["module_sha256"]:
        raise RuntimeError("augment.py differs from the frozen aug_heavy_v1 specification")
    root = cfg["project_root"]
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    train = sorted((r for r in rows if r["split"] == "train" and r["mask_path"]),
                   key=lambda r: (stable_seed("aug_heavy_review_v1", r["record_uid"]), r["record_uid"]))
    chosen = {}
    for r in train:
        les, meta = DC.interpret_mask(os.path.join(root, "DATASETS", *r["mask_path"].split("/")))
        k = (r["label_mel"], meta["original_mode"])
        if k not in chosen:
            chosen[k] = (r, les, meta)
        if len(chosen) == 4:
            break
    reader = PP.EvalReader(rows, "ham_train", os.path.join(root, "cache"))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"stage": "9D review"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    T = 150
    tiles, flags_seen = [], {}
    for key in sorted(chosen):
        r, les, meta = chosen[key]
        img = np.asarray(reader.img[int(r["cache_index"])]).copy()
        m256 = AU.mask_to_cache_grid(les)
        cells = [(img, m256.astype(bool), "original 256")]
        for ep in range(6):
            o = AU.augment(img, "aug_heavy_v1", 0, ep, r["image_uid"], m256)
            x = o["x"].transpose(1, 2, 0) * np.asarray(PP.STD) + np.asarray(PP.MEAN)
            cells.append((np.clip(x * 255, 0, 255).astype(np.uint8), o["mask"], f"heavy ep{ep}"))
            flags_seen[f"{r['record_uid']}|ep{ep}"] = o["mask_flags"]
        row = Image.new("RGB", (T * len(cells), T + 28), "white")
        d = ImageDraw.Draw(row)
        for j, (im, mk, lab) in enumerate(cells):
            tile = np.asarray(Image.fromarray(im).resize((T, T), Image.BILINEAR)).copy()
            mk2 = np.asarray(Image.fromarray(mk.astype(np.uint8) * 255).resize((T, T), Image.NEAREST)) > 0
            edge = mk2 & ~(np.roll(mk2, 1, 0) & np.roll(mk2, -1, 0) & np.roll(mk2, 1, 1) & np.roll(mk2, -1, 1))
            tile[edge] = (0, 255, 0)
            row.paste(Image.fromarray(tile), (j * T, 14))
            d.text((j * T + 2, 1), lab, fill="black")
        d.text((2, T + 15), f"{r['record_uid']} y={r['label_mel']} mask={meta['original_mode']}", fill="black")
        tiles.append(row)
    sheet = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)), "white")
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    publish_bytes(out, "review_aug_heavy_v1.png", buf.getvalue())
    # throughput
    sample = [r["record_uid"] for r in sorted((r for r in rows if r["split"] == "train"),
                                              key=lambda r: (stable_seed("aug_heavy_timing_v1", r["record_uid"]), r["record_uid"]))[:512]]
    timing = {}
    for recipe in ("aug_std_v1", "aug_heavy_v1", "aug_std_v1", "aug_heavy_v1"):
        ds = TD.ManifestDataset(rows, "ham_train", os.path.join(root, "cache"), os.path.join(root, "DATASETS"), sample,
                                mode="train", recipe=recipe, train_seed=0)
        t0 = time.perf_counter()
        n, shapes_ok = 0, True
        for b in TD.make_epoch_loader(ds, 0, 16, True, 20260927, 0):
            n += len(b["label"])
            x = b["x"]
            shapes_ok &= tuple(x.shape[1:]) == (3, 224, 224) and str(x.dtype) == "torch.float32" and bool(x.isfinite().all())
        dt = time.perf_counter() - t0
        timing.setdefault(recipe, []).append({"images": n, "seconds": dt, "images_per_s": n / dt, "tensor_contract_ok": shapes_ok})
    ratio = (np.mean([t["seconds"] for t in timing["aug_heavy_v1"]]) / np.mean([t["seconds"] for t in timing["aug_std_v1"]]))
    publish_json(out, "review_and_timing.json", {
        "sample": [{"record_uid": chosen[k][0]["record_uid"], "label": k[0], "mask_mode": k[1]} for k in sorted(chosen)],
        "mask_flags": flags_seen, "throughput": timing, "heavy_over_std_time_ratio": float(ratio),
        "note": "augmentation + DataLoader throughput only (workers 0); NOT a measured heavy-training time"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))}
    rp = P.write_receipt(receipt, out)
    print(out, json.dumps(timing, indent=0), ratio, "\nverify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


if __name__ == "__main__":
    main()
