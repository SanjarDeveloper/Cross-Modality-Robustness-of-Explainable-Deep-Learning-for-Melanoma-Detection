"""Stage 2B2D real-data augmentation review (HAM TRAIN only; no model).

  cd research_v1/src
  ..\\.venv_prep\\Scripts\\python.exe -B -m rv1.aug_review --config ../configs/aug_review_v1.json --out-root ../runs/aug_review

Sample: HAM train records with status "ok" in the Stage 2B2B mask audit; 3 P-mode and
3 L-mode masks, ordered by stable_seed("aug_review_v1", record_uid).
For each: aug_none, then aug_std_v1 at (train_seed, epoch) in (0,0), (0,1), (1,0).
Writes review_sheet.png, review.json and a receipt. Validation/test data are not used.
"""
import argparse
import csv
import json
import os
import time

from . import augment as AU, config as C, decode as DC, preprocess as PP, provenance as P
from .seeds import stable_seed

SEEDS = ((0, 0), (0, 1), (1, 0))


def _panel(x, m):
    import numpy as np
    img = np.clip(np.rint((x.transpose(1, 2, 0) * np.array(PP.STD, np.float32)
                           + np.array(PP.MEAN, np.float32)) * 255), 0, 255).astype(np.uint8)
    ov = img.astype(np.float32)
    ov[m] = ov[m] * 0.55 + np.array([0, 255, 0], np.float32) * 0.45
    edge = (m ^ np.roll(m, 1, 0)) | (m ^ np.roll(m, 1, 1))
    ov[edge] = (255, 255, 0)
    return img, ov.astype(np.uint8)


def run(config_path, out_root):
    import numpy as np
    from PIL import Image, ImageDraw
    t0 = time.time()
    cfg = C.load(config_path)
    if cfg["purpose"] != "aug_review" or cfg["versions"].get("aug_version") != "aug_std_v1":
        raise C.ConfigError("aug_review requires purpose 'aug_review' and aug_std_v1")
    root = cfg["project_root"]
    ds_root, cache_dir = os.path.join(root, "DATASETS"), os.path.join(root, "cache")
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    with open(C.resolve(cfg, cfg["supporting_inputs"]["mask_audit_csv"]), newline="",
              encoding="utf-8") as f:
        mode = {m["record_uid"]: m["original_mode"] for m in csv.DictReader(f)
                if m["split"] == "train" and m["status"] == "ok"}
    train = sorted((r for r in rows if r["split"] == "train" and r["record_uid"] in mode),
                   key=lambda r: (stable_seed("aug_review_v1", r["record_uid"]), r["record_uid"]))
    sel = [r for r in train if mode[r["record_uid"]] == "P"][:3] + \
          [r for r in train if mode[r["record_uid"]] == "L"][:3]
    reader = PP.EvalReader(rows, "ham_train", cache_dir)          # validates cache alignment
    by_uid = {q["record_uid"]: q for q in reader.rows}
    tiles, results = [], []
    for r in sel:
        q = by_uid[r["record_uid"]]
        img256 = np.asarray(reader.img[int(q["cache_index"])])
        les, mm = DC.interpret_mask(os.path.join(ds_root, *r["mask_path"].split("/")))
        m256 = AU.mask_to_cache_grid(les)
        outs = [AU.augment_record(r, img256, "aug_none", mask256=m256, mask_flags=mm["flags"])]
        outs += [AU.augment_record(r, img256, "aug_std_v1", ts, ep, m256, mm["flags"])
                 for ts, ep in SEEDS]
        panels, info = [], []
        for o in outs:
            img, ov = _panel(o["x"], o["mask"])
            panels += [img, ov]
            info.append({"recipe": o["recipe"], "train_seed": o["train_seed"], "epoch": o["epoch"],
                         "aug_seed": o["aug_seed"], "mask_fraction": float(o["mask"].mean()),
                         "mask_flags": o["mask_flags"],
                         "record_unchanged": o["record"] == r, "x_shape": list(o["x"].shape),
                         "x_dtype": str(o["x"].dtype)})
        row = np.concatenate(panels, 1)
        tile = Image.new("RGB", (row.shape[1], row.shape[0] + 16), "white")
        tile.paste(Image.fromarray(row), (0, 16))
        ImageDraw.Draw(tile).text((2, 2), f"{r['record_uid']} mode={mode[r['record_uid']]} | pairs: "
                                          "aug_none, std(0,0), std(0,1), std(1,0) [image | overlay]",
                                  fill="black")
        tiles.append(tile)
        results.append({"record_uid": r["record_uid"], "mask_mode": mode[r["record_uid"]],
                        "label_mel": r["label_mel"], "split": r["split"], "outputs": info})
    receipt = P.build_receipt(cfg, config_path, "completed",
                              details={"aug_version": "aug_std_v1", "deferred": list(AU.DEFERRED),
                                       "prep_version": PP.PREP_VERSION,
                                       "mask_interp_version": DC.MASK_INTERP_VERSION})
    out_dir = os.path.join(out_root, receipt["run_id"])
    os.makedirs(out_dir)
    sheet = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)), "white")
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height
    sheet.save(os.path.join(out_dir, "review_sheet.png"))
    summary = {"records": results, "seeds": SEEDS, "elapsed_seconds": round(time.time() - t0, 1),
               "visual_review_status": "generated; see STAGE2B2D_REPORT.md"}
    with open(os.path.join(out_dir, "review.json"), "x", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out_dir, n))
                          for n in ("review.json", "review_sheet.png")}
    return P.write_receipt(receipt, out_dir), summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    rp, s = run(a.config, a.out_root)
    print(rp)
    for r in s["records"]:
        print(r["record_uid"], r["mask_mode"], r["label_mel"],
              [(o["recipe"], o["train_seed"], o["epoch"], round(o["mask_fraction"], 3), o["mask_flags"],
                o["record_unchanged"]) for o in r["outputs"]])
    print("verify:", P.verify_receipt(rp, C.load(a.config)["project_root"]) or "OK")
