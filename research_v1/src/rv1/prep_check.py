"""Stage 2B2C real-data sample check for prep_v1_cached256 (run explicitly).

  cd research_v1/src
  ..\\.venv_prep\\Scripts\\python.exe -B -m rv1.prep_check --config ../configs/prep_check_v1.json --out-root ../runs/prep_check

Deterministic sample (stable_seed("prep_check_v1", record_uid) order):
  * 3 records from each of the 7 legacy caches;
  * HAM test: 3 palette ("P") and 3 "L" masks (modes from the Stage 2B2B audit CSV);
  * 2 PAD and 2 DDI images whose source files are RGBA (Stage 2B2A audit CSV);
  * the Derm7pt same-file pair with the smallest stable key.
For each: legacy cache entry vs reconstruction of the documented cache build from the
raw file; evaluation input dtype/shape/finiteness and repeat-read identity; diagnostic
raw->224 difference; Albumentations Resize vs explicit cv2 INTER_LINEAR; for HAM,
mask_interp_v1 grid vs legacy cached mask. Nothing is tuned to pass.
"""
import argparse
import csv
import json
import os
import time

from . import config as C, decode as DC, preprocess as PP, provenance as P
from .seeds import stable_seed

CACHES = {"ham10000": ("ham_train", "ham_val", "ham_test"),
          "derm7pt": ("derm7pt_clinical", "derm7pt_dermoscopy"),
          "pad_ufes_20": ("pad_ufes_20",), "ddi": ("ddi",)}


def _key(uid):
    return (stable_seed("prep_check_v1", uid), uid)


def _csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select(rows_by_ds, mask_csv, decode_csv):
    sel = []
    for ds, caches in CACHES.items():
        for c in caches:
            rr = sorted((r for r in rows_by_ds[ds] if r["cache_name"] == c),
                        key=lambda r: _key(r["record_uid"]))
            sel += [(ds, c, r, f"cache_sample:{c}") for r in rr[:3]]
    mode = {m["record_uid"]: m["original_mode"] for m in mask_csv}
    ham_test = sorted((r for r in rows_by_ds["ham10000"] if r["cache_name"] == "ham_test"),
                      key=lambda r: _key(r["record_uid"]))
    for md in ("P", "L"):
        sel += [("ham10000", "ham_test", r, f"mask_mode:{md}")
                for r in [r for r in ham_test if mode.get(r["record_uid"]) == md][:3]]
    rgba = {(d["dataset"], d["rel_path"]) for d in decode_csv
            if d["kind"] == "image" and d["mode"] == "RGBA"}
    for ds in ("pad_ufes_20", "ddi"):
        rr = sorted((r for r in rows_by_ds[ds] if (ds, r["rel_path"]) in rgba),
                    key=lambda r: _key(r["record_uid"]))
        sel += [(ds, ds, r, "source_rgba") for r in rr[:2]]
    pairs = sorted({r["pair_id"] for r in rows_by_ds["derm7pt"] if r["same_file_pair"] == "1"},
                   key=_key)
    for r in rows_by_ds["derm7pt"]:
        if r["pair_id"] == pairs[0]:
            sel.append(("derm7pt", r["cache_name"], r, "same_file_pair"))
    seen, out = set(), []
    for s in sel:  # keep first reason per record, preserve order
        if s[2]["record_uid"] not in seen:
            seen.add(s[2]["record_uid"])
            out.append(s)
    return out


def _artifact(items, path):
    import numpy as np
    from PIL import Image, ImageDraw
    tiles = []
    for it in items:
        x = it["x"].transpose(1, 2, 0) * np.array(PP.STD, np.float32) + np.array(PP.MEAN, np.float32)
        img = np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)       # display only
        m = it["mask"]
        ov = img.astype(np.float32)
        ov[m] = ov[m] * 0.55 + np.array([0, 255, 0], np.float32) * 0.45
        edge = m ^ np.roll(m, 1, 0) | m ^ np.roll(m, 1, 1)
        ov[edge] = (255, 255, 0)
        row = np.concatenate([img, np.repeat(m[..., None].astype(np.uint8) * 255, 3, 2),
                              ov.astype(np.uint8)], 1)
        tile = Image.new("RGB", (row.shape[1], row.shape[0] + 16), "white")
        tile.paste(Image.fromarray(row), (0, 16))
        ImageDraw.Draw(tile).text((2, 2), f"{it['record']['record_uid']}  mode={it['mode']}  "
                                          f"(224 input de-normalised | v1 mask | overlay)",
                                  fill="black")
        tiles.append(tile)
    sheet = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)), "white")
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height
    sheet.save(path)


def run(config_path, out_root):
    import cv2
    import numpy as np
    import albumentations as A
    t0 = time.time()
    cfg = C.load(config_path)
    if cfg["purpose"] != "prep_check" or cfg["versions"].get("prep_version") != PP.PREP_VERSION:
        raise C.ConfigError("prep_check requires purpose 'prep_check' and prep_v1_cached256")
    root = cfg["project_root"]
    ds_root, cache_dir = os.path.join(root, "DATASETS"), os.path.join(root, "cache")
    rows_by_ds = {ds: DC.read_manifest(C.resolve(cfg, cfg["manifests"][ds]), ds)[0]
                  for ds in CACHES}
    sel = select(rows_by_ds, _csv(C.resolve(cfg, cfg["supporting_inputs"]["mask_audit_csv"])),
                 _csv(C.resolve(cfg, cfg["supporting_inputs"]["decode_audit_csv"])))
    readers, results, art = {}, [], []
    alb = A.Resize(224, 224)                     # default interpolation (diagnostic only)
    for ds, cache, r, why in sel:
        key = (cache, ds == "ham10000")
        if key not in readers:
            readers[key] = PP.EvalReader(rows_by_ds[ds], cache, cache_dir, ds_root,
                                         want_mask=(ds == "ham10000"))
        rd = readers[key]
        k = next(i for i, q in enumerate(rd.rows) if q["record_uid"] == r["record_uid"])
        it, it2 = rd[k], rd[k]
        cached = np.asarray(rd.img[int(r["cache_index"])])
        rebuilt = PP.rebuild_cache_entry(os.path.join(ds_root, *r["rel_path"].split("/")))
        d = np.abs(cached.astype(np.int16) - rebuilt.astype(np.int16))
        raw = cv2.cvtColor(cv2.imread(os.path.join(ds_root, *r["rel_path"].split("/")),
                                      cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        raw224 = cv2.resize(raw, (224, 224), interpolation=cv2.INTER_AREA)
        x_raw = ((raw224.astype(np.float32) / 255 - np.array(PP.MEAN, np.float32))
                 / np.array(PP.STD, np.float32)).transpose(2, 0, 1)
        a_res = alb(image=np.ascontiguousarray(cached))["image"]
        c_res = cv2.resize(np.ascontiguousarray(cached), (224, 224), interpolation=cv2.INTER_LINEAR)
        res = {"record_uid": r["record_uid"], "image_uid": r["image_uid"],
               "pair_id": r.get("pair_id", ""), "dataset": ds, "cache": cache,
               "cache_index": int(r["cache_index"]), "reason": why,
               "cache_vs_rebuild_max_abs": int(d.max()),
               "cache_vs_rebuild_n_diff_values": int((d > 0).sum()),
               "x_dtype": str(it["x"].dtype), "x_shape": list(it["x"].shape),
               "x_finite": bool(np.isfinite(it["x"]).all()),
               "x_min": float(it["x"].min()), "x_max": float(it["x"].max()),
               "repeat_read_identical": bool(np.array_equal(it["x"], it2["x"])),
               "diag_raw224_vs_cached224_max_abs_norm": float(np.abs(x_raw - it["x"]).max()),
               "diag_raw224_vs_cached224_mean_abs_norm": float(np.abs(x_raw - it["x"]).mean()),
               "diag_albumentations_resize_equals_cv2_linear": bool(np.array_equal(a_res, c_res)),
               "prep_version": it["prep_version"]}
        if it["mask"] is not None:
            legacy = np.load(os.path.join(cache_dir, f"{cache}_mask.npy"), mmap_mode="r")
            leg = np.asarray(legacy[int(r["cache_index"])]) > 127
            les, mm = DC.interpret_mask(os.path.join(ds_root, *r["mask_path"].split("/")))
            v1_256 = cv2.resize(les.astype(np.uint8), (256, 256),
                                interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
            res.update(mask_mode=mm["original_mode"], mask_flags=mm["flags"],
                       mask_interp_version=it["mask_interp_version"],
                       mask_shape=list(it["mask"].shape), mask_dtype=str(it["mask"].dtype),
                       mask_fraction_224=float(it["mask"].mean()),
                       diag_v1_256_vs_legacy_cached_mask_diff_fraction=float((v1_256 != leg).mean()),
                       diag_legacy_cached_mask_values=sorted(int(v) for v in np.unique(legacy[int(r["cache_index"])])))
            if r["cache_name"] == "ham_test":
                art.append(dict(it, mode=mm["original_mode"]))
        results.append(res)
    pair = [x for x in results if x["reason"] == "same_file_pair"]
    pair_equal = None
    if len(pair) == 2:
        xs = []
        for x in pair:
            rd = readers[(x["cache"], False)]
            xs.append(rd[next(i for i, q in enumerate(rd.rows) if q["record_uid"] == x["record_uid"])]["x"])
        pair_equal = bool(np.array_equal(xs[0], xs[1]))

    receipt = P.build_receipt(cfg, config_path, "completed",
                              details={"prep_version": PP.PREP_VERSION,
                                       "mask_interp_version": DC.MASK_INTERP_VERSION,
                                       "cv2": cv2.__version__, "numpy": np.__version__,
                                       "albumentations": A.__version__})
    out_dir = os.path.join(out_root, receipt["run_id"])
    os.makedirs(out_dir)
    # alignment sheet: first 3 P-mode and first 3 L-mode HAM-test records of the sample
    art = [a for a in art if a["mode"] == "P"][:3] + [a for a in art if a["mode"] == "L"][:3]
    _artifact(art, os.path.join(out_dir, "alignment_sheet.png"))
    summary = {"prep_version": PP.PREP_VERSION, "n_records": len(results),
               "all_x_float32_3x224x224_finite": all(x["x_dtype"] == "float32" and
                                                     x["x_shape"] == [3, 224, 224] and x["x_finite"]
                                                     for x in results),
               "all_repeat_reads_identical": all(x["repeat_read_identical"] for x in results),
               "cache_vs_rebuild_exact": sum(x["cache_vs_rebuild_max_abs"] == 0 for x in results),
               "cache_vs_rebuild_max_abs_over_sample": max(x["cache_vs_rebuild_max_abs"] for x in results),
               "albumentations_equals_cv2_linear": sum(x["diag_albumentations_resize_equals_cv2_linear"]
                                                      for x in results),
               "same_file_pair_inputs_identical": pair_equal,
               "alignment_sheet_records": [a["record"]["record_uid"] + ":" + a["mode"] for a in art],
               "elapsed_seconds": round(time.time() - t0, 1), "records": results}
    with open(os.path.join(out_dir, "sample_check.json"), "x", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out_dir, n))
                          for n in ("sample_check.json", "alignment_sheet.png")}
    rp = P.write_receipt(receipt, out_dir)
    return rp, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    rp, s = run(a.config, a.out_root)
    print(rp)
    print(json.dumps({k: v for k, v in s.items() if k != "records"}, indent=1))
    print("verify:", P.verify_receipt(rp, C.load(a.config)["project_root"]) or "OK")
