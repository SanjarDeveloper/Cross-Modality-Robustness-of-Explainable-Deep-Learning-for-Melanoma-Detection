"""Stage 2B2A full decode audit (run explicitly; importing does nothing).

  cd research_v1/src
  python -B -m rv1.decode_audit --config ../configs/data_audit_v1.json --out-root ../runs/data_audit

Streams every UNIQUE physical image file referenced by the four frozen manifests
(one decode per file; all referencing record_uids are listed), re-hashes each
file against the manifest sha256, and decodes every HAM mask with checks against
its image size. Writes decode_audit.csv, summary.json and a receipt (purpose
data_audit) in a new run directory. Nothing is repaired, excluded or rewritten.
"""
import argparse
import csv
import io
import json
import os
import time
from collections import Counter, OrderedDict

from . import config as C, decode as DC, provenance as P

AUDIT_COLUMNS = ["kind", "dataset", "rel_path", "n_reference_rows", "record_uids",
                 "status", "reason", "sha256_matches_manifest", "width", "height", "mode",
                 "format", "exif_orientation", "converted_from", "mask_values",
                 "mask_foreground_fraction", "mask_palette", "mask_flags", "image_rel_path"]
DATASET_MANIFESTS = ("ham10000", "derm7pt", "pad_ufes_20", "ddi")


def _peak_rss_mib():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        v = getattr(mi, "peak_wset", None)  # Windows peak working set
        return round(v / 2 ** 20, 1) if v else None
    except Exception:  # noqa: BLE001
        return None


def run(config_path, out_root):
    t0 = time.time()
    cfg = C.load(config_path)
    if cfg["purpose"] != "data_audit":
        raise C.ConfigError("decode_audit requires purpose 'data_audit'")
    import PIL
    ds_root = os.path.join(cfg["project_root"], "DATASETS")
    rows_out, problems = [], []
    counts = OrderedDict()
    image_meta_by_path = {}

    for ds in DATASET_MANIFESTS:
        rows, _ = DC.read_manifest(C.resolve(cfg, cfg["manifests"][ds]), ds)
        files = OrderedDict()
        for r in rows:
            files.setdefault(r["rel_path"], []).append(r)
        c = Counter()
        c["reference_rows"] = len(rows)
        c["unique_files"] = len(files)
        for rel, refs in files.items():
            p = os.path.join(ds_root, *rel.split("/"))
            shas = {r["sha256"] for r in refs}
            out = {"kind": "image", "dataset": ds, "rel_path": rel,
                   "n_reference_rows": len(refs),
                   "record_uids": ";".join(r["record_uid"] for r in refs)}
            try:
                sha_ok = os.path.isfile(p) and len(shas) == 1 and P.sha256_file(p) in shas
                _, meta = DC.decode_image(p)
                out.update(status="ok", sha256_matches_manifest=int(sha_ok),
                           **{k: meta[k] for k in ("width", "height", "mode", "format",
                                                   "exif_orientation", "converted_from")})
                image_meta_by_path[rel] = meta
                c["decoded_ok"] += 1
                c[f"mode:{meta['mode']}"] += 1
                if meta["exif_orientation"] not in (None, 1):
                    c["exif_orientation_not_1"] += 1
                if not sha_ok:
                    out.update(status="critical", reason="sha256_mismatch_vs_manifest")
                    problems.append({"dataset": ds, "record_uids": out["record_uids"],
                                     "path": rel, "reason": out["reason"], "critical": True})
            except DC.DecodeError as e:
                out.update(status="critical", reason=str(e))
                c["decode_failed"] += 1
                problems.append({"dataset": ds, "record_uids": out["record_uids"], "path": rel,
                                 "reason": str(e), "critical": True})
            rows_out.append(out)
        counts[ds] = dict(c)

        if ds == "ham10000":
            m = Counter()
            for r in rows:
                mp = r.get("mask_path", "")
                out = {"kind": "mask", "dataset": ds, "rel_path": mp, "n_reference_rows": 1,
                       "record_uids": r["record_uid"], "image_rel_path": r["rel_path"]}
                m["records"] += 1
                if not mp:
                    out.update(status="critical", reason="mask_path_empty")
                else:
                    try:
                        _, mm = DC.decode_mask(os.path.join(ds_root, *mp.split("/")))
                        im = image_meta_by_path.get(r["rel_path"])
                        crit, warn = (DC.mask_flags(mm, im) if im
                                      else (["image_not_decoded"], []))
                        out.update(width=mm["width"], height=mm["height"], mode=mm["mode"],
                                   format=mm["format"],
                                   mask_values=" ".join(map(str, mm.get("values") or [])),
                                   mask_foreground_fraction=mm.get("foreground_fraction"),
                                   mask_palette=json.dumps(mm.get("palette_rgb_of_values"))
                                   if mm.get("palette_rgb_of_values") else "",
                                   mask_flags=";".join(crit + warn))
                        m["decoded_ok"] += 1
                        m[f"mode:{mm['mode']}"] += 1
                        m["values:" + ",".join(map(str, mm.get("values") or []))] += 1
                        for w in warn:
                            m[f"warning:{w}"] += 1
                        if crit:
                            out.update(status="critical", reason=";".join(crit))
                        else:
                            out.update(status="warning" if warn else "ok")
                    except DC.DecodeError as e:
                        m["decode_failed"] += 1
                        out.update(status="critical", reason=str(e))
                if out["status"] == "critical":
                    m["critical"] += 1
                    problems.append({"dataset": ds, "record_uids": r["record_uid"], "path": mp,
                                     "reason": out["reason"], "critical": True})
                elif out["status"] == "warning":
                    problems.append({"dataset": ds, "record_uids": r["record_uid"], "path": mp,
                                     "reason": out["mask_flags"], "critical": False})
                rows_out.append(out)
            counts["ham10000_masks"] = dict(m)

    n_crit = sum(p["critical"] for p in problems)
    summary = {
        "status": "PASS" if n_crit == 0 else "BLOCKED",
        "decoder": {"library": "Pillow", "version": PIL.__version__,
                    "LOAD_TRUNCATED_IMAGES": __import__("PIL.ImageFile").ImageFile.LOAD_TRUNCATED_IMAGES,
                    "MAX_IMAGE_PIXELS": __import__("PIL.Image").Image.MAX_IMAGE_PIXELS,
                    "opencv_imported": False},
        "counts": counts,
        "totals": {"reference_rows": sum(v["reference_rows"] for k, v in counts.items()
                                         if k in DATASET_MANIFESTS),
                   "unique_image_files": sum(v["unique_files"] for k, v in counts.items()
                                             if k in DATASET_MANIFESTS),
                   "images_decoded_ok": sum(v.get("decoded_ok", 0) for k, v in counts.items()
                                            if k in DATASET_MANIFESTS),
                   "masks_decoded_ok": counts["ham10000_masks"].get("decoded_ok", 0),
                   "critical_problems": n_crit,
                   "warnings": len(problems) - n_crit},
        "problems": problems,
        "elapsed_seconds": round(time.time() - t0, 1),
        "peak_working_set_mib": _peak_rss_mib(),
    }

    receipt = P.build_receipt(cfg, config_path, "completed",
                              details={"decoder": summary["decoder"],
                                       "audit_status": summary["status"]})
    out_dir = os.path.join(out_root, receipt["run_id"])
    os.makedirs(out_dir)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=AUDIT_COLUMNS, lineterminator="\n", extrasaction="raise")
    w.writeheader()
    w.writerows(rows_out)
    for name, data in (("decode_audit.csv", buf.getvalue()),
                       ("summary.json", json.dumps(summary, indent=1, default=str))):
        with open(os.path.join(out_dir, name), "x", encoding="utf-8", newline="\n") as f:
            f.write(data)
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out_dir, n))
                          for n in ("decode_audit.csv", "summary.json")}
    rp = P.write_receipt(receipt, out_dir)
    return rp, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    rp, s = run(a.config, a.out_root)
    print(rp)
    print(json.dumps({k: s[k] for k in ("status", "totals", "elapsed_seconds",
                                        "peak_working_set_mib")}, indent=1))
    print("verify:", P.verify_receipt(rp, C.load(a.config)["project_root"]) or "OK")
