"""Stage 2B2B: validate HAM mask interpretation (mask_interp_v1) + visual review sheets.

  cd research_v1/src
  python -B -m rv1.mask_audit --config ../configs/mask_audit_v1.json --out-root ../runs/mask_audit

Reads the frozen HAM manifest; for every record hashes the mask file, interprets
it with decode.interpret_mask, and records results. Writes, in a new run dir:
mask_interpretation.csv, summary.json, review_selection.json, review_page_*.png
and a receipt. Source masks are never modified. No localisation metric is computed.

Review sample (stable, recorded): P masks (12) and L masks (6, excluding
all-foreground) are split into foreground-fraction tertiles of their own group;
within each tertile records are ordered by stable_seed("mask_review_v1",
record_uid) and the first 4 (P) / 2 (L) are taken. Both all-foreground masks
are always added.
"""
import argparse
import csv
import io
import json
import os
import time
from collections import Counter

from . import config as C, decode as DC, provenance as P
from .seeds import stable_seed

COLUMNS = ["record_uid", "image_rel_path", "mask_rel_path", "mask_sha256", "label_mel", "dx_raw",
           "split", "original_mode", "palette_map", "rendered_values", "foreground_fraction",
           "flags", "status", "reason", "interp_version"]


def _tertile_pick(rows, k, tag):
    rows = sorted(rows, key=lambda r: (r["_frac"], r["record_uid"]))
    n = len(rows)
    out = []
    for t, name in enumerate(("small", "medium", "large")):
        seg = rows[t * n // 3:(t + 1) * n // 3]
        seg = sorted(seg, key=lambda r: (stable_seed("mask_review_v1", r["record_uid"]),
                                         r["record_uid"]))
        out += [(name, tag, r) for r in seg[:k]]
    return out


def _render_pages(sel, ds_root, out_dir, per_page=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    pages = []
    for pi in range(0, len(sel), per_page):
        chunk = sel[pi:pi + per_page]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(10.5, 2.9 * len(chunk)), dpi=90)
        axes = np.atleast_2d(axes)
        for ax_row, s in zip(axes, chunk):
            img, _ = DC.decode_image(os.path.join(ds_root, *s["image_rel_path"].split("/")))
            les, meta = DC.interpret_mask(os.path.join(ds_root, *s["mask_rel_path"].split("/")))
            ov = img.astype(np.float32).copy()
            ov[les] = ov[les] * 0.55 + np.array([0, 255, 0], np.float32) * 0.45
            ax_row[0].imshow(img)
            ax_row[1].imshow(les.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)
            ax_row[2].imshow(ov.astype(np.uint8))
            ax_row[2].contour(les.astype(np.uint8), levels=[0.5], colors="yellow", linewidths=0.8)
            ax_row[0].set_title(f"{s['record_uid'].split(':')[1]} [{s['group']}/{s['tertile']}]",
                                fontsize=8)
            ax_row[1].set_title(f"rendered mask (white=lesion) mode={meta['original_mode']}",
                                fontsize=8)
            ax_row[2].set_title(f"overlay fg={meta['foreground_fraction']:.3f}", fontsize=8)
            for a in ax_row:
                a.axis("off")
        fig.tight_layout()
        name = f"review_page_{pi // per_page + 1}.png"
        fig.savefig(os.path.join(out_dir, name))
        plt.close(fig)
        pages.append(name)
    return pages


def run(config_path, out_root):
    t0 = time.time()
    cfg = C.load(config_path)
    if cfg["purpose"] != "mask_audit":
        raise C.ConfigError("mask_audit requires purpose 'mask_audit'")
    ds_root = os.path.join(cfg["project_root"], "DATASETS")
    rows, _ = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")
    out, cnt = [], Counter()
    for r in rows:
        mp = r.get("mask_path", "")
        o = {"record_uid": r["record_uid"], "image_rel_path": r["rel_path"], "mask_rel_path": mp,
             "label_mel": r["label_mel"], "dx_raw": r["dx_raw"], "split": r["split"],
             "interp_version": DC.MASK_INTERP_VERSION}
        if not mp:
            o.update(status="rejected", reason="mask_path_empty")
        else:
            p = os.path.join(ds_root, *mp.split("/"))
            try:
                o["mask_sha256"] = P.sha256_file(p)
                les, m = DC.interpret_mask(p)
                o.update(original_mode=m["original_mode"],
                         palette_map=json.dumps(m["palette_map"]) if m["palette_map"] else "",
                         rendered_values=" ".join(map(str, m["rendered_values"])),
                         foreground_fraction=repr(m["foreground_fraction"]),
                         flags=";".join(m["flags"]),
                         status="flagged" if m["flags"] else "ok")
                cnt[f"mode:{m['original_mode']}"] += 1
                cnt[f"mode:{m['original_mode']}|rendered:{o['rendered_values']}"] += 1
                if m["palette_map"]:
                    cnt["palette:" + o["palette_map"]] += 1
                for f in m["flags"]:
                    cnt[f"flag:{f}"] += 1
            except (DC.DecodeError, DC.MaskInterpretationError, OSError) as e:
                o.update(status="rejected", reason=str(e))
        cnt[f"status:{o['status']}"] += 1
        out.append(o)

    import numpy as np
    fr = {}
    for mode in ("L", "P"):
        v = np.array([float(o["foreground_fraction"]) for o in out
                      if o.get("original_mode") == mode and o["status"] != "rejected"])
        fr[mode] = {"n": int(v.size), **{f"p{q}": round(float(np.percentile(v, q)), 4)
                                         for q in (0, 5, 25, 50, 75, 95, 100)}} if v.size else {}
    flagged = [{k: o[k] for k in ("record_uid", "label_mel", "dx_raw", "split", "flags",
                                  "original_mode")} for o in out if o["status"] == "flagged"]

    # --- deterministic visual review sample
    ok = [dict(o, _frac=float(o["foreground_fraction"])) for o in out if o["status"] == "ok"]
    sel = (_tertile_pick([o for o in ok if o["original_mode"] == "P"], 4, "P")
           + _tertile_pick([o for o in ok if o["original_mode"] == "L"], 2, "L"))
    sel = [dict(r, tertile=t, group=g) for t, g, r in sel]
    sel += [dict(o, tertile="all_white", group=o["original_mode"]) for o in out
            if "all_foreground" in (o.get("flags") or "")]

    receipt = P.build_receipt(cfg, config_path, "completed",
                              details={"interp_version": DC.MASK_INTERP_VERSION})
    out_dir = os.path.join(out_root, receipt["run_id"])
    os.makedirs(out_dir)
    pages = _render_pages(sel, ds_root, out_dir)
    selection = [{"record_uid": s["record_uid"], "group": s["group"], "tertile": s["tertile"],
                  "foreground_fraction": s["foreground_fraction"], "label_mel": s["label_mel"],
                  "split": s["split"]} for s in sel]
    n_rej = cnt["status:rejected"]
    summary = {"status": "PASS" if n_rej == 0 else "BLOCKED",
               "interp_version": DC.MASK_INTERP_VERSION,
               "records": len(out), "counts": dict(sorted(cnt.items())),
               "foreground_fraction_by_mode": fr, "flagged": flagged,
               "rejected": [{"record_uid": o["record_uid"], "reason": o["reason"]}
                            for o in out if o["status"] == "rejected"],
               "review_pages": pages, "review_sample_size": len(selection),
               "visual_review_status": "artifact generated; see report for review status",
               "elapsed_seconds": round(time.time() - t0, 1)}
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n", restval="")
    w.writeheader()
    w.writerows(out)
    files = {"mask_interpretation.csv": buf.getvalue(),
             "summary.json": json.dumps(summary, indent=1),
             "review_selection.json": json.dumps(selection, indent=1)}
    for name, data in files.items():
        with open(os.path.join(out_dir, name), "x", encoding="utf-8", newline="\n") as f:
            f.write(data)
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out_dir, n))
                          for n in list(files) + pages}
    rp = P.write_receipt(receipt, out_dir)
    return rp, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    rp, s = run(a.config, a.out_root)
    print(rp)
    print(json.dumps({k: s[k] for k in ("status", "records", "counts", "foreground_fraction_by_mode",
                                        "flagged", "rejected", "review_pages",
                                        "elapsed_seconds")}, indent=1))
    print("verify:", P.verify_receipt(rp, C.load(a.config)["project_root"]) or "OK")
