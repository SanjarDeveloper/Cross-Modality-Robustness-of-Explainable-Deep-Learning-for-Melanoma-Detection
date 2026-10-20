"""Stage 8A bounded visual review of perturb_msim_v1 on a small deterministic HAM TRAIN sample.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.perturb_review --config ../configs/perturb_review_8a_v1.json --out-root ../runs/perturb_review_8a

Sample: HAM TRAIN only; for each (label, mask PNG mode L/P) the first record in
(stable_seed('perturb_review_v1', record_uid), record_uid) order -> 4 images. No model is used.
"""
import argparse
import io
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    import numpy as np
    from PIL import Image, ImageDraw
    from . import augment as AU, config as C, decode as DC, perturb as PT, preprocess as PP, provenance as P
    from .desc_4b3 import publish_bytes, publish_json
    from .seeds import stable_seed
    cfg = C.load(a.config)
    spec_p = C.resolve(cfg, cfg["supporting_inputs"]["perturb_spec"])
    spec = json.load(open(spec_p, encoding="utf-8"))
    if P.sha256_file(os.path.join(os.path.dirname(__file__), "perturb.py")) != spec["module_sha256"]:
        raise RuntimeError("perturb.py differs from the frozen specification")
    root = cfg["project_root"]
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    train = sorted((r for r in rows if r["split"] == "train" and r["mask_path"]),
                   key=lambda r: (stable_seed("perturb_review_v1", r["record_uid"]), r["record_uid"]))
    want, chosen = {(l, m) for l in ("1", "0") for m in ("L", "P")}, {}
    for r in train:
        les, meta = DC.interpret_mask(os.path.join(root, "DATASETS", *r["mask_path"].split("/")))
        key = (r["label_mel"], meta["original_mode"])
        if key in want and key not in chosen:
            chosen[key] = (r, les, meta)
        if len(chosen) == 4:
            break
    if len(chosen) != 4:
        raise RuntimeError(f"could not fill sample: {sorted(chosen)}")
    reader = PP.EvalReader(rows, "ham_train", os.path.join(root, "cache"))
    idx = {r["record_uid"]: i for i, r in enumerate(reader.rows)}
    receipt = P.build_receipt(cfg, a.config, "completed", details={"stage": "8A visual review"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    sample_meta, params = [], []
    T = 160
    for f in PT.FACTORS:
        tiles = []
        for key in sorted(chosen):
            r, les, mmeta = chosen[key]
            img = np.asarray(reader.img[int(r["cache_index"])]).copy()
            m256 = AU.mask_to_cache_grid(les)
            cells = [(img, m256, "original")]
            for s in (0.0, 0.5, 1.0):
                o, mo, meta = PT.apply(img, f, s, r["image_uid"], mask256=m256)
                cells.append((o, mo, f"{f} s={s}"))
                params.append(meta)
            row = Image.new("RGB", (T * 4, T + 14), "white")
            d = ImageDraw.Draw(row)
            for j, (im, mk, lab) in enumerate(cells):
                tile = Image.fromarray(im).resize((T, T), Image.BILINEAR)
                edge = mk.astype(bool) & ~np.pad(mk.astype(bool), 1)[2:, 1:-1] | mk.astype(bool) & ~np.pad(mk.astype(bool), 1)[:-2, 1:-1] \
                    | mk.astype(bool) & ~np.pad(mk.astype(bool), 1)[1:-1, 2:] | mk.astype(bool) & ~np.pad(mk.astype(bool), 1)[1:-1, :-2]
                ov = np.asarray(tile).copy()
                e = np.asarray(Image.fromarray((edge * 255).astype(np.uint8)).resize((T, T), Image.NEAREST)) > 0
                ov[e] = (0, 255, 0)
                row.paste(Image.fromarray(ov), (j * T, 14))
                d.text((j * T + 2, 1), lab, fill="black")
            d.text((2, T + 1), "", fill="black")
            tiles.append((row, f"{r['record_uid']} y={r['label_mel']} mask={mmeta['original_mode']}"))
        sheet = Image.new("RGB", (T * 4, (T + 26) * len(tiles)), "white")
        dd = ImageDraw.Draw(sheet)
        for i, (row, cap) in enumerate(tiles):
            sheet.paste(row, (0, i * (T + 26)))
            dd.text((2, i * (T + 26) + T + 14), cap, fill="black")
        buf = io.BytesIO()
        sheet.save(buf, format="PNG")
        publish_bytes(out, f"review_{f}.png", buf.getvalue())
    for key in sorted(chosen):
        r, les, mmeta = chosen[key]
        sample_meta.append({"record_uid": r["record_uid"], "image_uid": r["image_uid"], "label_mel": r["label_mel"],
                            "split": r["split"], "mask_mode": mmeta["original_mode"], "mask_flags": mmeta.get("flags")})
    publish_json(out, "review_sample.json", {"selection": "HAM TRAIN; first per (label, mask mode) by stable_seed order",
                                             "sample": sample_meta, "overlay": "green = mask boundary (256 grid)"})
    publish_json(out, "perturbation_params.json", params)
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))}
    rp = P.write_receipt(receipt, out)
    print(out, json.dumps(sample_meta), "\nverify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


if __name__ == "__main__":
    main()
