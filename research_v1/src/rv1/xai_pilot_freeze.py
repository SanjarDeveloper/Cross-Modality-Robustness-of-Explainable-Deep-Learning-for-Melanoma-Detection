"""Stage 3B step 1: freeze the 24-pair XAI pilot sample BEFORE any inference (Addendum D3).

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.xai_pilot_freeze --project-root ../..

Writes (never overwrites) configs/xai_pilot_stage3b_v1_pairs.csv and _refs.csv.
"""
import argparse
import csv
import os

from . import xai_pairs as XP

N_PER_LABEL = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    a = ap.parse_args()
    root = os.path.abspath(a.project_root)
    mdir = os.path.join(root, "research_v1", "manifests", "v1")
    with open(os.path.join(mdir, "xai_derm7pt_v1.csv"), encoding="utf-8", newline="") as f:
        xai = list(csv.DictReader(f))
    with open(os.path.join(mdir, "derm7pt_v1.csv"), encoding="utf-8", newline="") as f:
        derm = list(csv.DictReader(f))
    pairs = XP.select_pilot_pairs(xai, derm, N_PER_LABEL)
    sub = set(XP.batch_subset(pairs, 2))
    cdir = os.path.join(root, "research_v1", "configs")
    with open(os.path.join(cdir, "xai_pilot_stage3b_v1_pairs.csv"), "x", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pilot_order", "pair_id", "label_mel", "label_rank", "rank_key", "batch_check_subset",
                    "record_uid_dermoscopy", "record_uid_clinical"])
        for i, p in enumerate(pairs):
            w.writerow([i, p["pair_id"], p["label_mel"], p["label_rank"], p["rank_key"], int(p["pair_id"] in sub),
                        p["record_uid_dermoscopy"], p["record_uid_clinical"]])
    by = {r["record_uid"]: r for r in derm}
    with open(os.path.join(cdir, "xai_pilot_stage3b_v1_refs.csv"), "x", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["ref_order", "ref_key", "pair_id", "label_mel", "modality", "record_uid", "image_uid",
                    "cache_name", "cache_index", "sha256"])
        k = 0
        for p in pairs:
            for mod in ("dermoscopy", "clinical"):
                u = p[f"record_uid_{mod}"]
                d = by[u]
                w.writerow([k, f"{u}|{mod}", p["pair_id"], p["label_mel"], mod, u, d["image_uid"],
                            d["cache_name"], d["cache_index"], d["sha256"]])
                k += 1
    print(len(pairs), "pairs;", k, "refs; batch subset:", sorted(sub))


if __name__ == "__main__":
    main()
