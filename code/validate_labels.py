"""Week-0 gate: prove every label in every dataset maps explicitly, and
report the real melanoma counts the power analysis depends on.

Run:  python code/validate_labels.py
"""
import csv, glob, os, sys, collections
sys.path.insert(0, os.path.dirname(__file__))
import labels

ROOT = os.path.join(os.path.dirname(__file__), "..", "DATASETS")


def _read(path, **kw):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, **kw))


def check(name, rows, dxfield, mapper, imgfields, imgroot):
    """Map every row; verify referenced image files exist."""
    counts = collections.Counter()
    bad = []
    missing = 0
    for r in rows:
        try:
            counts[mapper(r[dxfield])] += 1
        except labels.LabelError as e:
            bad.append(str(e))
        for fld in imgfields:
            p = os.path.join(imgroot, r[fld])
            if not os.path.exists(p):
                missing += 1
    n = sum(counts.values())
    mel = counts[labels.MELANOMA]
    print(f"  {name:14s} n={n:<6d} melanoma={mel:<5d} non_melanoma={counts[labels.NON_MELANOMA]:<6d} "
          f"prev={mel/n*100 if n else 0:5.1f}%  missing_files={missing}")
    if bad:
        print(f"    !! {len(bad)} UNMAPPED LABELS:")
        for b in sorted(set(bad))[:10]:
            print("      ", b)
    return mel, n, len(bad), missing


def main():
    print("=" * 78)
    print("LABEL MAPPING VALIDATION  (Week-0 gate)")
    print("=" * 78)
    tot_bad = tot_missing = 0

    print("\n[1] Derm7pt  -- PRIMARY cross-modality evidence (paired, same lesion)")
    d7 = _read(f"{ROOT}/derm7pt/meta/meta.csv")
    mel7, n7, b, m = check("derm7pt", d7, "diagnosis", labels.derm7pt,
                           ["clinic", "derm"], f"{ROOT}/derm7pt/images")
    tot_bad += b; tot_missing += m
    same = [r for r in d7 if r["clinic"] == r["derm"]]
    print("                 Paired lesion identity; clinical is not proven smartphone capture.")
    print(f"                 WARNING: {len(same)} pairs reference the SAME image "
          f"({sum(labels.derm7pt(r['diagnosis']) for r in same)} melanoma).")
    print("                 Retained pending population/exclusion protocol approval.")

    print("\n[2] HAM10000 -- TRAINING set (dermoscopy) + lesion masks")
    ham = _read(f"{ROOT}/ham10000/HAM10000_metadata.tab", delimiter="\t")
    for r in ham:
        r["_img"] = r["image_id"] + ".jpg"
    melh, nh, b, m = check("ham10000", ham, "dx", labels.ham10000,
                           ["_img"], f"{ROOT}/ham10000/images")
    tot_bad += b; tot_missing += m
    nseg = len(glob.glob(f"{ROOT}/ham10000/segmentations/*"))
    print(f"                 lesion masks present: {nseg} (needed for localisation axis)")
    # leakage check: HAM10000 has multiple images per lesion_id
    les = collections.Counter(r["lesion_id"] for r in ham)
    dup = sum(1 for v in les.values() if v > 1)
    print(f"                 unique lesions={len(les)}; lesions with >1 image={dup}")
    print("                 => MUST split by lesion_id, not image_id (leakage risk).")

    print("\n[3] PAD-UFES-20 -- real smartphone images (supporting evidence)")
    pad = _read(f"{ROOT}/pad_ufes_20/metadata.csv")
    melp, np_, b, m = check("pad_ufes_20", pad, "diagnostic", labels.pad_ufes_20,
                            ["img_id"], f"{ROOT}/pad_ufes_20/images")
    tot_bad += b; tot_missing += m
    pats = len(set(r["patient_id"] for r in pad))
    print(f"                 patients={pats} => split by patient_id.")
    print(f"                 !! only {melp} melanomas: supporting, NOT primary.")

    print("\n[4] DDI -- skin-tone fairness (RQ5)")
    ddi = _read(f"{ROOT}/ddidiversedermatologyimages/ddi_metadata.csv")
    meld, nd, b, m = check("ddi", ddi, "disease", labels.ddi,
                           ["DDI_file"], f"{ROOT}/ddidiversedermatologyimages")
    tot_bad += b; tot_missing += m
    st = collections.Counter(r["skin_tone"] for r in ddi)
    print(f"                 skin_tone groups: {dict(sorted(st.items()))}")
    melbytone = collections.Counter(
        r["skin_tone"] for r in ddi if labels.ddi(r["disease"]) == labels.MELANOMA)
    print(f"                 melanoma per tone: {dict(sorted(melbytone.items()))}")
    print("                 !! tiny per-tone melanoma counts -> EXPLORATORY only.")

    print("\n[5] Fitzpatrick17k -- CSV only, images are URLs (not downloaded)")
    fz = _read(f"{ROOT}/fitzpatrick17k/fitzpatrick17k.csv")
    print(f"  {'fitzpatrick17k':14s} n={len(fz)}  local images=0 "
          f"(requires URL scrape; treat as OPTIONAL)")

    print("\n" + "=" * 78)
    print("LEGACY POWER APPROXIMATION -- UNVALIDATED, not a study decision gate")
    print("=" * 78)
    power_table(mel7)

    print("\n" + "=" * 78)
    if tot_bad:
        print(f"FAIL: {tot_bad} unmapped labels. Fix code/labels.py before proceeding.")
        return 1
    if tot_missing:
        print(f"FAIL: {tot_missing} referenced image files missing.")
        return 1
    print("PASS: every label maps explicitly; every referenced image file exists.")
    print("=" * 78)
    return 0


def power_table(n_mel_paired):
    import math

    def power(n, p1, p2, rho=0.5):
        pd = (p1 * (1 - p2) + p2 * (1 - p1)) * (1 - rho)
        if pd <= 0:
            return 0.0
        nd = n * pd
        if nd < 1:
            return 0.0
        pi = (p1 * (1 - p2)) * (1 - rho) / pd
        z = 1.959964
        se0 = math.sqrt(0.25 / nd)
        se1 = math.sqrt(max(pi * (1 - pi), 1e-9) / nd)
        return 0.5 * (1 + math.erf(((abs(pi - .5) - z * se0) / se1) / math.sqrt(2)))

    print(f"Derm7pt melanomas usable as TEST data: {n_mel_paired}")
    print("WARNING: the legacy rho scaling does not preserve the specified sensitivity")
    print("difference. Do not use these numbers as achieved/planning power assurance.")
    print("It also does not match the current all-case correctness McNemar endpoint.\n")
    print(f"  {'sens. drop':>11s} | {'n=101 (stock test split)':>25s} | {'n=252 (all paired)':>20s}")
    print("  " + "-" * 62)
    for d in (0.05, 0.10, 0.15, 0.20):
        a, b = power(101, .95, .95 - d), power(n_mel_paired, .95, .95 - d)
        print(f"  {d*100:8.0f} pp | {a:25.2f} | {b:20.2f}")
    print("\n  Historical values only. Endpoint, pair exclusions and a coherent joint")
    print("  probability model require approval before a replacement power analysis.")


if __name__ == "__main__":
    sys.exit(main())
