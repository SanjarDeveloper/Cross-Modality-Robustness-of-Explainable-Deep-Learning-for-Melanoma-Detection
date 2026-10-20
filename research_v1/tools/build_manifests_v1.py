"""Stage 2A / T-A2: build and validate research_v1 cohort manifests (v1).

Read-only with respect to all legacy paths. No image is decoded; image files are
only streamed for SHA-256. Cache .npy arrays are never opened; only
cache/*_meta.json is read for the order check.

Inputs: DATASETS metadata; code/labels.py (label maps); code/data.py (legacy
split + record order; its module level only defines constants and imports
torch/cv2, no I/O). Run with PYTHONDONTWRITEBYTECODE=1 and python -B.

  python -B research_v1/tools/build_manifests_v1.py --out research_v1/manifests

Exit code 0 = all validations PASS; 1 = BLOCKED (diagnostics still written,
card status "BLOCKED").

Stable hash (never Python hash()):
  key(s) = hashlib.sha256(s.encode("utf-8")).hexdigest()   (lower-case hex)
  PAD role order : sort by (key(f"0|{patient_id}"), patient_id)
  XAI selection  : per class, sort by (key(f"20260927|{pair_id}"), pair_id), take first 249
"""
import argparse, csv, hashlib, io, json, math, os, sys, time
from collections import Counter, defaultdict

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DS = os.path.join(PROJECT, "DATASETS")
CACHE = os.path.join(PROJECT, "cache")
AUDIT = os.path.join(PROJECT, "audit", "manifests")
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(PROJECT, "code"))
import labels as L   # pure dictionaries + lookup functions
import data as D     # legacy loaders: metadata only (no image I/O in the functions used)

SPLIT_SEED = 0
XAI_SEED = 20260927
XAI_PER_CLASS = 249
INFO = []
STRATUM_RULE = "pad_rq5_v1: S1 any MEL > S2 any BCC/SCC/ACK > S3; roles eval .50 / adapt_train .35 / adapt_val rest, ceil"
COLUMNS = ["record_uid", "dataset", "image_uid", "rel_path", "sha256", "label_mel", "dx_raw",
           "modality", "cluster_id", "pair_id", "same_file_pair", "in_derm7pt_all",
           "in_derm7pt_primary", "cohort_flags", "split", "split_seed", "rq5_role",
           "stratum_rq5", "stratum_rule_version", "mask_path", "skin_tone",
           "cache_name", "cache_index", "legacy_row_index"]


def key(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, DS).replace("\\", "/")


def read_csv(path, **kw):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, **kw))


class Checks:
    def __init__(self):
        self.items = []

    def add(self, group, name, ok, observed, expected=None, source=None):
        self.items.append({"group": group, "check": name, "status": "PASS" if ok else "FAIL",
                           "observed": observed, "expected": expected, "source": source})

    @property
    def ok(self):
        return all(i["status"] == "PASS" for i in self.items)


def row(**kw):
    r = {c: "" for c in COLUMNS}
    r.update({k: ("" if v is None else v) for k, v in kw.items()})
    return r


def build(chk, hashes):
    t = {}
    # ------------------------------------------------------------ HAM10000
    ham_rows, ham_split = [], {}
    split_recs = {s: D.ham10000(s, seed=SPLIT_SEED) for s in ("train", "val", "test")}
    for s, recs in split_recs.items():
        for i, r in enumerate(recs):
            ham_split[r.meta["image_id"]] = (s, i)
    all_ham = D.ham10000()
    for li, r in enumerate(all_ham):
        iid = r.meta["image_id"]
        s, ci = ham_split[iid]
        uid = f"ham10000:{iid}"
        ham_rows.append(row(record_uid=uid, dataset="ham10000", image_uid=uid, rel_path=rel(r.img),
                            sha256=hashes(r.img), label_mel=r.label, dx_raw=r.meta["dx"],
                            modality="dermoscopy", cluster_id=r.group, split=s,
                            split_seed=SPLIT_SEED, mask_path=rel(r.mask) if r.mask else "",
                            cache_name=f"ham_{s}", cache_index=ci, legacy_row_index=li))
    t["ham10000"] = ham_rows

    # independent re-implementation of the legacy split rule (validation 5)
    def bucket(g):
        h = (int(hashlib.md5(str(g).encode()).hexdigest()[:8], 16)
             ^ int(hashlib.md5(str(SPLIT_SEED).encode()).hexdigest()[:8], 16)) % 10000 / 10000.0
        return "test" if h < .15 else "val" if h < .30 else "train"
    mism = sum(1 for r in ham_rows if bucket(r["cluster_id"]) != r["split"])
    chk.add(5, "ham_split_equals_independent_reimplementation_of_legacy_rule", mism == 0, mism, 0,
            "code/data.py:ham10000 (md5 xor bucket)")

    # ------------------------------------------------------------ Derm7pt
    clin, derm = D.derm7pt()
    d_rows = []
    for i, (c, d) in enumerate(zip(clin, derm)):
        same = os.path.normcase(os.path.abspath(c.img)) == os.path.normcase(os.path.abspath(d.img))
        pid = c.group
        flags = "derm7pt_all" if same else "derm7pt_all;derm7pt_primary"
        for r, cache in ((c, "derm7pt_clinical"), (d, "derm7pt_dermoscopy")):
            d_rows.append(row(record_uid=f"derm7pt:{pid}:{r.modality}", dataset="derm7pt",
                              image_uid=f"derm7pt:{rel(r.img)}", rel_path=rel(r.img),
                              sha256=hashes(r.img), label_mel=r.label, dx_raw=r.meta["dx"],
                              modality=r.modality, cluster_id=pid, pair_id=pid,
                              same_file_pair=int(same), in_derm7pt_all=1,
                              in_derm7pt_primary=int(not same), cohort_flags=flags,
                              cache_name=cache, cache_index=i, legacy_row_index=i))
    t["derm7pt"] = d_rows

    # ------------------------------------------------------------ PAD
    pad = D.pad_ufes_20()
    pat_dx = defaultdict(set)
    for r in pad:
        pat_dx[r.group].add(r.meta["dx"].strip().upper())

    def stratum(dxs):
        return "S1" if "MEL" in dxs else "S2" if dxs & {"BCC", "SCC", "ACK"} else "S3"
    role = {}
    for s in ("S1", "S2", "S3"):
        pats = sorted((p for p, dx in pat_dx.items() if stratum(dx) == s),
                      key=lambda p: (key(f"{SPLIT_SEED}|{p}"), p))
        n = len(pats)
        n_eval = math.ceil(0.50 * n)
        n_tr = math.ceil(0.35 * n)
        for j, p in enumerate(pats):
            role[p] = "eval" if j < n_eval else "adapt_train" if j < n_eval + n_tr else "adapt_val"
    p_rows = []
    for i, r in enumerate(pad):
        name = os.path.basename(r.img)
        p_rows.append(row(record_uid=f"pad_ufes_20:{name}", dataset="pad_ufes_20",
                          image_uid=f"pad_ufes_20:{name}", rel_path=rel(r.img), sha256=hashes(r.img),
                          label_mel=r.label, dx_raw=r.meta["dx"], modality="smartphone",
                          cluster_id=r.group, rq5_role=role[r.group],
                          stratum_rq5=stratum(pat_dx[r.group]), stratum_rule_version=STRATUM_RULE,
                          skin_tone=str(r.meta.get("fitzpatrick", "")).strip(),
                          cache_name="pad_ufes_20", cache_index=i, legacy_row_index=i))
    t["pad_ufes_20"] = p_rows
    t["_pad_patient_dx"] = pat_dx

    # ------------------------------------------------------------ DDI
    ddi = D.ddi()
    t["ddi"] = [row(record_uid=f"ddi:{os.path.basename(r.img)}", dataset="ddi",
                    image_uid=f"ddi:{os.path.basename(r.img)}", rel_path=rel(r.img),
                    sha256=hashes(r.img), label_mel=r.label, dx_raw=r.meta["disease"],
                    modality="clinical", cluster_id=r.group, skin_tone=r.meta["skin_tone"],
                    cache_name="ddi", cache_index=i, legacy_row_index=i)
                for i, r in enumerate(ddi)]

    # ------------------------------------------------------------ XAI pairs
    pairs = {}
    for r in d_rows:
        if r["in_derm7pt_primary"] == 1:
            pairs.setdefault(r["pair_id"], {})[r["modality"]] = r
    xai = []
    for lab in (1, 0):
        ids = sorted((p for p, v in pairs.items() if v["clinical"]["label_mel"] == lab),
                     key=lambda p: (key(f"{XAI_SEED}|{p}"), p))[:XAI_PER_CLASS]
        for p in ids:
            v = pairs[p]
            xai.append({"pair_id": p, "label_mel": lab,
                        "record_uid_dermoscopy": v["dermoscopy"]["record_uid"],
                        "record_uid_clinical": v["clinical"]["record_uid"],
                        "image_uid_dermoscopy": v["dermoscopy"]["image_uid"],
                        "image_uid_clinical": v["clinical"]["image_uid"],
                        "selection_rank_key": key(f"{XAI_SEED}|{p}"),
                        "sample_seed": XAI_SEED})
    xai.sort(key=lambda r: (-r["label_mel"], r["selection_rank_key"], r["pair_id"]))
    t["xai"] = xai
    return t


def validate(t, chk):
    ham, dm, pad, ddi, xai = t["ham10000"], t["derm7pt"], t["pad_ufes_20"], t["ddi"], t["xai"]
    # 1. files exist (hashing already required a readable file)
    for name, rows in (("ham10000", ham), ("derm7pt", dm), ("pad_ufes_20", pad), ("ddi", ddi)):
        missing = [r["rel_path"] for r in rows if not os.path.isfile(os.path.join(DS, r["rel_path"]))]
        chk.add(1, f"{name}_image_files_exist", not missing, len(missing), 0)
    mm = [r["mask_path"] for r in ham if r["mask_path"] and not os.path.isfile(os.path.join(DS, r["mask_path"]))]
    chk.add(1, "ham10000_mask_files_exist", not mm, len(mm), 0)
    chk.add(1, "ham10000_rows_with_mask", True, sum(1 for r in ham if r["mask_path"]), None)
    # 2. labels: every row mapped (LabelError would have aborted earlier)
    for name, rows in (("ham10000", ham), ("derm7pt", dm), ("pad_ufes_20", pad), ("ddi", ddi)):
        bad = [r for r in rows if r["label_mel"] not in (0, 1)]
        chk.add(2, f"{name}_all_labels_mapped", not bad, len(bad), 0, "code/labels.py (raises on unknown)")
    # 3. counts
    def c(group, name, obs, exp, src):
        chk.add(3, name, obs == exp, obs, exp, src)
    S = "raw metadata via code/data.py loaders"
    c(3, "ham_images", len(ham), 10015, S)
    c(3, "ham_melanoma", sum(r["label_mel"] for r in ham), 1113, S)
    for s, n, m in (("train", 6938, 783), ("val", 1529, 160), ("test", 1548, 170)):
        c(3, f"ham_{s}_images", sum(r["split"] == s for r in ham), n, S)
        c(3, f"ham_{s}_melanoma", sum(r["split"] == s and r["label_mel"] for r in ham), m, S)
    pairs = {r["pair_id"]: r for r in dm}
    c(3, "derm7pt_rows", len(dm), 2022, S)
    c(3, "derm7pt_pairs", len(pairs), 1011, S)
    c(3, "derm7pt_melanoma_pairs", sum(r["label_mel"] for r in pairs.values()), 252, S)
    same = {r["pair_id"]: r for r in dm if r["same_file_pair"] == 1}
    c(3, "derm7pt_same_file_pairs", len(same), 9, S)
    c(3, "derm7pt_same_file_melanoma", sum(r["label_mel"] for r in same.values()), 3, S)
    prim = {r["pair_id"]: r for r in dm if r["in_derm7pt_primary"] == 1}
    c(3, "derm7pt_primary_pairs", len(prim), 1002, S)
    c(3, "derm7pt_primary_melanoma", sum(r["label_mel"] for r in prim.values()), 249, S)
    c(3, "derm7pt_unique_files", len({r["image_uid"] for r in dm}), 2013, S)
    c(3, "pad_images", len(pad), 2298, S)
    c(3, "pad_melanoma_images", sum(r["label_mel"] for r in pad), 52, S)
    c(3, "pad_patients", len({r["cluster_id"] for r in pad}), 1373, S)
    c(3, "pad_melanoma_patients", len({r["cluster_id"] for r in pad if r["label_mel"]}), 36, S)
    c(3, "pad_multi_diagnosis_patients",
      sum(1 for v in t["_pad_patient_dx"].values() if len(v) > 1), 179, S)
    c(3, "ddi_images", len(ddi), 656, S)
    c(3, "ddi_melanoma", sum(r["label_mel"] for r in ddi), 21, S)
    for tone in ("12", "34", "56"):
        c(3, f"ddi_melanoma_tone_{tone}",
          sum(1 for r in ddi if r["skin_tone"] == tone and r["label_mel"]), 7, S)
    # 4. disjointness
    ls = defaultdict(set)
    for r in ham:
        ls[r["cluster_id"]].add(r["split"])
    chk.add(4, "ham_lesion_in_one_split", all(len(v) == 1 for v in ls.values()),
            sum(len(v) > 1 for v in ls.values()), 0)
    pr = defaultdict(set)
    for r in pad:
        pr[r["cluster_id"]].add(r["rq5_role"])
    chk.add(4, "pad_patient_in_one_role", all(len(v) == 1 for v in pr.values()),
            sum(len(v) > 1 for v in pr.values()), 0)
    by_pair = defaultdict(list)
    for r in dm:
        by_pair[r["pair_id"]].append(r)
    bad = [p for p, v in by_pair.items()
           if len(v) != 2 or {x["modality"] for x in v} != {"clinical", "dermoscopy"}
           or len({x["label_mel"] for x in v}) != 1]
    chk.add(4, "derm7pt_each_pair_two_rows_same_label", not bad, len(bad), 0)
    for name, rows in (("ham10000", ham), ("derm7pt", dm), ("pad_ufes_20", pad), ("ddi", ddi)):
        dup = len(rows) - len({r["record_uid"] for r in rows})
        chk.add(4, f"{name}_record_uid_unique", dup == 0, dup, 0)
    # 5. legacy consistency: cache meta order/labels/groups
    for name, rows in (("ham_train", [r for r in ham if r["split"] == "train"]),
                       ("ham_val", [r for r in ham if r["split"] == "val"]),
                       ("ham_test", [r for r in ham if r["split"] == "test"]),
                       ("derm7pt_clinical", [r for r in dm if r["modality"] == "clinical"]),
                       ("derm7pt_dermoscopy", [r for r in dm if r["modality"] == "dermoscopy"]),
                       ("pad_ufes_20", pad), ("ddi", ddi)):
        meta = json.load(open(os.path.join(CACHE, f"{name}_meta.json")))
        rows = sorted(rows, key=lambda r: r["cache_index"])
        ok = (meta["n"] == len(rows)
              and [int(x) for x in meta["labels"]] == [r["label_mel"] for r in rows]
              and [str(g) for g in meta["groups"]] == [str(r["cluster_id"]) for r in rows]
              and [r["cache_index"] for r in rows] == list(range(len(rows))))
        if ok and name.startswith("ham"):
            ok = [m["image_id"] for m in meta["meta"]] == [r["image_uid"].split(":", 1)[1] for r in rows]
        chk.add(5, f"cache_meta_order_labels_groups_{name}", ok, meta["n"], len(rows),
                f"cache/{name}_meta.json")
    # legacy_row_index equals loader order
    for name, rows, loader_len in (("ham10000", ham, None), ("pad_ufes_20", pad, None), ("ddi", ddi, None)):
        chk.add(5, f"{name}_legacy_row_index_sequential",
                [r["legacy_row_index"] for r in rows] == list(range(len(rows))), len(rows), len(rows))
    # audit manifests cross-check (file set + labels)
    for name, rows, id_of in (("ham10000", ham, lambda r: r["rel_path"]),
                              ("derm7pt", dm, lambda r: (r["rel_path"], r["modality"], r["pair_id"])),
                              ("pad_ufes_20", pad, lambda r: r["rel_path"]),
                              ("ddi", ddi, lambda r: r["rel_path"])):
        a = read_csv(os.path.join(AUDIT, f"{name}_manifest_v1.csv"))
        if name == "derm7pt":
            # Spec: file sets + labels. Pair identifiers are NOT compared here: the audit
            # used 'case<case_num>' for every case, whereas the legacy loader (and the
            # legacy caches) use the non-empty meta.csv case_id for 27 cases. That
            # convention difference is reported separately in card["info"].
            aset = Counter((x["relative_path"], x["modality"], int(x["label"])) for x in a)
            mset = Counter((r["rel_path"], r["modality"], r["label_mel"]) for r in rows)
            conv = Counter(x["lesion_id"] for x in a) != Counter(r["pair_id"] for r in rows)
            INFO.append({"check": "derm7pt_pair_id_convention_vs_audit_manifest",
                         "differs": conv,
                         "explanation": "audit lesion_id = 'case<case_num>'; v1 pair_id follows "
                                        "code/data.py:derm7pt (case_id if non-empty, else "
                                        "'case<case_num>'), which matches cache/derm7pt_*_meta.json groups"})
        else:
            aset = Counter((x["relative_path"], int(x["label"])) for x in a)
            mset = Counter((r["rel_path"], r["label_mel"]) for r in rows)
        sha_a = {x["relative_path"]: x["sha256"] for x in a}
        sha_diff = sum(1 for r in rows if sha_a.get(r["rel_path"]) not in (None, r["sha256"]))
        chk.add(5, f"{name}_matches_audit_manifest_paths_labels", aset == mset,
                sum((aset - mset).values()) + sum((mset - aset).values()), 0,
                f"audit/manifests/{name}_manifest_v1.csv (cross-check only)")
        chk.add(5, f"{name}_sha256_matches_audit_manifest", sha_diff == 0, sha_diff, 0,
                f"audit/manifests/{name}_manifest_v1.csv (cross-check only)")
    # 6. XAI
    prim = {r["pair_id"] for r in dm if r["in_derm7pt_primary"] == 1}
    chk.add(6, "xai_rows", len(xai) == 2 * XAI_PER_CLASS, len(xai), 2 * XAI_PER_CLASS)
    chk.add(6, "xai_melanoma_pairs", sum(r["label_mel"] for r in xai) == XAI_PER_CLASS,
            sum(r["label_mel"] for r in xai), XAI_PER_CLASS)
    chk.add(6, "xai_all_in_primary", all(r["pair_id"] in prim for r in xai),
            sum(r["pair_id"] not in prim for r in xai), 0)
    chk.add(6, "xai_unique_pairs", len({r["pair_id"] for r in xai}) == len(xai),
            len({r["pair_id"] for r in xai}), len(xai))
    recs = {r["record_uid"] for r in dm}
    chk.add(6, "xai_record_uids_resolve", all(r["record_uid_dermoscopy"] in recs and
                                              r["record_uid_clinical"] in recs for r in xai), True, True)
    # 7. RQ5
    for role in ("eval", "adapt_train", "adapt_val"):
        n = len({r["cluster_id"] for r in pad if r["rq5_role"] == role and r["stratum_rq5"] == "S1"})
        chk.add(7, f"rq5_{role}_has_S1_patient", n >= 1, n, ">=1")
    pat_role = {r["cluster_id"]: r["rq5_role"] for r in pad}
    pat_str = {r["cluster_id"]: r["stratum_rq5"] for r in pad}
    ok = True
    for s in ("S1", "S2", "S3"):
        n = sum(1 for p in pat_str if pat_str[p] == s)
        exp = (math.ceil(.5 * n), math.ceil(.35 * n), n - math.ceil(.5 * n) - math.ceil(.35 * n))
        obs = tuple(sum(1 for p in pat_str if pat_str[p] == s and pat_role[p] == ro)
                    for ro in ("eval", "adapt_train", "adapt_val"))
        chk.add(7, f"rq5_{s}_role_counts_follow_rule", obs == exp, list(obs), list(exp))


def summary(t):
    ham, dm, pad, ddi = t["ham10000"], t["derm7pt"], t["pad_ufes_20"], t["ddi"]
    out = {"ham10000": {}, "derm7pt": {}, "pad_ufes_20": {}, "ddi": {}}
    for s in ("train", "val", "test"):
        rr = [r for r in ham if r["split"] == s]
        out["ham10000"][s] = {"images": len(rr), "melanoma_images": sum(r["label_mel"] for r in rr),
                              "lesions": len({r["cluster_id"] for r in rr}),
                              "melanoma_lesions": len({r["cluster_id"] for r in rr if r["label_mel"]})}
    for coh in ("in_derm7pt_all", "in_derm7pt_primary"):
        pr = {r["pair_id"]: r for r in dm if r[coh] == 1}
        out["derm7pt"][coh] = {"pairs": len(pr), "image_reference_rows": 2 * len(pr),
                               "melanoma_pairs": sum(r["label_mel"] for r in pr.values())}
    out["derm7pt"]["unique_files"] = len({r["image_uid"] for r in dm})
    tab = {}
    for role in ("eval", "adapt_train", "adapt_val"):
        for s in ("S1", "S2", "S3"):
            rr = [r for r in pad if r["rq5_role"] == role and r["stratum_rq5"] == s]
            tab[f"{role}/{s}"] = {"patients": len({r["cluster_id"] for r in rr}), "images": len(rr),
                                  "melanoma_images": sum(r["label_mel"] for r in rr),
                                  "melanoma_patients": len({r["cluster_id"] for r in rr if r["label_mel"]})}
        rr = [r for r in pad if r["rq5_role"] == role]
        tab[f"{role}/ALL"] = {"patients": len({r["cluster_id"] for r in rr}), "images": len(rr),
                              "melanoma_images": sum(r["label_mel"] for r in rr),
                              "melanoma_patients": len({r["cluster_id"] for r in rr if r["label_mel"]}),
                              "melanoma_images_by_fitzpatrick": dict(sorted(Counter(
                                  r["skin_tone"] or "missing" for r in rr if r["label_mel"]).items()))}
    out["pad_ufes_20"]["rq5_partition"] = tab
    out["pad_ufes_20"]["missing_skin_tone_images"] = sum(1 for r in pad if not r["skin_tone"])
    out["ddi"] = {tone: {"images": sum(r["skin_tone"] == tone for r in ddi),
                         "melanoma": sum(r["skin_tone"] == tone and r["label_mel"] for r in ddi)}
                  for tone in ("12", "34", "56")}
    return out


def write_csv(path, rows, cols):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    with open(path, "xb") as f:
        f.write(data)
    with open(path + ".sha256", "x", encoding="utf-8") as f:
        f.write(f"{hashlib.sha256(data).hexdigest()}  {os.path.basename(path)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    if os.listdir(out):
        raise SystemExit(f"output directory not empty: {out}")
    t0 = time.time()
    cache = {}

    def hashes(p):
        if p not in cache:
            cache[p] = file_sha(p)
        return cache[p]
    chk = Checks()
    t = build(chk, hashes)
    validate(t, chk)
    for name in ("ham10000", "derm7pt", "pad_ufes_20", "ddi"):
        write_csv(os.path.join(out, f"{name}_v1.csv"), t[name], COLUMNS)
    write_csv(os.path.join(out, "xai_derm7pt_v1.csv"), t["xai"], list(t["xai"][0].keys()))
    card = {"status": "PASS" if chk.ok else "BLOCKED",
            "note": "PASS means all Stage 2A validations passed; it is not an approval for training.",
            "split_seed": SPLIT_SEED, "xai_sample_seed": XAI_SEED, "stratum_rule_version": STRATUM_RULE,
            "stable_hash": "sha256(utf-8 string).hexdigest(); PAD key f'{split_seed}|{patient_id}', "
                           "XAI key f'{sample_seed}|{pair_id}'; ties broken by the id string; "
                           "Python hash() not used",
            "identity": {"record_uid": "unique per image-reference row",
                         "image_uid": "dataset + relative path (path-based file identity); content hash in sha256",
                         "pair_id": "Derm7pt case_id or 'case<case_num>' as in the legacy loader"},
            "counts": summary(t),
            "validation": chk.items,
            "info": INFO,
            "n_checks": len(chk.items), "n_failed": sum(i["status"] == "FAIL" for i in chk.items),
            "elapsed_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out, "manifest_card_v1.json"), "x", encoding="utf-8") as f:
        json.dump(card, f, indent=1, default=str)
    print(json.dumps({k: card[k] for k in ("status", "n_checks", "n_failed", "elapsed_seconds")}))
    for i in chk.items:
        if i["status"] == "FAIL":
            print("FAIL", i)
    sys.exit(0 if chk.ok else 1)


if __name__ == "__main__":
    main()
