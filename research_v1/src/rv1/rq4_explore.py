"""Stage 7D: exploratory E4.4 (melanoma vs non-melanoma shift contrasts) and E4.5 (Grad-CAM vs IG map
agreement), plus a donor-mapping audit, from verified Stage 7A/7B artifacts (CPU only).

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq4_explore --config ../configs/rq4_explore_7d_v1.json --phase spec --out-root ../runs/rq4_explore_7d
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.rq4_explore --config ../configs/rq4_explore_7d_v1.json --phase analyze --run-dir <dir>
"""
import argparse
import datetime
import json
import math
import os

import numpy as np

B = 2000
TOP_FRAC = 0.10
MODS = ("dermoscopy", "clinical")


# ------------------------------------------------------------------ pure helpers (tested)
def topk_indices(m, frac=TOP_FRAC):
    """Exactly k = ceil(frac*N) pixels by descending value; ties broken by ascending flat index."""
    flat = np.asarray(m, dtype=np.float64).reshape(-1)
    k = math.ceil(frac * flat.size)
    order = np.lexsort((np.arange(flat.size), -flat))      # primary: -value, secondary: index
    sel = order[:k]
    tie_at_cutoff = bool(k < flat.size and flat[order[k - 1]] == flat[order[k]])
    return sel, k, tie_at_cutoff


def topk_overlap(a, b, frac=TOP_FRAC):
    """|top_k(a) ∩ top_k(b)| / k (NOT Jaccard)."""
    sa, k, ta = topk_indices(a, frac)
    sb, _, tb = topk_indices(b, frac)
    return len(np.intersect1d(sa, sb)) / k, k, ta, tb


def spearman(a, b):
    """Pearson correlation of average ranks (ties -> average ranks). None if either map is constant."""
    from scipy.stats import rankdata
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return None
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def stratified_bootstrap(groups, stat, B_, seed):
    """groups: {label: 1-D array}. Each replicate resamples within each group separately (sizes fixed).
    stat(array) -> float. Returns per-group CIs and the CI of stat(g1) - stat(g0) (first minus second key)."""
    keys = list(groups)
    rng = np.random.Generator(np.random.PCG64(seed))
    reps = {k: [] for k in keys}
    for _ in range(B_):
        for k in keys:
            v = np.asarray(groups[k], float)
            reps[k].append(stat(v[rng.integers(0, len(v), size=len(v))]))
    reps = {k: np.asarray(v) for k, v in reps.items()}
    diff = reps[keys[0]] - reps[keys[1]]
    q = lambda a: np.quantile(a, [0.025, 0.975], method="linear").tolist()
    return {"group_ci95": {k: q(reps[k]) for k in keys}, "diff_ci95": q(diff), "diff_label": f"{keys[0]} - {keys[1]}",
            "replicates": {**{k: reps[k] for k in keys}, "diff": diff}}


def donor_audit(assignment):
    """Per (label, modality, explainer, condition) group: donor multiplicity and derangement check."""
    groups = {}
    for v in assignment.values():
        groups.setdefault(tuple(v["group"]), []).append(v)
    out = {}
    for g, vs in groups.items():
        recips = [v["record_uid"] for v in vs]
        donors = [v["donor"] for v in vs if v["donor"] is not None]
        mult = {}
        for d in donors:
            mult[d] = mult.get(d, 0) + 1
        out["|".join(g)] = {"n_recipients": len(recips), "n_assigned": len(donors),
                            "max_donor_multiplicity": max(mult.values()) if mult else 0,
                            "donors_equal_recipient_set": sorted(set(donors)) == sorted(recips),
                            "one_to_one_derangement": len(donors) == len(recips) and max(mult.values()) == 1 and
                            sorted(donors) == sorted(recips) and all(v["donor"] != v["record_uid"] for v in vs),
                            "shift": sorted({v["shift"] for v in vs})}
    return out


# ------------------------------------------------------------------ phases
def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _dirs(cfg):
    from . import config as C
    si = cfg["supporting_inputs"]
    return {k: C.resolve(cfg, si[f"{k}_receipt"]).rsplit(os.sep, 1)[0] for k in ("stage7a", "stage7b", "stage7c")}


def _receipt_name(k):
    return {"stage7a": "receipt_audit.json", "stage7b": "receipt_analyze.json", "stage7c": "receipt_analyze.json"}[k]


def phase_spec(cfg, a):
    from . import provenance as P
    from .desc_4b3 import publish_bytes, publish_json
    dirs = _dirs(cfg)
    outs = {}
    for k, d in dirs.items():
        rec = json.load(open(os.path.join(d, _receipt_name(k)), encoding="utf-8"))
        for n, h in rec["outputs"].items():
            if P.sha256_file(os.path.join(d, n)) != h:
                raise RuntimeError(f"changed {k}/{n}")
        outs[k] = {"run": os.path.basename(d), "outputs": rec["outputs"]}
    s = cfg["seeds"]["bootstrap_seed"]
    spec = {"spec_version": "rq4_explore_7d_v1", "created_utc": _now(), "inputs": outs,
            "E4_4": {"source": "02_rev2 §6.7 E4.4 (exploratory; not a formal interaction test)",
                     "population": "Stage 7B primary matched pairs (Grad-CAM 275, IG 373)",
                     "D": "random-referenced contrast_clinical - contrast_dermoscopy (insertion and deletion)",
                     "estimates": "per class: sample median and HL; difference melanoma - non-melanoma for each",
                     "bootstrap": "RETROSPECTIVE: class-stratified case bootstrap (resample cases within each class, "
                                  "both modalities kept via the per-pair D, multiplicity kept), B=2000, percentile 95%",
                     "seeds": {"gradcam": s, "ig512": s + 1}},
            "E4_5": {"source": "02_rev2 §6.7 E4.5 (Spearman correlation, top-10% overlap)",
                     "population": "pairs with BOTH explainers primary-eligible in BOTH modalities (intersection of the "
                                   "Stage 7B primary populations)",
                     "maps": "stored primary positive-evidence maps (Grad-CAM upsampled layer4 ReLU map; IG ReLU of channel "
                             "sum), 224x224, unaltered",
                     "spearman": "Pearson correlation of average ranks (scipy.stats.rankdata, method 'average')",
                     "top_overlap": "RETROSPECTIVE (no prior exact definition): k = ceil(0.10*H*W) = 5018; exactly k pixels "
                                    "by descending value, ties by ascending flat index; overlap = |intersection|/k; NOT Jaccard",
                     "unavailable": "constant / non-finite maps -> unavailable (none expected in this population)",
                     "summaries": "per modality, within-case clinical - dermoscopy difference; median (and HL) with case "
                                  "bootstrap 95% CI, B=2000; class-specific descriptive medians",
                     "seed": s + 2,
                     "limits": "pixels are not sample units; no pixel-level tests; agreement is not correctness or faithfulness"},
            "donor_audit": "per class x modality x explainer x condition group: donor multiplicity and derangement check of the "
                           "frozen Stage 7A mismatch_assignment.json"}
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "analysis_spec.json", spec)
    h = P.sha256_file(os.path.join(out, "analysis_spec.json"))
    publish_bytes(out, "analysis_spec.json.sha256", f"{h}  analysis_spec.json\n".encode())
    receipt["outputs"] = {"analysis_spec.json": h}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("run dir", out)


def phase_analyze(cfg, a):
    import io
    from . import provenance as P, rq4_stats as R4
    from .desc_4b3 import publish_bytes, publish_csv, publish_json
    from .xai_prod import _read_csv_rows, _read_json
    out = os.path.abspath(a.run_dir)
    sp = os.path.join(out, "analysis_spec.json")
    if open(sp + ".sha256", encoding="utf-8").read().split()[0] != P.sha256_file(sp):
        raise RuntimeError("spec hash mismatch")
    spec = json.load(open(sp, encoding="utf-8"))
    dirs = _dirs(cfg)
    for k, d in dirs.items():
        for n, h in spec["inputs"][k]["outputs"].items():
            if P.sha256_file(os.path.join(d, n)) != h:
                raise RuntimeError(f"changed {k}/{n}")
    res, reps = {"spec_sha256": P.sha256_file(sp)}, {}
    # ---------------- donor audit
    assign = _read_json(os.path.join(dirs["stage7a"], "mismatch_assignment.json"))["assignment"]
    audit = donor_audit(assign)
    res["donor_audit"] = {"groups": audit, "all_one_to_one_derangements": all(v["one_to_one_derangement"] for v in audit.values())}
    # ---------------- E4.4
    pp = [r for r in _read_csv_rows(os.path.join(dirs["stage7b"], "per_pair_inputs.csv")) if r["population"] == "primary"]
    status = _read_csv_rows(os.path.join(dirs["stage7b"], "per_image_inputs.csv"))
    seeds = spec["E4_4"]["seeds"]
    e44_rows, e44 = [], {}
    for e in ("gradcam", "ig512"):
        rows = [r for r in pp if r["explainer"] == e]
        if len(rows) != {"gradcam": 275, "ig512": 373}[e] or len({r["pair_id"] for r in rows}) != len(rows):
            raise RuntimeError(f"{e} population")
        st = [r for r in status if r["explainer"] == e]
        elig = {lab: {"eligible_pairs": sum(r["label_mel"] == lab for r in rows),
                      "excluded_pairs": len({r["pair_id"] for r in st if r["label_mel"] == lab}) - sum(r["label_mel"] == lab for r in rows)}
                for lab in ("1", "0")}
        e44[e] = {"eligibility_by_class": elig}
        for c in ("ins", "del"):
            D = {lab: np.array([float(r[f"{c}_clinical"]) - float(r[f"{c}_dermoscopy"]) for r in rows if r["label_mel"] == lab])
                 for lab in ("1", "0")}
            blk = {}
            for nm, fn in (("median", lambda v: float(np.median(v))), ("hl", R4.hodges_lehmann)):
                bs = stratified_bootstrap({"melanoma": D["1"], "non_melanoma": D["0"]}, fn, B, seeds[e] + (0 if c == "ins" else 10))
                reps.update({f"e44|{e}|{c}|{nm}|{k}": v for k, v in bs.pop("replicates").items()})
                blk[nm] = {"melanoma": fn(D["1"]), "non_melanoma": fn(D["0"]), "difference": fn(D["1"]) - fn(D["0"]), **bs}
            e44[e][c] = {"n": {"melanoma": len(D["1"]), "non_melanoma": len(D["0"])}, **blk}
            for lab in ("1", "0"):
                for nm in ("median", "hl"):
                    e44_rows.append({"explainer": e, "curve": c, "class": "melanoma" if lab == "1" else "non_melanoma",
                                     "n": len(D[lab]), "estimator": nm, "estimate": blk[nm]["melanoma" if lab == "1" else "non_melanoma"],
                                     "ci_lo": blk[nm]["group_ci95"]["melanoma" if lab == "1" else "non_melanoma"][0],
                                     "ci_hi": blk[nm]["group_ci95"]["melanoma" if lab == "1" else "non_melanoma"][1],
                                     "mel_minus_nonmel": blk[nm]["difference"], "diff_ci_lo": blk[nm]["diff_ci95"][0],
                                     "diff_ci_hi": blk[nm]["diff_ci95"][1]})
    res["E4.4"] = e44
    # ---------------- E4.5
    gpairs = {r["pair_id"] for r in pp if r["explainer"] == "gradcam"}
    ipairs = {r["pair_id"] for r in pp if r["explainer"] == "ig512"}
    common = sorted(gpairs & ipairs)
    lab = {r["pair_id"]: r["label_mel"] for r in pp}
    maps = {}
    for d in sorted(os.listdir(dirs["stage7a"])):
        if d.startswith("chunk_") and os.path.isdir(os.path.join(dirs["stage7a"], d)):
            z = np.load(os.path.join(dirs["stage7a"], d, "maps.npz"))
            for i, k in enumerate(z["ref_keys"]):
                maps[str(k)] = (z["gradcam"][i], z["ig512_primary"][i])
    by_pair = {}
    for k in maps:
        uid, mod = k.split("|")
        pid = uid.split(":")[1]
        by_pair.setdefault(pid, {})[mod] = k
    img_rows, vals = [], {}
    for p in common:
        for m in MODS:
            g, ig = maps[by_pair[p][m]]
            rho = spearman(g, ig)
            ov, k, tg, ti = topk_overlap(g, ig)
            if rho is None:
                raise RuntimeError(f"unavailable map in agreement population {p}/{m}")
            img_rows.append({"pair_id": p, "modality": m, "label_mel": lab[p], "spearman": rho, "top10_overlap": ov, "k": k,
                             "gradcam_zero_frac": float((g == 0).mean()), "ig_zero_frac": float((ig == 0).mean()),
                             "gradcam_tie_at_cutoff": tg, "ig_tie_at_cutoff": ti})
            vals[(p, m)] = (rho, ov)
    summ = {}
    vecs = {}
    for j, met in enumerate(("spearman", "top10_overlap")):
        for m in MODS:
            vecs[f"{met}|{m}"] = np.array([vals[(p, m)][j] for p in common])
        vecs[f"{met}|D"] = vecs[f"{met}|clinical"] - vecs[f"{met}|dermoscopy"]
    bs, meta = R4.bootstrap_stats(vecs, B, spec["E4_5"]["seed"])
    for k, v in vecs.items():
        summ[k] = {"n_pairs": len(v), "median": R4.sample_median(v), "median_ci95": bs[k]["median"]["ci95"],
                   "hl": R4.hodges_lehmann(v), "hl_ci95": bs[k]["hl"]["ci95"],
                   "by_class_median": {cl: float(np.median(v[[lab[p] == cl for p in common]])) for cl in ("1", "0")}}
        reps[f"e45|{k}"] = bs[k]["median"]["replicates"]
    res["E4.5"] = {"population": {"n_pairs": len(common), "mel": sum(lab[p] == "1" for p in common),
                                  "nonmel": sum(lab[p] == "0" for p in common),
                                  "gradcam_only_pairs": len(gpairs - ipairs), "ig_only_pairs": len(ipairs - gpairs)},
                   "summaries": summ, "bootstrap": meta,
                   "ties": {"median_gradcam_zero_frac": float(np.median([r["gradcam_zero_frac"] for r in img_rows])),
                            "median_ig_zero_frac": float(np.median([r["ig_zero_frac"] for r in img_rows])),
                            "gradcam_maps_tie_at_cutoff": sum(r["gradcam_tie_at_cutoff"] for r in img_rows),
                            "ig_maps_tie_at_cutoff": sum(r["ig_tie_at_cutoff"] for r in img_rows), "n_maps": len(img_rows)}}
    publish_json(out, "results.json", res)
    publish_csv(out, "e44_subgroup_table.csv", e44_rows, list(e44_rows[0]))
    publish_csv(out, "e45_per_image_agreement.csv", img_rows, list(img_rows[0]))
    da = [dict(v, group=g) for g, v in audit.items()]
    publish_csv(out, "donor_mapping_audit.csv", da, list(da[0]))
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k.replace("|", "__"): v for k, v in reps.items()})
    publish_bytes(out, "bootstrap_replicates.npz", buf.getvalue())
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze"})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out)) if not n.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    print("verify:", P.verify_receipt(rp, cfg["project_root"]) or "OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=("spec", "analyze"))
    ap.add_argument("--out-root")
    ap.add_argument("--run-dir")
    a = ap.parse_args()
    from . import config as C
    cfg = C.load(a.config)
    if cfg["purpose"] != "analysis_v1":
        raise C.ConfigError("requires purpose 'analysis_v1'")
    (phase_spec if a.phase == "spec" else phase_analyze)(cfg, a)


if __name__ == "__main__":
    main()
