"""Stage 4B3: Derm7pt E1.4 paired correctness, positive-class ECE + reliability tables/plots
(Derm7pt by modality, HAM TEST), HAM TEST lesion-cluster bootstrap CIs. Saved predictions only.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.desc_4b3 --config ../configs/analysis_desc_4b3_v1.json --phase spec --out-root ../runs/analysis_desc_4b3
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.desc_4b3 --config ../configs/analysis_desc_4b3_v1.json --phase analyze --run-dir ../runs/analysis_desc_4b3/<run_id>

All outputs are validated (JSON round-trip) and published via a temp file + non-overwriting rename.
"""
import argparse
import csv
import datetime
import io
import json
import os

B = 2000


def publish_bytes(out, name, data):
    final = os.path.join(out, name)
    if os.path.exists(final):
        raise FileExistsError(final)
    tmp = final + ".tmp"
    with open(tmp, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, final)                   # refuses to overwrite on Windows


def publish_json(out, name, obj):
    s = json.dumps(obj, indent=1, default=lambda o: o.item())
    json.loads(s)                           # validate before publishing
    publish_bytes(out, name, s.encode("utf-8"))


def publish_csv(out, name, rows, fields):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    publish_bytes(out, name, buf.getvalue().encode("utf-8"))


def _spec(cfg):
    from . import config as C, provenance as P
    si = cfg["supporting_inputs"]
    inputs = {k: {"path": si[k], "sha256": P.sha256_file(C.resolve(cfg, si[k]))}
              for k in ("pred_derm7pt", "pred_ham_test", "threshold", "threshold_sidecar", "receipt_predict",
                        "verification", "model_checkpoint", "model_best_pointer", "legacy_metrics")}
    inputs["ham10000_manifest"] = {"path": cfg["manifests"]["ham10000"],
                                   "sha256": P.sha256_file(C.resolve(cfg, cfg["manifests"]["ham10000"]))}
    with open(C.resolve(cfg, si["threshold"]), encoding="utf-8") as f:
        thr = json.load(f)
    seed = cfg["seeds"]["bootstrap_seed"]
    return {
        "spec_version": "desc_4b3_v1", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": "02_rev2 §1 (probability: uncalibrated sigmoid; ECE descriptive), §2 bootstrap, §3 E1.4, "
                    "§4 (HAM test cluster = lesion); 03_rev2 §14",
        "inputs": inputs,
        "model": {"sha256": thr["model"]["sha256"], "epoch": thr["model"]["best_epoch"]},
        "threshold": {"value": thr["threshold"], "repr": thr["threshold_repr"], "sha256": inputs["threshold"]["sha256"]},
        "derm7pt": {"cohorts": {"PRIMARY": "in_derm7pt_primary=='1' (1002 pairs)", "ALL": "1011 pairs (sensitivity analysis)"},
                    "pairing": "rq1_stats.build_pairs by pair_id",
                    "E1.4_correctness": "correct = (pred == label); paired 2x2 of correct/incorrect; accuracy per modality; "
                                        "diff = dermoscopy - clinical; two-sided exact McNemar on discordant correctness "
                                        "(rq1_stats.mcnemar_exact) - DESCRIPTIVE, UNADJUSTED; decomposition by label"},
        "ece": {"convention": "ece_pos_v1 (NEWLY SPECIFIED in Stage 4B3 - retrospective implementation decision, "
                              "not preregistered)",
                "type": "positive-class probability calibration (melanoma probability vs melanoma indicator), "
                        "not top-label confidence",
                "bins": "15 equal-width bins on [0,1] with EXACT rational edges b/15; [b/15,(b+1)/15) left-closed/"
                        "right-open; last bin [14/15,1] includes 1; bin = min(floor(Fraction(p)*15), 14)",
                "formula": "ECE = sum_b (n_b/N) |mean(p_b) - mean(y_b)|; empty bins n=0, weight 0",
                "legacy_comparison": "legacy code/eval/metrics.py ece(): positive-class, 15 equal-width bins but "
                                     "right-closed (first bin [0,1/15]); boundary convention differs; v1 not tuned to legacy",
                "populations": ["Derm7pt PRIMARY dermoscopy", "Derm7pt PRIMARY clinical", "HAM TEST",
                                "Derm7pt ALL (sensitivity analysis)"],
                "no_tests_no_fitting": True},
        "ham_test_ci": {"metrics": "image-level sensitivity, specificity (frozen threshold, p>=t), AUROC (Mann-Whitney, "
                                   "ties 1/2)",
                        "cluster": "lesion = manifest cluster_id (HAM10000 lesion_id); NOT patients",
                        "bootstrap": "ordinary lesion-cluster percentile bootstrap, B=2000; all images of a drawn lesion "
                                     "included; multiplicity kept",
                        "seed": seed, "rng": "numpy Generator(PCG64(seed)); integers(0,K,K) over sorted lesion IDs; the "
                                             "same seed for the three metrics, so all metrics share identical draws",
                        "quantiles": "numpy.quantile(method='linear') 0.025/0.975",
                        "undefined": "counted per metric; excluded; not redrawn; >5% -> unreliable"},
        "conditionality": "conditional on the fixed seed-0 model and frozen threshold"}


def phase_spec(cfg, a):
    from . import provenance as P
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "spec"})
    out = os.path.join(a.out_root, receipt["run_id"])
    os.makedirs(out)
    publish_json(out, "analysis_spec.json", _spec(cfg))
    h = P.sha256_file(os.path.join(out, "analysis_spec.json"))
    publish_bytes(out, "analysis_spec.json.sha256", f"{h}  analysis_spec.json\n".encode())
    receipt["outputs"] = {"analysis_spec.json": h}
    P.write_receipt(receipt, out, name="receipt_spec.json")
    print("spec sha256", h, "\nrun dir", out)


def reliability_png(panels, path_bytes_out):
    """panels: list of (title, ece_result). Two rows per panel: reliability (top), counts (bottom)."""
    from PIL import Image, ImageDraw
    W, H, pad = 360, 520, 50
    img = Image.new("RGB", (W * len(panels), H), "white")
    d = ImageDraw.Draw(img)
    for k, (title, r) in enumerate(panels):
        x0 = k * W + pad
        top, side = 40, W - pad - 20
        d.text((k * W + 10, 10), f"{title}  ECE={r['ece']:.4f}  N={r['N']}", fill="black")
        d.rectangle([x0, top, x0 + side, top + side], outline="black")
        d.line([x0, top + side, x0 + side, top], fill=(160, 160, 160))
        for t in (0, 0.5, 1):
            d.text((x0 + t * side - 6, top + side + 3), f"{t:g}", fill="black")
            d.text((x0 - 22, top + side - t * side - 5), f"{t:g}", fill="black")
        d.text((x0 + side // 2 - 50, top + side + 15), "mean predicted p", fill="black")
        d.text((k * W + 2, top + side // 2), "obs.", fill="black")
        bw = side / r["n_bins"]
        for b in r["bins"]:
            if b["empty"]:
                continue
            xm, ym = x0 + b["mean_prob"] * side, top + side - b["observed_mel_fraction"] * side
            d.ellipse([xm - 3, ym - 3, xm + 3, ym + 3], fill=(200, 40, 40))
        ctop = top + side + 40
        ch = H - ctop - 25
        mx = max(b["n"] for b in r["bins"])
        d.rectangle([x0, ctop, x0 + side, ctop + ch], outline="black")
        for b in r["bins"]:
            h = 0 if mx == 0 else b["n"] / mx * ch
            d.rectangle([x0 + b["bin"] * bw + 1, ctop + ch - h, x0 + (b["bin"] + 1) * bw - 1, ctop + ch],
                        fill=(70, 110, 190) if b["n"] else None, outline=(70, 110, 190))
        d.text((x0, ctop + ch + 5), f"counts per bin (max {mx}); 15 bins on [0,1]", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def phase_analyze(cfg, a):
    import numpy as np
    from . import auroc_paired as AP, config as C, decode as DC, desc_stats as DS, provenance as P, rq1_stats as RS
    from . import threshold as TH
    out = os.path.abspath(a.run_dir)
    sp = os.path.join(out, "analysis_spec.json")
    with open(sp + ".sha256", encoding="utf-8") as f:
        if f.read().split()[0] != P.sha256_file(sp):
            raise RuntimeError("spec hash mismatch")
    with open(sp, encoding="utf-8") as f:
        spec = json.load(f)
    for k, v in spec["inputs"].items():
        if P.sha256_file(C.resolve(cfg, v["path"])) != v["sha256"]:
            raise RuntimeError(f"input changed: {k}")
    t = spec["threshold"]["value"]
    si = cfg["supporting_inputs"]
    with open(C.resolve(cfg, si["pred_derm7pt"]), encoding="utf-8", newline="") as f:
        drows = list(csv.DictReader(f))
    with open(C.resolve(cfg, si["pred_ham_test"]), encoding="utf-8", newline="") as f:
        hrows = list(csv.DictReader(f))
    pairs = RS.build_pairs(drows, t, spec["threshold"]["sha256"], spec["model"]["sha256"])
    cohorts = {"PRIMARY": [v for k, v in sorted(pairs.items()) if v["primary"]],
               "ALL": [v for k, v in sorted(pairs.items())]}
    if (len(cohorts["PRIMARY"]), len(cohorts["ALL"])) != (1002, 1011):
        raise RuntimeError("cohort counts")
    res = {"spec_sha256": P.sha256_file(sp)}
    # ---------------- E1.4 correctness
    e14 = {}
    for name, ps in cohorts.items():
        tot = DS.correctness_table(ps)
        mel = DS.correctness_table([p for p in ps if p["label"] == 1])
        neg = DS.correctness_table([p for p in ps if p["label"] == 0])
        mc = RS.mcnemar_exact(tot["dermoscopy_only_correct"], tot["clinical_only_correct"])
        e14[name] = {"role": "descriptive, unadjusted" + ("" if name == "PRIMARY" else "; sensitivity analysis"),
                     "all_cases": tot, "melanoma": mel, "non_melanoma": neg,
                     "decomposition_check": all(tot[k] == mel[k] + neg[k] for k in
                                                ("both_correct", "dermoscopy_only_correct", "clinical_only_correct",
                                                 "both_incorrect")),
                     "mcnemar_correctness_exact": {k: (str(v) if k == "p_exact" else v) for k, v in mc.items()},
                     "class_proportion_mel": mel["n"] / tot["n"]}
    pr = e14["PRIMARY"]
    res["crosscheck_expected"] = {
        "mel_d_only_117": pr["melanoma"]["dermoscopy_only_correct"] == 117,
        "mel_c_only_9": pr["melanoma"]["clinical_only_correct"] == 9,
        "neg_d_only_53": pr["non_melanoma"]["dermoscopy_only_correct"] == 53,
        "neg_c_only_260": pr["non_melanoma"]["clinical_only_correct"] == 260,
        "all_b_170_c_269": (pr["all_cases"]["dermoscopy_only_correct"], pr["all_cases"]["clinical_only_correct"]) == (170, 269)}
    if not all(res["crosscheck_expected"].values()):
        raise RuntimeError(f"cross-check failed {res['crosscheck_expected']}")
    res["E1_4_correctness"] = e14
    # ---------------- HAM TEST join + metrics
    ham = {r["record_uid"]: r for r in DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]}
    hr = []
    seen = set()
    for r in hrows:
        u = r["record_uid"]
        m = ham.get(u)
        if u in seen or m is None or m["split"] != "test" or m["label_mel"] != r["label_mel"]:
            raise RuntimeError(f"HAM TEST join failure at {u}")
        if r["threshold_sha256"] != spec["threshold"]["sha256"] or r["checkpoint_sha256"] != spec["model"]["sha256"]:
            raise RuntimeError("identity mismatch")
        p = float(r["prob"])
        if not (0.0 <= p <= 1.0) or int(r["pred"]) != int(p >= t):
            raise RuntimeError(f"bad prob/pred {u}")
        seen.add(u)
        hr.append({"record_uid": u, "lesion": m["cluster_id"], "label": int(r["label_mel"]), "prob": p})
    if seen != {u for u, m in ham.items() if m["split"] == "test"}:
        raise RuntimeError("HAM TEST coverage")
    pt = TH.counts([r["prob"] for r in hr], [r["label"] for r in hr], t)
    auc = AP.auc([r["prob"] for r in hr if r["label"] == 1], [r["prob"] for r in hr if r["label"] == 0])
    if (pt["TP"], pt["FN"], pt["TN"], pt["FP"], len(hr)) != (157, 13, 911, 467, 1548):
        raise RuntimeError(f"HAM TEST counts differ {pt}")
    lesions = {}
    for r in hr:
        lesions.setdefault(r["lesion"], []).append(r)
    lids = sorted(lesions)
    mel_lesions = sum(any(r["label"] for r in v) for v in lesions.values())
    mixed = sum(len({r["label"] for r in v}) > 1 for v in lesions.values())

    def m_sens(cl):
        return (lambda s: None if s is None else (s,))(DS.binary_metrics(DS.flatten_clusters(cl), t)[0])

    def m_spec(cl):
        return (lambda s: None if s is None else (s,))(DS.binary_metrics(DS.flatten_clusters(cl), t)[1])

    def m_auc(cl):
        f = DS.flatten_clusters(cl)
        pos = [r["prob"] for r in f if r["label"] == 1]
        neg = [r["prob"] for r in f if r["label"] == 0]
        return (AP.auc(pos, neg),) if pos and neg else None
    seed = spec["ham_test_ci"]["seed"]
    boots, reps = {}, {}
    for name, fn in (("sensitivity", m_sens), ("specificity", m_spec), ("auroc", m_auc)):
        b = RS.pair_bootstrap(lesions, lids, fn, B, seed)
        reps[name] = b.pop("replicates")
        boots[name] = b
    res["HAM_TEST"] = {"n_images": len(hr), "n_melanoma_images": sum(r["label"] for r in hr),
                       "n_lesions": len(lids), "n_melanoma_lesions": mel_lesions, "n_mixed_label_lesions": mixed,
                       "max_images_per_lesion": max(len(v) for v in lesions.values()),
                       "point": dict(pt, auroc=auc),
                       "ci95": {k: boots[k]["ci"][0] if boots[k]["ci"] else None for k in boots},
                       "bootstrap": boots,
                       "note": "image-level metrics; lesion-cluster bootstrap for uncertainty; lesions are not patients"}
    # ---------------- ECE
    ece = {}
    pops = {"Derm7pt PRIMARY dermoscopy": [(p["prob_d"], p["label"]) for p in cohorts["PRIMARY"]],
            "Derm7pt PRIMARY clinical": [(p["prob_c"], p["label"]) for p in cohorts["PRIMARY"]],
            "HAM TEST": [(r["prob"], r["label"]) for r in hr],
            "Derm7pt ALL dermoscopy (sensitivity)": [(p["prob_d"], p["label"]) for p in cohorts["ALL"]],
            "Derm7pt ALL clinical (sensitivity)": [(p["prob_c"], p["label"]) for p in cohorts["ALL"]]}
    for name, pv in pops.items():
        ece[name] = DS.ece_bins([x[0] for x in pv], [x[1] for x in pv])
    res["ECE"] = {k: {"ece": v["ece"], "N": v["N"], "n_empty_bins": v["n_empty_bins"]} for k, v in ece.items()}
    # ---------------- publish
    publish_json(out, "results.json", res)
    fields = ["population", "bin", "lower", "upper", "upper_closed", "n", "empty", "mean_prob", "observed_mel_fraction",
              "abs_gap", "weight", "contribution"]
    publish_csv(out, "reliability_bins.csv", [dict(b, population=k) for k, v in ece.items() for b in v["bins"]], fields)
    publish_json(out, "ece_full.json", ece)
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"ham_test_{k}": v for k, v in reps.items()})
    publish_bytes(out, "bootstrap_replicates_ham_test.npz", buf.getvalue())
    publish_bytes(out, "reliability_derm7pt_primary.png",
                  reliability_png([(k, ece[k]) for k in ("Derm7pt PRIMARY dermoscopy", "Derm7pt PRIMARY clinical")], None))
    publish_bytes(out, "reliability_ham_test.png", reliability_png([("HAM TEST", ece["HAM TEST"])], None))
    receipt = P.build_receipt(cfg, a.config, "completed", details={"phase": "analyze", "spec_sha256": res["spec_sha256"]})
    receipt["outputs"] = {n: P.sha256_file(os.path.join(out, n)) for n in sorted(os.listdir(out))
                          if not n.startswith("receipt")}
    rp = P.write_receipt(receipt, out, name="receipt_analyze.json")
    print(json.dumps({k: res[k] for k in ("crosscheck_expected", "ECE")}, indent=1))
    for n, v in e14.items():
        print(n, v["all_cases"], v["mcnemar_correctness_exact"]["p_value_sci"] if "p_value_sci" in v["mcnemar_correctness_exact"] else v["mcnemar_correctness_exact"])
    h = res["HAM_TEST"]
    print({k: h[k] for k in ("n_images", "n_lesions", "n_melanoma_lesions", "n_mixed_label_lesions", "max_images_per_lesion", "point", "ci95")})
    print({k: (v["n_undefined"], v["unreliable"]) for k, v in boots.items()})
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
