"""Artifact contracts checked before publishing and on every resume."""
import json
import math
from pathlib import Path

FACTORS = {"illumination", "blur", "scale", "compression", "nuisance"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def finite_fields(obj, fields):
    for key in fields:
        v = obj[key]
        require(isinstance(v, (int, float)) and math.isfinite(v),
                f"Nonfinite or missing metric {key}: {v}")


def validate_outputs(step, directory, config):
    directory = Path(directory)
    for name in step.outputs:
        p = directory / name
        require(p.is_file() and p.stat().st_size > 0, f"Missing/empty output {p}")
    res = directory / "results"
    tag, kind, smoke = step.tag, step.kind, config["smoke"]
    if kind == "train":
        import torch
        ck = torch.load(directory / "checkpoints" / f"{tag}.pt",
                        map_location="cpu", weights_only=False)
        base = read(res / f"baseline_{tag}.json")
        require(ck["state_dict"], "Empty checkpoint state")
        require(f"{ck['arch']}_{ck['strategy']}_s{ck['seed']}" == tag, "Wrong model tag")
        require(len(ck["history"]) == (1 if smoke else 8), "Incomplete training history")
        require(ck["config"]["subset"] == (300 if smoke else 0), "Wrong training subset")
        require(ck["config"]["epochs"] == (1 if smoke else 8), "Wrong epoch setting")
        require(base["config"] == ck["config"], "Baseline/checkpoint config mismatch")
        require(base["history"] == ck["history"], "Baseline/checkpoint history mismatch")
        require(base["threshold"] == ck["threshold"], "Frozen threshold mismatch")
        finite_fields(ck, ["threshold", "val_auroc"])
        finite_fields(base["test"], ["auroc", "sensitivity"])
        for epoch in ck["history"]:
            finite_fields(epoch, ["train_loss", "val_auroc"])
        for key, value in ck["state_dict"].items():
            require(bool(torch.isfinite(value).all()), f"Nonfinite model tensor: {key}")
    elif kind in ("real", "sweep", "xai"):
        import data as D
        from run_eval import stratified_subset
        source = {"ham10000": D.ham10000("test", seed=0)}
        if kind != "sweep":
            source["derm7pt"] = D.derm7pt()[1]
        if kind == "real":
            source["pad_ufes_20"] = D.pad_ufes_20()
        rows = read(res / f"{kind}_{tag}.json")
        expected = 4 if kind == "real" else (15 if smoke else 30) if kind == "sweep" else 13
        require(len(rows) == expected, f"{kind}: expected {expected} rows, got {len(rows)}")
        seen = set()
        for r in rows:
            require(f"{r['arch']}_{r['strategy']}_s{r['seed']}" == tag, "Wrong row model")
            identity = (r["dataset"], r["modality"], r["factor"], r["severity"])
            require(identity not in seen, f"Duplicate condition {identity}")
            seen.add(identity)
            finite_fields(r, ["sensitivity", "auroc"])
            require(r["n"] > 0 and 0 < r["n_mel"] < r["n"], "Missing evaluation classes")
            recs = source[r["dataset"]]
            if kind == "xai":
                idx = stratified_subset([v.label for v in recs], 40 if smoke else 400)
            else:
                idx = D.smoke_indices(recs, 200) if smoke else range(len(recs))
            require(r["n"] == len(idx) and r["n_mel"] == sum(recs[i].label for i in idx),
                    "Evaluation population does not match requested configuration")
            if kind == "xai":
                require(r["n"] == (40 if smoke else 400), "Wrong XAI sample count")
                finite_fields(r, ["insertion_auc", "deletion_auc",
                                  "insertion_auc_mel", "insertion_auc_ben"])
                if r["dataset"] == "ham10000":
                    finite_fields(r, ["iou", "pointing"])
            elif smoke:
                require(r["n"] == 200, "Wrong smoke evaluation sample count")
        if kind == "real":
            require({(r["dataset"], r["modality"]) for r in rows} == {
                ("derm7pt", "dermoscopy"), ("derm7pt", "clinical"),
                ("pad_ufes_20", "smartphone"), ("ham10000", "dermoscopy")},
                "Incomplete real conditions")
            paired = read(res / f"derm7pt_paired_{tag}.json")
            require(paired["tag"] == tag, "Wrong paired tag")
            d = next(r for r in rows if r["dataset"] == "derm7pt"
                     and r["modality"] == "dermoscopy")
            require(paired["n_pairs"] == d["n"] and paired["n_melanoma"] == d["n_mel"],
                    "Paired population mismatch")
            finite_fields(paired, ["sens_drop", "delong_p", "mcnemar_p"])
            require(len(paired["sens_drop_ci95"]) == 2 and
                    all(math.isfinite(v) for v in paired["sens_drop_ci95"]), "Invalid CI")
        if kind == "sweep":
            sevs = {0.0, 0.5, 1.0} if smoke else {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
            require({(r["factor"], r["severity"]) for r in rows} ==
                    {(f, s) for f in FACTORS for s in sevs}, "Incomplete severity grid")
        if kind == "xai":
            require({(r["factor"], r["severity"]) for r in rows
                     if r["modality"] == "simulated"} ==
                    {(f, s) for f in FACTORS for s in (0.5, 1.0)}, "Incomplete XAI grid")
    elif kind == "rq4":
        r = read(res / f"rq4_tests_{tag}.json")
        require(r["n"] == (40 if smoke else 400) and r["n_melanoma"] == r["n"] // 2,
                "RQ4 population mismatch")
        finite_fields(r["faithfulness_drop_all"], ["mean_change", "wilcoxon_p", "cohens_d"])
        finite_fields(r["melanoma_vs_benign_drop"], ["difference", "mannwhitney_p", "cohens_d"])
    elif kind == "analyse":
        r = read(res / f"report_{tag}.json")
        require(r["tag"] == tag and set(r["rq1_thresholds"]) == FACTORS, "Incomplete report")
        require(len(r["rq2_ranking"]) == 5 and "real_shift" in r["rq4_faithfulness"],
                "Missing report sections")
        finite_fields(r["cmrs_real"], ["cmrs"])
    elif kind == "figures":
        from PIL import Image
        for name in step.outputs:
            with Image.open(directory / name) as image:
                image.verify()
    elif kind == "compare":
        rows = read(res / "rq3_comparison.json")
        expected = {f"{a}_{s}_s0" for a in config["archs"] for s in config["strats"]}
        require(len(rows) == len(expected) and {r["tag"] for r in rows} == expected,
                "Partial architecture comparison")
        for r in rows:
            finite_fields(r, ["sens_retention", "faith_retention", "baseline_auroc_ham"])
    elif kind == "fairness":
        r = read(res / "fairness_report.json")
        require(r["checkpoint"] == f"{tag}.pt", "Wrong fairness checkpoint")
        require(r["ddi"]["groups"] and r["pad_ufes_20"]["groups"], "Empty fairness groups")
        require(r["endpoint_warning"], "Missing off-target warning")
        require(r["off_target_exploratory"]["ddi"]["groups"], "Missing off-target section")
        require(all("malignant_auroc" not in g for g in r["ddi"]["groups"].values()),
                "Off-target malignancy metrics leaked into primary fairness section")
    else:
        raise ValueError(f"No artifact contract for {kind}")
