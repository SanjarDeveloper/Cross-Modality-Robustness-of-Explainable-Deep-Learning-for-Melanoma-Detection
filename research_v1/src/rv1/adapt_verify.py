"""Stage 5A post-training verification (fresh process): selected-checkpoint identity, reload and
re-evaluation of adapt_val (same batch/order/precision), step-log isolation and fresh-optimizer
evidence, per-epoch loss summary.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.adapt_verify --config ../configs/train_adapt_pad_s0_v1.json --run-dir <run> --out <json>
"""
import argparse
import glob
import json
import os

TOL = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import timm
    import torch
    from . import adapt_train as AT, config as C, decode as DC, provenance as P, torch_data as TD, trainer as TR
    from .seeds import seed_everything
    cfg = C.load(a.config)
    seed_everything(0, cfg["determinism"])
    rd = os.path.abspath(a.run_dir)
    with open(os.path.join(rd, "best.json"), encoding="utf-8") as f:
        best = json.load(f)
    ck_path = os.path.join(rd, best["checkpoint"])
    v = {"best_json": best, "file_sha256": P.sha256_file(ck_path)}
    with open(ck_path + ".sha256", encoding="utf-8") as f:
        v["sidecar_sha256"] = f.read().split()[0]
    v["hash_ok"] = v["file_sha256"] == v["sidecar_sha256"] == best["sha256"]
    ck = TR._load_verified(ck_path, best["sha256"])
    v["ckpt_epoch_completed"] = ck["epoch_completed"]
    model = timm.create_model("resnet50", pretrained=False, num_classes=1)
    model.load_state_dict(ck["model"], strict=True)
    model = model.to("cuda").eval()
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["pad_ufes_20"]), "pad_ufes_20")[0]
    _, va_u = AT.check_split(cfg, rows)
    root = cfg["project_root"]
    va = TD.ManifestDataset(rows, "pad_ufes_20", os.path.join(root, "cache"), os.path.join(root, "DATASETS"),
                            va_u, mode="eval")
    uids, probs, labels, auc = TR.evaluate_selection(model, va, "cuda", cfg["training"]["batch_size"], "xai_precision_v1")
    saved = json.load(open(glob.glob(os.path.join(rd, f"val_epoch{best['epoch']}_*.json"))[0], encoding="utf-8"))
    d = max(abs(x - y) for x, y in zip(saved["probs"], probs))
    v["reload"] = {"record_uids_equal": saved["record_uids"] == uids, "labels_equal": saved["labels"] == labels,
                   "max_abs_prob_diff": d, "tolerance": TOL, "auroc_recomputed": auc, "auroc_saved": saved["val_auroc"],
                   "pass": saved["record_uids"] == uids and saved["labels"] == labels and d <= TOL and auc == saved["val_auroc"]}
    steps = [json.loads(x) for x in open(os.path.join(rd, "steps.jsonl"), encoding="utf-8") if x.strip()]
    role = {r["record_uid"]: r["rq5_role"] for r in rows}
    used = {u for s in steps for u in s["record_uids"]}
    v["steps"] = {"n_steps": len(steps), "expected": 5 * -(-805 // 16), "global_steps_1_to_n": [s["global_step"] for s in steps] == list(range(1, len(steps) + 1)),
                  "first_lr_used": steps[0]["lr_used"], "fresh_scheduler_start_lr_equals_config": steps[0]["lr_used"] == cfg["training"]["lr"],
                  "records_used": len(used), "all_adapt_train": all(role[u] == "adapt_train" for u in used),
                  "eval_records_used": sum(role[u] == "eval" for u in used),
                  "adapt_val_records_used": sum(role[u] == "adapt_val" for u in used)}
    loss = {}
    for s in steps:
        loss.setdefault(s["epoch"] + 1, []).append((s["loss"], s["n"]))
    v["epoch_mean_train_loss"] = {e: sum(l * n for l, n in xs) / sum(n for _, n in xs) for e, xs in loss.items()}
    v["epoch_val_auroc"] = {json.loads(x)["epoch"]: json.loads(x)["val_auroc"]
                           for x in open(os.path.join(rd, "selection_decisions.jsonl"), encoding="utf-8") if x.strip()}
    rc = sorted(glob.glob(os.path.join(rd, "receipt_*.json")))
    v["receipts"] = {os.path.basename(r): (P.verify_receipt(r, cfg["project_root"]) or "OK") for r in rc}
    v["all_pass"] = v["hash_ok"] and v["reload"]["pass"] and v["steps"]["all_adapt_train"] and \
        v["steps"]["n_steps"] == v["steps"]["expected"] and v["steps"]["fresh_scheduler_start_lr_equals_config"] and \
        all(x == "OK" for x in v["receipts"].values())
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(v, f, indent=1)
    print(json.dumps(v, indent=1, default=str))


if __name__ == "__main__":
    main()
