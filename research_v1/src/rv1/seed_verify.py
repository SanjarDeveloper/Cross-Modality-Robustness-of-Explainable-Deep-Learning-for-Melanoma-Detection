"""Stage 6A post-training verification for a HAM seed replicate (fresh process): selected checkpoint
identity, HAM VAL reproduction (same order/batch/precision; tolerance frozen in Addendum I3), step-log
data isolation and fresh-start evidence, receipts, per-epoch summaries.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.seed_verify --config <cfg> --run-dir <run> --out <json>
"""
import argparse
import csv
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
    from . import config as C, decode as DC, provenance as P, torch_data as TD, trainer as TR
    from .seeds import seed_everything
    cfg = C.load(a.config)
    seed_everything(cfg["seeds"]["train_seed"], cfg["determinism"])
    rd = os.path.abspath(a.run_dir)
    best = json.load(open(os.path.join(rd, "best.json"), encoding="utf-8"))
    ck_path = os.path.join(rd, best["checkpoint"])
    v = {"best_json": best, "file_sha256": P.sha256_file(ck_path),
         "sidecar_sha256": open(ck_path + ".sha256", encoding="utf-8").read().split()[0]}
    v["hash_ok"] = v["file_sha256"] == v["sidecar_sha256"] == best["sha256"]
    ck = TR._load_verified(ck_path, best["sha256"])
    v["ckpt_epoch_completed"] = ck["epoch_completed"]
    v["ckpt_train_seed"] = ck["config"]["seeds"]["train_seed"]
    v["identity_ok"] = ck["epoch_completed"] == best["epoch"] and ck["config"]["seeds"]["train_seed"] == cfg["seeds"]["train_seed"]
    model = timm.create_model("resnet50", pretrained=False, num_classes=1)
    model.load_state_dict(ck["model"], strict=True)
    model = model.to("cuda").eval()
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    with open(C.resolve(cfg, cfg["training"]["probe_file"]), encoding="utf-8", newline="") as f:
        uids = [r["record_uid"] for r in csv.DictReader(f)]
    root = cfg["project_root"]
    va = TD.ManifestDataset(rows, "ham_val", os.path.join(root, "cache"), os.path.join(root, "DATASETS"), uids, mode="eval")
    u2, probs, labels, auc = TR.evaluate_selection(model, va, "cuda", cfg["training"]["batch_size"], "xai_precision_v1")
    saved = json.load(open(glob.glob(os.path.join(rd, f"val_epoch{best['epoch']}_*.json"))[0], encoding="utf-8"))
    d = max(abs(x - y) for x, y in zip(saved["probs"], probs))
    v["reload"] = {"record_uids_equal": saved["record_uids"] == u2, "labels_equal": saved["labels"] == labels,
                   "max_abs_prob_diff": d, "tolerance": TOL, "auroc_recomputed": auc, "auroc_saved": saved["val_auroc"],
                   "pass": saved["record_uids"] == u2 and saved["labels"] == labels and d <= TOL and auc == saved["val_auroc"]}
    split = {r["record_uid"]: r["split"] for r in rows}
    steps = [json.loads(x) for x in open(os.path.join(rd, "steps.jsonl"), encoding="utf-8") if x.strip()]
    used = {u for s in steps for u in s["record_uids"]}
    v["steps"] = {"n_steps": len(steps), "expected": 8 * -(-6938 // 16), "first_lr_used": steps[0]["lr_used"],
                  "fresh_start_lr_ok": steps[0]["lr_used"] == cfg["training"]["lr"],
                  "global_steps_1_to_n": [s["global_step"] for s in steps] == list(range(1, len(steps) + 1)),
                  "records_used": len(used), "all_ham_train": all(split.get(u) == "train" for u in used),
                  "non_train_records": sum(split.get(u) != "train" for u in used),
                  "all_finite": all(s["finite"] for s in steps)}
    v["val_records_all_ham_val"] = all(split.get(u) == "val" for u in saved["record_uids"])
    rs = json.load(open(os.path.join(rd, "run_start.json"), encoding="utf-8"))
    v["run_start"] = {"backbone_tensors_verified": rs["build_info"]["backbone_tensors_verified"],
                      "pos_weight": rs["build_info"]["pos_weight"], "fc_head_sha256": rs["build_info"]["fc_head"]["weight_sha256"],
                      "seed_record_seed": rs["seed_record"].get("seed")}
    loss = {}
    for s in steps:
        loss.setdefault(s["epoch"] + 1, []).append((s["loss"], s["n"]))
    v["epoch_mean_train_loss"] = {e: sum(l * n for l, n in xs) / sum(n for _, n in xs) for e, xs in loss.items()}
    v["epoch_val_auroc"] = {json.loads(x)["epoch"]: json.loads(x)["val_auroc"]
                           for x in open(os.path.join(rd, "selection_decisions.jsonl"), encoding="utf-8") if x.strip()}
    v["epochs_log"] = [json.loads(x) for x in open(os.path.join(rd, "epochs.jsonl"), encoding="utf-8") if x.strip()]
    v["receipts"] = {os.path.basename(r): (P.verify_receipt(r, cfg["project_root"]) or "OK")
                     for r in sorted(glob.glob(os.path.join(rd, "receipt_*.json")))}
    v["all_pass"] = v["hash_ok"] and v["identity_ok"] and v["reload"]["pass"] and v["steps"]["all_ham_train"] and \
        v["steps"]["n_steps"] == v["steps"]["expected"] and v["steps"]["fresh_start_lr_ok"] and \
        v["val_records_all_ham_val"] and all(x == "OK" for x in v["receipts"].values())
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(v, f, indent=1)
    print(json.dumps({k: v[k] for k in ("best_json", "hash_ok", "identity_ok", "reload", "steps", "run_start",
                                        "epoch_val_auroc", "all_pass")}, indent=1))


if __name__ == "__main__":
    main()
