"""Stage 2C3: time ONE full HAM VAL evaluation pass (timing only; no scores kept).

  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.val_timing --config <pilot cfg> --run-dir <pilot run> --out <json>

Loads the pilot's last committed checkpoint (verified SHA-256), builds the same
ResNet-50 (weights file verified), evaluates all HAM VAL records with the evaluation
preprocessing (batch = training batch size, num_workers = config), CUDA-synchronised.
Probabilities are computed (that is the work being timed) but are NOT written, and no
metric is computed: validation results are not used for selection or tuning.
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = json.load(open(a.config, encoding="utf-8"))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = raw["training"]["cublas_workspace_config"]
    import torch
    from . import config as C, decode as DC, torch_data as TD, trainer as TR
    from .smoke_train import build_resnet50
    cfg = C.load(a.config)
    t = cfg["training"]
    dev = "cuda"
    with open(os.path.join(a.run_dir, "latest.json"), encoding="utf-8") as f:
        ptr = json.load(f)
    ck = TR._load_verified(os.path.join(a.run_dir, ptr["checkpoint"]), ptr["sha256"])
    model, _, _, _, _ = build_resnet50(cfg, dev)
    model.load_state_dict(ck["model"], strict=True)
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    root = cfg["project_root"]
    ds = TD.ManifestDataset(rows, "ham_val", os.path.join(root, "cache"), os.path.join(root, "DATASETS"),
                            None, mode="eval")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    with torch.no_grad():
        for b in TD.make_epoch_loader(ds, 0, t["batch_size"], False, 0, t["num_workers"]):
            p = torch.sigmoid(model(b["x"].to(dev)).squeeze(1))
            n += int(p.shape[0])                # probabilities intentionally discarded
    torch.cuda.synchronize()
    s = time.perf_counter() - t0
    res = {"checkpoint": ptr["checkpoint"], "checkpoint_sha256": ptr["sha256"], "n_val": n,
           "batch_size": t["batch_size"], "num_workers": t["num_workers"], "val_pass_seconds": round(s, 3),
           "samples_per_s": round(n / s, 2),
           "peak_cuda_alloc_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
           "peak_cuda_reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1),
           "scores_written": False, "metrics_computed": False}
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
