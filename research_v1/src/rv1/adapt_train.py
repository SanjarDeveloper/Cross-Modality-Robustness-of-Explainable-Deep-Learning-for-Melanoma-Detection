"""Stage 5A RQ5 PAD adaptation (Addendum H): build function for the EXISTING trainer
(rv1.trainer.train) - model initialised from the verified Stage 3A checkpoint (all parameters and
buffers), PAD adapt_train / adapt_val datasets from the frozen manifest. The trainer creates a fresh
optimizer and scheduler; nothing of the base checkpoint's optimizer/scheduler is used.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.adapt_train --config ../configs/train_adapt_pad_s0_v1.json --run-dir ../runs/adapt_train/<name>
"""
import argparse
import csv
import json
import os
import sys
import time

BASE_SHA = "303d21e71986dc8f26dec84c6822a301f696060e5f08f613c90fc33c41a003dd"


def _uids(cfg, rel):
    from . import config as C
    with open(C.resolve(cfg, rel), newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)]


def check_split(cfg, rows):
    """Lists must equal the frozen manifest roles exactly; eval never present; patients disjoint."""
    t = cfg["training"]
    tr, va = _uids(cfg, t["train_subset_file"]), _uids(cfg, t["probe_file"])
    by = {r["record_uid"]: r for r in rows}
    for lst, role in ((tr, "adapt_train"), (va, "adapt_val")):
        u = [r["record_uid"] for r in lst]
        if len(set(u)) != len(u):
            raise RuntimeError(f"{role}: duplicate records")
        if sorted(u) != sorted(r["record_uid"] for r in rows if r["rq5_role"] == role):
            raise RuntimeError(f"{role}: list differs from manifest role")
        for r in lst:
            m = by[r["record_uid"]]
            if m["rq5_role"] != role or m["label_mel"] != r["label_mel"] or m["cluster_id"] != r["cluster_id"]:
                raise RuntimeError(f"{role}: identity mismatch {r['record_uid']}")
    pt = {by[r["record_uid"]]["cluster_id"] for r in tr}
    pv = {by[r["record_uid"]]["cluster_id"] for r in va}
    pe = {r["cluster_id"] for r in rows if r["rq5_role"] == "eval"}
    if pt & pv or pt & pe or pv & pe:
        raise RuntimeError("patient roles overlap")
    return [r["record_uid"] for r in tr], [r["record_uid"] for r in va]


def build_resnet50_adapt(cfg, device):
    import timm
    import torch
    from . import config as C, decode as DC, provenance as P, torch_data as TD, trainer as TR
    t = cfg["training"]
    ck_path = C.resolve(cfg, t["pretrained_weights_file"])
    with open(os.path.join(os.path.dirname(ck_path), "best.json"), encoding="utf-8") as f:
        best = json.load(f)
    if not (t["pretrained_weights_sha256"] == BASE_SHA == best["sha256"] and
            best["checkpoint"] == os.path.basename(ck_path)):
        raise RuntimeError("base checkpoint identity mismatch")
    ck = TR._load_verified(ck_path, BASE_SHA)                  # file hash + sidecar before torch.load
    model = timm.create_model("resnet50", pretrained=False, num_classes=t["num_classes"])
    model.load_state_dict(ck["model"], strict=True)            # parameters AND buffers (BN running stats)
    sd = model.state_dict()
    mism = [k for k in ck["model"] if not torch.equal(sd[k].cpu(), ck["model"][k].cpu())]
    if mism or set(sd) != set(ck["model"]):
        raise RuntimeError(f"base weights not loaded exactly: {mism[:5]}")
    n_tensors = len(sd)
    del ck                                                     # optimizer/scheduler states never used
    model = model.to(device)
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["pad_ufes_20"]), "pad_ufes_20")[0]
    tr_u, va_u = check_split(cfg, rows)
    root = cfg["project_root"]
    cache, dsr = os.path.join(root, "cache"), os.path.join(root, "DATASETS")
    tr = TD.ManifestDataset(rows, "pad_ufes_20", cache, dsr, tr_u, mode="train",
                            recipe=cfg["versions"]["aug_version"], train_seed=cfg["seeds"]["train_seed"])
    va = TD.ManifestDataset(rows, "pad_ufes_20", cache, dsr, va_u, mode="eval")
    labels = [int(r["label_mel"]) for r in tr.rows]
    pos = sum(labels)
    pos_weight = torch.tensor([(len(labels) - pos) / pos], dtype=torch.float32)
    info = {"init": "Stage 3A best checkpoint, strict load of all parameters and buffers",
            "base_checkpoint_sha256": BASE_SHA, "state_tensors_verified": n_tensors,
            "optimizer_scheduler": "fresh (trainer-created); base checkpoint optimizer/scheduler NOT loaded",
            "pos_weight": float(pos_weight.item()), "pos_weight_source": "adapt_train only",
            "n_train": len(tr), "n_train_mel": pos, "n_val": len(va),
            "n_val_mel": sum(int(r["label_mel"]) for r in va.rows), "timm": timm.__version__}
    return model, tr, va, pos_weight, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stop-after-epoch", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    with open(a.config, encoding="utf-8") as f:
        raw = json.load(f)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = raw["training"]["cublas_workspace_config"]
    import torch
    from . import config as C, provenance as P, trainer as TR
    cfg = C.load(a.config)
    if cfg["purpose"] != "train_adapt":
        raise C.ConfigError("requires purpose 'train_adapt'")
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    status, err, res = "completed", None, None
    try:
        res = TR.train(cfg, a.run_dir, build_resnet50_adapt, device, a.stop_after_epoch, a.resume)
    except Exception as e:  # noqa: BLE001 - recorded in the receipt, then re-raised via exit code
        status, err = "failed", f"{type(e).__name__}: {e}"
    inv = {"invocation_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), "resume": a.resume,
           "stop_after_epoch": a.stop_after_epoch, "device": device, "elapsed_seconds": round(time.time() - t0, 1),
           "error": err, "peak_cuda_alloc_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
           "peak_cuda_reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1),
           "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "result": res}
    rec = P.build_receipt(cfg, a.config, status, details=inv)
    P.write_receipt(rec, a.run_dir, f"receipt_{inv['invocation_utc']}_{'resume' if a.resume else 'start'}.json")
    print(json.dumps(inv, indent=1, default=str))
    if status != "completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
