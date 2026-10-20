"""Stage 2C2 CLI: bounded ResNet-50 FP32 smoke training with epoch-boundary resume.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.smoke_train --config ../configs/train_smoke_v1.json --run-dir <dir> [--stop-after-epoch 1] [--resume]

--stop-after-epoch / --resume / --run-dir are OPERATIONAL arguments; they are not part
of the config or of the resume fingerprint. CUBLAS_WORKSPACE_CONFIG is set from the
config BEFORE torch is imported (i.e. before CUDA initialisation).
"""
import argparse
import csv
import json
import os
import sys
import time


def build_resnet50(cfg, device):
    import timm
    import torch
    from safetensors.torch import load_file
    from . import config as C, decode as DC, provenance as P, torch_data as TD
    t = cfg["training"]
    wpath = C.resolve(cfg, t["pretrained_weights_file"])
    if P.sha256_file(wpath) != t["pretrained_weights_sha256"]:
        raise RuntimeError("pretrained weight file hash differs from the config")
    model = timm.create_model("resnet50", pretrained=True, num_classes=t["num_classes"],
                              pretrained_cfg_overlay={"file": wpath})
    ref = load_file(wpath)
    sd = model.state_dict()
    backbone = [k for k in ref if not k.startswith("fc.")]
    mism = [k for k in backbone if not torch.equal(sd[k].cpu(), ref[k])]
    if mism or not backbone:
        raise RuntimeError(f"pretrained weights not loaded exactly: {mism[:5]}")
    fc_head = {"weight_shape": list(sd["fc.weight"].shape), "bias_shape": list(sd["fc.bias"].shape),
               "init": "timm/torch nn.Linear default (not from the checkpoint; ImageNet fc discarded "
                       "because num_classes=1), drawn after seed_everything(train_seed)",
               "weight_sha256": __import__("hashlib").sha256(sd["fc.weight"].numpy().tobytes()).hexdigest()}
    model = model.to(device)
    root = cfg["project_root"]
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]

    def uids(rel):
        with open(C.resolve(cfg, rel), newline="", encoding="utf-8") as f:
            return [r["record_uid"] for r in csv.DictReader(f)]
    cache, dsr = os.path.join(root, "cache"), os.path.join(root, "DATASETS")
    tr = TD.ManifestDataset(rows, "ham_train", cache, dsr, uids(t["train_subset_file"]),
                            mode="train", recipe=cfg["versions"]["aug_version"],
                            train_seed=cfg["seeds"]["train_seed"])
    pr = TD.ManifestDataset(rows, "ham_val", cache, dsr, uids(t["probe_file"]), mode="eval")
    labels = [int(r["label_mel"]) for r in tr.rows]
    pos = sum(labels)
    pos_weight = torch.tensor([(len(labels) - pos) / max(pos, 1)], dtype=torch.float32)
    info = {"backbone_tensors_verified": len(backbone), "fc_head": fc_head,
            "label_convention": "logit>0 / sigmoid -> melanoma (label_mel=1); non-melanoma=0",
            "pos_weight": float(pos_weight.item()), "n_train": len(tr), "n_probe": len(pr),
            "timm": timm.__version__}
    return model, tr, pr, pos_weight, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stop-after-epoch", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    raw = json.load(open(a.config, encoding="utf-8"))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = raw["training"]["cublas_workspace_config"]
    import torch
    from . import config as C, provenance as P, trainer as TR
    cfg = C.load(a.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    status, err = "completed", None
    try:
        res = TR.train(cfg, a.run_dir, build_resnet50, device, a.stop_after_epoch, a.resume)
    except Exception as e:  # noqa: BLE001 - recorded in the receipt, then re-raised
        status, err, res = "failed", f"{type(e).__name__}: {e}", None
    inv = {"invocation_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
           "resume": a.resume, "stop_after_epoch": a.stop_after_epoch, "device": device,
           "elapsed_seconds": round(time.time() - t0, 1), "error": err,
           "peak_cuda_alloc_mib": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                                   if device == "cuda" else None),
           "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
           "result": res}
    rec = P.build_receipt(cfg, a.config, status, details=inv)
    name = f"receipt_{inv['invocation_utc']}_{'resume' if a.resume else 'start'}.json"
    P.write_receipt(rec, a.run_dir, name)
    print(json.dumps(inv, indent=1, default=str))
    if status != "completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
