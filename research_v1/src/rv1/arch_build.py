"""Stage 9A RQ3 architecture adapter: build functions for the EXISTING trainer for timm
efficientnet_b3.ra2_in1k and densenet121.ra_in1k (and resnet50 for reference), with pinned
safetensors weights verified tensor-by-tensor (classifier excluded), fresh one-logit head drawn after
seed_everything(train_seed), HAM datasets from the frozen lists (prep_v1_cached256 at 224 px).

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.arch_build --config <cfg> --run-dir <dir> [--stop-after-epoch N] [--resume]
"""
import argparse
import csv
import json
import os
import sys
import time

CLASSIFIER_PREFIX = {"efficientnet_b3": "classifier.", "densenet121": "classifier.", "resnet50": "fc."}


def verify_backbone(model, ref, arch):
    """Every non-classifier tensor in the weight file must equal the model's parameter/buffer exactly,
    and every non-classifier model tensor must come from the file. Returns counts."""
    import torch
    pre = CLASSIFIER_PREFIX[arch]
    sd = model.state_dict()
    file_keys = [k for k in ref if not k.startswith(pre)]
    model_keys = [k for k in sd if not k.startswith(pre)]
    missing = sorted(set(model_keys) - set(file_keys))
    extra = sorted(set(file_keys) - set(model_keys))
    mism = [k for k in file_keys if k in sd and not torch.equal(sd[k].cpu(), ref[k])]
    if missing or extra or mism or not file_keys:
        raise RuntimeError(f"pretrained weights not loaded exactly: missing={missing[:3]} extra={extra[:3]} mism={mism[:3]}")
    head = {k: list(sd[k].shape) for k in sd if k.startswith(pre)}
    return {"backbone_tensors_verified": len(file_keys), "head_tensors": head,
            "file_classifier_tensors_discarded": sorted(k for k in ref if k.startswith(pre))}


def build_timm(cfg, device):
    import timm
    import torch
    from safetensors.torch import load_file
    from . import config as C, decode as DC, provenance as P, torch_data as TD
    t = cfg["training"]
    arch = t["arch"]
    wpath = C.resolve(cfg, t["pretrained_weights_file"])
    if P.sha256_file(wpath) != t["pretrained_weights_sha256"]:
        raise RuntimeError("pretrained weight file hash differs from the config")
    model = timm.create_model(arch, pretrained=True, num_classes=t["num_classes"], pretrained_cfg_overlay={"file": wpath})
    ref = load_file(wpath)
    info = verify_backbone(model, ref, arch)
    pre = CLASSIFIER_PREFIX[arch]
    w = [v for k, v in model.state_dict().items() if k.startswith(pre) and k.endswith("weight")][0]
    info["head_weight_sha256"] = __import__("hashlib").sha256(w.cpu().numpy().tobytes()).hexdigest()
    info["head_init"] = "timm/torch default nn.Linear init after seed_everything(train_seed); ImageNet classifier discarded"
    info["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["total_params"] = sum(p.numel() for p in model.parameters())
    if info["trainable_params"] != info["total_params"]:
        raise RuntimeError("not all parameters trainable")
    model = model.to(device)
    root = cfg["project_root"]
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    split = {r["record_uid"]: r["split"] for r in rows}

    def uids(rel, want):
        with open(C.resolve(cfg, rel), newline="", encoding="utf-8") as f:
            u = [r["record_uid"] for r in csv.DictReader(f)]
        if any(split.get(x) != want for x in u) or len(set(u)) != len(u):
            raise RuntimeError(f"{rel}: role isolation failure (expected {want})")
        return u
    cache, dsr = os.path.join(root, "cache"), os.path.join(root, "DATASETS")
    tr = TD.ManifestDataset(rows, "ham_train", cache, dsr, uids(t["train_subset_file"], "train"), mode="train",
                            recipe=cfg["versions"]["aug_version"], train_seed=cfg["seeds"]["train_seed"])
    pr = TD.ManifestDataset(rows, "ham_val", cache, dsr, uids(t["probe_file"], "val"), mode="eval")
    labels = [int(r["label_mel"]) for r in tr.rows]
    pos = sum(labels)
    pos_weight = torch.tensor([(len(labels) - pos) / max(pos, 1)], dtype=torch.float32)
    info.update(arch=arch, pretrained_model_id=t["pretrained_model_id"], pos_weight=float(pos_weight.item()),
                n_train=len(tr), n_probe=len(pr), timm=timm.__version__, input_size=224,
                input_note="study-wide prep_v1_cached256 at 224 px (pretrained recommendation recorded in provenance)")
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
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    status, err, res = "completed", None, None
    try:
        res = TR.train(cfg, a.run_dir, build_timm, device, a.stop_after_epoch, a.resume)
    except Exception as e:  # noqa: BLE001
        status, err = "failed", f"{type(e).__name__}: {e}"
    inv = {"invocation_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), "resume": a.resume,
           "stop_after_epoch": a.stop_after_epoch, "elapsed_seconds": round(time.time() - t0, 1), "error": err,
           "peak_cuda_alloc_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
           "peak_cuda_reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1), "result": res}
    os.makedirs(a.run_dir, exist_ok=True)
    P.write_receipt(P.build_receipt(cfg, a.config, status, details=inv), a.run_dir,
                    f"receipt_{inv['invocation_utc']}_{'resume' if a.resume else 'start'}.json")
    print(json.dumps(inv, indent=1, default=str))
    if status != "completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
