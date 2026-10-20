"""Stage 2C2 contract setup: frozen subset/probe lists + pretrained-weight provenance.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.smoke_setup --lists-out ../configs --weights-out ../runs/stage2c2/weights

Sampling rules (fixed before any training; never changed after seeing results):
  train subset: HAM split=="train"; per class (label_mel 1 / 0) order records by
                (stable_seed("smoke_subset_v1", record_uid), record_uid); take 128 + 128.
  probe:        HAM split=="val"; same rule with "smoke_probe_v1"; take 32 + 32.
                The probe is used ONLY for resume comparisons, never for tuning.
Weights: timm "resnet50" resolves to the default pretrained tag; the file named by its
pretrained_cfg.hf_hub_id is downloaded with huggingface_hub from the Hugging Face Hub
(the source timm itself uses), into a project-local cache. Its SHA-256 and the Hub
commit are recorded. No fallback to random initialisation.
"""
import argparse
import csv
import hashlib
import json
import os

from . import decode as DC
from .seeds import stable_seed

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _pick(rows, split, tag, n_per_class):
    out = []
    for lab in ("1", "0"):
        rr = sorted((r for r in rows if r["split"] == split and r["label_mel"] == lab),
                    key=lambda r: (stable_seed(tag, r["record_uid"]), r["record_uid"]))
        if len(rr) < n_per_class:
            raise ValueError(f"only {len(rr)} {split} records with label {lab}")
        out += rr[:n_per_class]
    return out


def write_list(path, rows):
    cols = ["record_uid", "image_uid", "label_mel", "split", "cluster_id", "cache_name", "cache_index"]
    with open(path, "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists-out", required=True)
    ap.add_argument("--weights-out", required=True)
    a = ap.parse_args()
    rows = DC.read_manifest(os.path.join(PROJECT, "research_v1", "manifests", "v1", "ham10000_v1.csv"),
                            "ham10000")[0]
    sub = _pick(rows, "train", "smoke_subset_v1", 128)
    probe = _pick(rows, "val", "smoke_probe_v1", 32)
    assert not {r["cluster_id"] for r in sub} & {r["cluster_id"] for r in probe}
    write_list(os.path.join(a.lists_out, "smoke_train_subset_v1.csv"), sub)
    write_list(os.path.join(a.lists_out, "smoke_val_probe_v1.csv"), probe)

    import timm
    import huggingface_hub as hh
    cfg = timm.models.get_pretrained_cfg("resnet50")
    repo = cfg.hf_hub_id
    info = hh.model_info(repo, files_metadata=True)
    os.makedirs(a.weights_out, exist_ok=True)
    local = hh.hf_hub_download(repo, "model.safetensors", revision=info.sha,
                               cache_dir=os.path.abspath(a.weights_out))
    h = hashlib.sha256()
    with open(local, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    remote = next((s for s in info.siblings if s.rfilename == "model.safetensors"), None)
    prov = {"timm_version": timm.__version__, "huggingface_hub_version": hh.__version__,
            "timm_model_name": "resnet50", "resolved_pretrained_tag": cfg.tag,
            "pretrained_model_id": f"resnet50.{cfg.tag}", "hf_hub_id": repo,
            "hf_revision_commit": info.sha, "file": "model.safetensors",
            "source_url": f"https://huggingface.co/{repo}/resolve/{info.sha}/model.safetensors",
            "timm_cfg_legacy_url_not_used": cfg.url,
            "local_path": os.path.relpath(local, PROJECT).replace("\\", "/"),
            "sha256": h.hexdigest(),
            "hub_lfs_sha256": getattr(getattr(remote, "lfs", None), "sha256", None) if remote else None,
            "size_bytes": os.path.getsize(local), "license": cfg.license,
            "pretrained_cfg": {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in cfg.to_dict().items()},
            "train_subset": {"n": len(sub), "melanoma": sum(r["label_mel"] == "1" for r in sub)},
            "probe": {"n": len(probe), "melanoma": sum(r["label_mel"] == "1" for r in probe)}}
    with open(os.path.join(a.weights_out, "weight_provenance.json"), "x", encoding="utf-8") as f:
        json.dump(prov, f, indent=1)
    print(json.dumps({k: prov[k] for k in ("pretrained_model_id", "hf_hub_id", "hf_revision_commit",
                                           "sha256", "hub_lfs_sha256", "local_path", "train_subset",
                                           "probe")}, indent=1))


if __name__ == "__main__":
    main()
