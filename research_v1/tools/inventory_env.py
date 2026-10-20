"""Stage 2A / T-A1: read-only environment inventory.

Writes research_v1/env/env_inventory_v1.json and a .sha256 sidecar.
Installs nothing, repairs nothing. GPU is queried only (no tensor compute).
Run with:  python -B research_v1/tools/inventory_env.py [--out PATH]
"""
import argparse, hashlib, importlib, importlib.metadata as md, importlib.util, json
import os, platform, sys, time

# (distribution name, import name)
PACKAGES = [
    ("torch", "torch"), ("torchvision", "torchvision"), ("timm", "timm"),
    ("albumentations", "albumentations"),
    ("opencv-python", "cv2"), ("opencv-python-headless", "cv2"),
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
    ("scikit-learn", "sklearn"), ("matplotlib", "matplotlib"),
    ("pyarrow", "pyarrow"), ("captum", "captum"), ("streamlit", "streamlit"),
]


def dist_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def probe_import(mod):
    rec = {"import_name": mod, "find_spec": importlib.util.find_spec(mod) is not None}
    if not rec["find_spec"]:
        return rec
    try:
        m = importlib.import_module(mod)
        rec["import_ok"] = True
        rec["module_version"] = getattr(m, "__version__", None)
    except Exception as e:  # record, never repair
        rec["import_ok"] = False
        rec["import_error"] = f"{type(e).__name__}: {e}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "env",
                                                  "env_inventory_v1.json"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite existing {out}")

    inv = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "python": {"executable": sys.executable, "version": sys.version,
                      "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                      "in_virtualenv": sys.prefix != sys.base_prefix,
                      "conda_prefix": os.environ.get("CONDA_PREFIX"),
                      "dont_write_bytecode": sys.dont_write_bytecode},
           "platform": {"platform": platform.platform(), "machine": platform.machine()},
           "env_vars": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                        "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE")},
           "packages": []}
    imports = {}
    for dist, mod in PACKAGES:
        if mod not in imports:
            imports[mod] = probe_import(mod)
        inv["packages"].append({"distribution": dist, "distribution_version": dist_version(dist),
                                **imports[mod]})

    gpu = {}
    if imports["torch"].get("import_ok"):
        import torch
        gpu["torch_cuda_build"] = torch.version.cuda
        gpu["cuda_available"] = torch.cuda.is_available()
        gpu["cudnn_version"] = torch.backends.cudnn.version() if gpu["cuda_available"] else None
        if gpu["cuda_available"]:
            p = torch.cuda.get_device_properties(0)
            gpu.update(device_count=torch.cuda.device_count(), name=p.name,
                       total_memory_mib=p.total_memory // (1 << 20),
                       capability=f"{p.major}.{p.minor}")
    inv["gpu"] = gpu

    alb = {}
    if imports["albumentations"].get("import_ok"):
        import inspect, albumentations as A
        alb = {"compose_init_has_seed": "seed" in inspect.signature(A.Compose.__init__).parameters,
               "compose_has_set_random_seed": hasattr(A.Compose, "set_random_seed"),
               "compose_has_set_random_state": hasattr(A.Compose, "set_random_state"),
               "note": "attribute existence only; behaviour not tested"}
    inv["albumentations_seed_api"] = alb

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump(inv, f, indent=2)
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
    with open(out + ".sha256", "x", encoding="utf-8") as f:
        f.write(f"{digest}  {os.path.basename(out)}\n")
    print(json.dumps(inv, indent=2))


if __name__ == "__main__":
    main()
