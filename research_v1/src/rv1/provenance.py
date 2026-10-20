"""Run receipts (schema rv1.receipt/1) and integrity checking.

A receipt is written once (open mode "x"); an existing receipt is never
overwritten. Its own digest is written to a sidecar `<receipt>.sha256`, never
inside the receipt. Only an explicit allowlist of environment variables is
recorded.

Recorded file hashes (all SHA-256 of file bytes):
  manifests            each manifest; its .sha256 sidecar is REQUIRED and must match
  environment_inventory  likewise requires a matching .sha256 sidecar
  sources              every .py under research_v1/src/rv1 plus config.legacy_sources
  config_file          the config file as read
  supporting_inputs    optional non-executable inputs (e.g. the manifest card): recorded
                       hash only, no sidecar required

Importing this module writes nothing and seeds nothing.
"""
import datetime
import hashlib
import json
import os
import platform
import sys
import uuid

from . import config as C

RECEIPT_SCHEMA = "rv1.receipt/1"
ENV_ALLOWLIST = ("PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE", "CUBLAS_WORKSPACE_CONFIG",
                 "CUDA_VISIBLE_DEVICES")
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
STATUSES = ("completed", "failed")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rel(cfg, path):
    return os.path.relpath(path, cfg["project_root"]).replace("\\", "/")


class ProvenanceError(ValueError):
    pass


_HEX = set("0123456789abcdef")


def read_sidecar(path):
    """Return the digest in `<path>.sha256`. Raises ProvenanceError if the sidecar is
    missing, empty or malformed (first token must be 64 lower-case hex chars; an
    optional second token must be the file's basename)."""
    sc = path + ".sha256"
    if not os.path.isfile(sc):
        raise ProvenanceError(f"sidecar missing: {sc}")
    with open(sc, encoding="utf-8") as f:
        tokens = f.read().split()
    if not tokens:
        raise ProvenanceError(f"sidecar empty: {sc}")
    d = tokens[0]
    if len(d) != 64 or not set(d) <= _HEX:
        raise ProvenanceError(f"sidecar malformed (expected 64 lower-case hex digest): {sc}")
    if len(tokens) > 2 or (len(tokens) == 2 and tokens[1] != os.path.basename(path)):
        raise ProvenanceError(f"sidecar malformed (unexpected file name field): {sc}")
    return d


def checked_input(path):
    """Hash an input file and require that it matches its own sidecar."""
    actual = sha256_file(path)
    side = read_sidecar(path)
    if side != actual:
        raise ProvenanceError(f"sidecar digest does not match file bytes: {path}")
    return {"sha256": actual, "sidecar_sha256": side}


def source_files(cfg):
    files = sorted(os.path.join(SRC_DIR, n) for n in os.listdir(SRC_DIR) if n.endswith(".py"))
    rels = [_rel(cfg, p) for p in files] + list(cfg.get("legacy_sources", []))
    return sorted(set(rels))


def build_receipt(cfg, config_path, status, seed_record=None, details=None):
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    h = C.config_hash(cfg)
    now = datetime.datetime.now(datetime.timezone.utc)
    manifests = {}
    for name, rel in sorted(cfg["manifests"].items()):
        manifests[name] = {"path": rel, **checked_input(C.resolve(cfg, rel))}
    env_rel = cfg["environment_inventory"]
    env_rec = {"path": env_rel, **checked_input(C.resolve(cfg, env_rel))}
    return {
        "schema_version": RECEIPT_SCHEMA,
        "run_id": f"{now:%Y%m%dT%H%M%SZ}_{h[:8]}_{uuid.uuid4().hex[:8]}",
        "created_utc": now.isoformat(timespec="seconds"),
        "run_kind": cfg["run_kind"],
        "purpose": cfg["purpose"],
        "status": status,
        "config": {k: v for k, v in cfg.items() if k != "project_root"},
        "config_sha256": h,
        "config_file": {"path": _rel(cfg, os.path.abspath(config_path)),
                        "sha256": sha256_file(config_path)},
        "manifests": manifests,
        "environment_inventory": env_rec,
        "sources": {r: sha256_file(C.resolve(cfg, r)) for r in source_files(cfg)},
        "supporting_inputs": {n: {"path": rel, "sha256": sha256_file(C.resolve(cfg, rel))}
                              for n, rel in sorted(cfg.get("supporting_inputs", {}).items())},
        "seed_record": seed_record,
        "runtime": {"python_executable": sys.executable, "python_version": sys.version,
                    "platform": platform.platform(),
                    "env_allowlist": {k: os.environ.get(k) for k in ENV_ALLOWLIST}},
        "details": details or {},
    }


def write_receipt(receipt, out_dir, name="receipt.json"):
    """Write receipt + sidecar. Refuses to overwrite. Returns the receipt path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    data = json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8")
    with open(path, "xb") as f:          # FileExistsError if present
        f.write(data)
    with open(path + ".sha256", "x", encoding="utf-8") as f:
        f.write(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
    return path


def verify_receipt(receipt_path, project_root):
    """Re-check everything a receipt references. Returns a list of problems (empty = OK).

    Checks: the receipt's own sidecar; config internal hash; config file re-load
    and canonical equality (project_root excluded); run_kind/purpose agreement;
    for manifests and the environment inventory: actual bytes vs recorded hash,
    recorded sidecar vs recorded hash, current sidecar vs actual bytes; sources.
    """
    problems = []
    try:
        if read_sidecar(receipt_path) != sha256_file(receipt_path):
            problems.append("receipt file does not match its sidecar digest")
    except ProvenanceError as e:
        problems.append(f"receipt: {e}")
    with open(receipt_path, encoding="utf-8") as f:
        r = json.load(f)

    def path_of(rel):
        return os.path.join(project_root, *rel.split("/"))

    # --- configuration consistency
    emb = r.get("config")
    if not isinstance(emb, dict):
        problems.append("config: embedded configuration missing")
    else:
        if C.config_hash(emb) != r.get("config_sha256"):
            problems.append("config: embedded configuration does not match config_sha256")
        for k in ("run_kind", "purpose"):
            if r.get(k) != emb.get(k):
                problems.append(f"config: receipt {k} {r.get(k)!r} != embedded {emb.get(k)!r}")
        cf = r.get("config_file", {})
        cp = path_of(cf.get("path", ""))
        if not os.path.isfile(cp):
            problems.append(f"config_file: missing {cf.get('path')}")
        else:
            if sha256_file(cp) != cf.get("sha256"):
                problems.append(f"config_file: hash mismatch {cf.get('path')}")
            try:
                loaded = C.load(cp)
                if C.canonical_bytes(loaded) != C.canonical_bytes(emb):
                    problems.append("config: embedded configuration differs from config file")
                for k in ("run_kind", "purpose"):
                    if r.get(k) != loaded.get(k):
                        problems.append(f"config: receipt {k} {r.get(k)!r} != config file "
                                        f"{loaded.get(k)!r}")
            except C.ConfigError as e:
                problems.append(f"config_file: invalid ({e})")

    # --- sidecar-backed inputs
    def check_input(label, rec):
        rel = rec.get("path", "")
        p = path_of(rel)
        if not os.path.isfile(p):
            problems.append(f"{label}: missing {rel}")
            return
        actual = sha256_file(p)
        if actual != rec.get("sha256"):
            problems.append(f"{label}: hash mismatch {rel}")
        if rec.get("sidecar_sha256") != rec.get("sha256"):
            problems.append(f"{label}: recorded sidecar digest != recorded hash")
        try:
            if read_sidecar(p) != actual:
                problems.append(f"{label}: current sidecar does not match file bytes")
        except ProvenanceError as e:
            problems.append(f"{label}: {e}")

    for name, m in r.get("manifests", {}).items():
        check_input(f"manifest {name}", m)
    check_input("environment_inventory", r.get("environment_inventory", {}))

    # --- supporting inputs (recorded hash only; no sidecar required)
    emb_si = (emb or {}).get("supporting_inputs", {}) if isinstance(emb, dict) else {}
    rec_si = r.get("supporting_inputs", {})
    if {k: v["path"] for k, v in rec_si.items()} != emb_si:
        problems.append("supporting_inputs: recorded entries differ from embedded config")
    for name, s in rec_si.items():
        p = path_of(s.get("path", ""))
        if not os.path.isfile(p):
            problems.append(f"supporting_input {name}: missing {s.get('path')}")
        elif sha256_file(p) != s.get("sha256"):
            problems.append(f"supporting_input {name}: hash mismatch {s.get('path')}")

    # --- sources (recorded hashes only; no sidecars)
    for rel, want in r.get("sources", {}).items():
        p = path_of(rel)
        if not os.path.isfile(p):
            problems.append(f"source: missing {rel}")
        elif sha256_file(p) != want:
            problems.append(f"source: hash mismatch {rel}")
    return problems


def run_foundation_check(config_path, out_root):
    """Load config, seed explicitly, write a 'foundation_check' receipt."""
    from . import seeds
    cfg = C.load(config_path)
    if cfg["purpose"] != "foundation_check":
        raise C.ConfigError("run_foundation_check requires purpose 'foundation_check'")
    base = cfg["seeds"].get("train_seed")
    if base is None:
        base = cfg["seeds"]["split_seed"]
    rec = seeds.seed_everything(base, cfg["determinism"])
    receipt = build_receipt(cfg, config_path, "completed", rec,
                            {"note": "foundation check only; no data, model or training"})
    out = os.path.join(out_root, receipt["run_id"])
    return write_receipt(receipt, out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    p = run_foundation_check(a.config, a.out_root)
    print(p)
    print("verify:", verify_receipt(p, C.load(a.config)["project_root"]) or "OK")
