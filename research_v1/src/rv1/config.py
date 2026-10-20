"""Explicit, small configuration schema for research_v1 runs (schema rv1.config/1).

Config files are JSON. Unknown fields, wrong types, missing required fields and
invalid enum values raise ConfigError with the offending key path.

Paths
-----
`project_root` is declared in the config. If relative, it is resolved against
the directory containing the config file (never the caller's CWD). Every other
path is relative to project_root, must stay inside it, and is stored in
canonical POSIX form.

Hash
----
config_sha256 = SHA-256 of the canonical JSON of the validated config with
`project_root` removed (so the hash does not depend on where the project lives).
Canonical JSON: sort_keys=True, separators=(",", ":"), ensure_ascii=False, UTF-8.
The schema has no timestamp fields, so nothing volatile enters the hash.

Scientific settings for later stages (training, preprocessing parameters, ...)
are intentionally NOT part of this schema yet; version identifiers are
restricted to the names defined in 03_TECHNICAL_IMPLEMENTATION_PLAN_rev2.md §7
and may be null when not applicable.
"""
import hashlib
import json
import os
import posixpath

SCHEMA_VERSION = "rv1.config/1"
RUN_KINDS = ("lr", "v1")
PURPOSES = ("foundation_check", "data_audit", "mask_audit", "prep_check", "aug_review", "loader_check", "train_smoke", "train_pilot", "xai_diag", "train_main", "predict_v1", "analysis_v1", "train_adapt")
VERSION_ENUMS = {
    "prep_version": ("prep_v1_cached256", "prep_v1_raw224", "legacy_cached256"),
    "aug_version": ("aug_none", "aug_std_v1", "aug_heavy_v1"),
    "perturb_version": ("msim_v1",),
    "metric_version": ("metrics_v1",),
    "xai_version": ("xai_v1",),
    "threshold_rule": ("legacy_first_roc_tpr95", "v1_highest_t_img_sens95"),
    "mask_interp_version": ("mask_interp_v1",),
}
SEED_FIELDS_REQUIRED = ("split_seed",)
SEED_FIELDS_OPTIONAL = ("train_seed", "sample_seed", "control_seed", "bootstrap_seed")
DETERMINISM_FIELDS = ("cudnn_deterministic", "cudnn_benchmark",
                      "use_deterministic_algorithms", "warn_only")
TOP_REQUIRED = ("schema_version", "run_kind", "purpose", "project_root", "manifests",
                "environment_inventory", "seeds", "versions", "determinism")
TOP_OPTIONAL = ("legacy_sources", "supporting_inputs", "description", "training")


class ConfigError(ValueError):
    pass


def _fail(path, msg):
    raise ConfigError(f"config{path}: {msg}")


def _check_keys(d, path, required, optional=()):
    if not isinstance(d, dict):
        _fail(path, f"expected an object, got {type(d).__name__}")
    unknown = sorted(set(d) - set(required) - set(optional))
    if unknown:
        _fail(path, f"unknown field(s) {unknown}")
    missing = [k for k in required if k not in d]
    if missing:
        _fail(path, f"missing required field(s) {missing}")


def _int_seed(v, path, nullable):
    if v is None and nullable:
        return None
    if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 2 ** 63 - 1:
        _fail(path, f"expected a non-negative int seed{' or null' if nullable else ''}, got {v!r}")
    return v


def _relpath(v, path, root, must_exist=True):
    if not isinstance(v, str) or not v:
        _fail(path, f"expected a non-empty relative path string, got {v!r}")
    if os.path.isabs(v) or v.startswith(("/", "\\")):
        _fail(path, f"must be relative to project_root, got absolute {v!r}")
    norm = posixpath.normpath(v.replace("\\", "/"))
    if norm == ".." or norm.startswith("../"):
        _fail(path, f"escapes project_root: {v!r}")
    if must_exist and not os.path.isfile(os.path.join(root, *norm.split("/"))):
        _fail(path, f"file not found under project_root: {norm}")
    return norm


def validate(raw, config_dir):
    """Validate a parsed config dict. Returns a new, normalised dict."""
    _check_keys(raw, "", TOP_REQUIRED, TOP_OPTIONAL)
    if raw["schema_version"] != SCHEMA_VERSION:
        _fail(".schema_version", f"expected {SCHEMA_VERSION!r}, got {raw['schema_version']!r}")
    if raw["run_kind"] not in RUN_KINDS:
        _fail(".run_kind", f"must be one of {list(RUN_KINDS)}, got {raw['run_kind']!r}")
    if raw["purpose"] not in PURPOSES:
        _fail(".purpose", f"must be one of {list(PURPOSES)}, got {raw['purpose']!r}")
    if not isinstance(raw["project_root"], str) or not raw["project_root"]:
        _fail(".project_root", "expected a non-empty string")
    root = os.path.normpath(os.path.join(config_dir, raw["project_root"]))
    if not os.path.isdir(root):
        _fail(".project_root", f"directory not found: {root}")

    out = {"schema_version": SCHEMA_VERSION, "run_kind": raw["run_kind"],
           "purpose": raw["purpose"], "project_root": root}

    m = raw["manifests"]
    if not isinstance(m, dict) or not m:
        _fail(".manifests", "expected a non-empty object {name: relative path}")
    out["manifests"] = {}
    for k, v in m.items():
        if not isinstance(k, str) or not k:
            _fail(".manifests", f"invalid manifest name {k!r}")
        out["manifests"][k] = _relpath(v, f".manifests.{k}", root)
    out["environment_inventory"] = _relpath(raw["environment_inventory"],
                                            ".environment_inventory", root)

    s = raw["seeds"]
    _check_keys(s, ".seeds", SEED_FIELDS_REQUIRED, SEED_FIELDS_OPTIONAL)
    out["seeds"] = {k: _int_seed(s[k], f".seeds.{k}", k not in SEED_FIELDS_REQUIRED)
                    for k in s}

    v = raw["versions"]
    _check_keys(v, ".versions", (), tuple(VERSION_ENUMS))
    out["versions"] = {}
    for k, val in v.items():
        if val is not None and val not in VERSION_ENUMS[k]:
            _fail(f".versions.{k}", f"must be null or one of {list(VERSION_ENUMS[k])}, got {val!r}")
        out["versions"][k] = val
    rule = out["versions"].get("threshold_rule")
    if rule is not None:
        want = "legacy_first_roc_tpr95" if out["run_kind"] == "lr" else "v1_highest_t_img_sens95"
        if rule != want:
            _fail(".versions.threshold_rule", f"run_kind {out['run_kind']!r} requires {want!r}")

    d = raw["determinism"]
    _check_keys(d, ".determinism", DETERMINISM_FIELDS)
    for k in DETERMINISM_FIELDS:
        if not isinstance(d[k], bool):
            _fail(f".determinism.{k}", f"expected a boolean, got {d[k]!r}")
    out["determinism"] = dict(d)

    if "legacy_sources" in raw:
        ls = raw["legacy_sources"]
        if not isinstance(ls, list):
            _fail(".legacy_sources", "expected a list of relative paths")
        out["legacy_sources"] = sorted({_relpath(p, f".legacy_sources[{i}]", root)
                                        for i, p in enumerate(ls)})
    if "supporting_inputs" in raw:
        si = raw["supporting_inputs"]
        if not isinstance(si, dict):
            _fail(".supporting_inputs", "expected an object {name: relative path}")
        out["supporting_inputs"] = {}
        for k, v in si.items():
            if not isinstance(k, str) or not k:
                _fail(".supporting_inputs", f"invalid name {k!r}")
            out["supporting_inputs"][k] = _relpath(v, f".supporting_inputs.{k}", root)
    if "training" in raw:
        out["training"] = _validate_training(raw["training"], root)
    elif out["purpose"] in ("train_smoke", "train_pilot", "train_main", "train_adapt"):
        _fail(".training", "purpose 'train_smoke' requires a training block")
    if "description" in raw:
        if not isinstance(raw["description"], str):
            _fail(".description", "expected a string")
        out["description"] = raw["description"]
    return out


def load(path):
    """Read and validate a JSON config file. Paths resolve from its directory."""
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"{path}: invalid JSON: {e}") from None
    return validate(raw, os.path.dirname(path))


def canonical_bytes(cfg):
    """Canonical JSON of the hashed part of a validated config (project_root excluded)."""
    body = {k: v for k, v in cfg.items() if k != "project_root"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def config_hash(cfg):
    return hashlib.sha256(canonical_bytes(cfg)).hexdigest()


def resolve(cfg, rel):
    """Absolute path for a config-relative POSIX path."""
    return os.path.join(cfg["project_root"], *rel.split("/"))


# ------------------------------------------------------------------ training block
TRAINING_SPEC = {
    # key: (type(s), allowed values or None)
    "arch": (str, ("resnet50", "toy_cnn", "efficientnet_b3", "densenet121")),
    "pretrained_model_id": (str, None),          # e.g. "resnet50.a1_in1k" or "none"
    "pretrained_weights_file": (str, None),      # relative path or "none"
    "pretrained_weights_sha256": (str, None),    # 64 hex or "none"
    "num_classes": (int, (1,)),
    "epochs": (int, None),
    "batch_size": (int, None),
    "optimizer": (str, ("adamw",)),
    "lr": (float, None),
    "weight_decay": (float, None),
    "scheduler": (str, ("cosine_per_step",)),
    "loss": (str, ("bce_with_logits_pos_weight_from_train_subset",)),
    "precision": (str, ("fp32",)),
    "num_workers": (int, None),
    "sampler_seed": (int, None),
    "train_subset_file": (str, None),
    "probe_file": (str, None),
    "cublas_workspace_config": (str, (":4096:8", ":16:8")),
    "best_model_selection": (str, ("disabled", "val_auroc_max_earliest_tie")),
}
TRAINING_OPTIONAL = {
    # Stage 3A: explicit training backend policy and evaluation-precision policy
    "train_backend": (str, ("pytorch_default_tf32conv",)),
    "eval_precision": (str, ("training_default", "xai_precision_v1")),
}


def _validate_training(t, root):
    _check_keys(t, ".training", tuple(TRAINING_SPEC), tuple(TRAINING_OPTIONAL))
    out = {}
    for k, (typ, allowed) in list(TRAINING_SPEC.items()) + list(TRAINING_OPTIONAL.items()):
        if k not in t:
            continue
        v = t[k]
        ok = isinstance(v, typ) and not isinstance(v, bool)
        if typ is float and isinstance(v, int) and not isinstance(v, bool):
            ok = True
        if not ok:
            _fail(f".training.{k}", f"expected {typ.__name__}, got {v!r}")
        if allowed is not None and v not in allowed:
            _fail(f".training.{k}", f"must be one of {list(allowed)}, got {v!r}")
        if typ is int and v < 0:
            _fail(f".training.{k}", "must be non-negative")
        out[k] = float(v) if typ is float else v
    for k in ("epochs", "batch_size"):
        if out[k] < 1:
            _fail(f".training.{k}", "must be >= 1")
    for k in ("train_subset_file", "probe_file"):
        out[k] = _relpath(out[k], f".training.{k}", root)
    if out["pretrained_weights_file"] != "none":
        out["pretrained_weights_file"] = _relpath(out["pretrained_weights_file"],
                                                  ".training.pretrained_weights_file", root)
    return out
