"""Stable seed derivation and explicit RNG setup.

stable_seed(*parts)
-------------------
Deterministic across processes, platforms and PYTHONHASHSEED values. Python's
built-in hash() is never used.

Encoding (version "rv1.stable_seed/1"):
    payload = b"rv1.stable_seed/1" + u64(len(parts)) + enc(p1) + enc(p2) + ...
    enc(p)  = tag (1 byte) + u64(len(body)) + body          (u64 = 8-byte big-endian)
      bool  -> tag b"B", body b"\x01"/b"\x00"   (checked before int; bool is an int subclass)
      int   -> tag b"I", body = ASCII decimal (with leading "-" if negative)
      str   -> tag b"S", body = UTF-8
      bytes -> tag b"Y", body = raw bytes
    Any other type raises TypeError (floats are rejected: their text form is ambiguous).
The type tag removes type ambiguity (1 vs "1"); the length prefix removes
delimiter ambiguity ("a|b","c" vs "a","b|c"). Distinct inputs map to distinct
payloads; the SHA-256 digests of distinct payloads can in principle collide,
but no collision is known.

Result: int(first 8 bytes of SHA-256(payload), big-endian) & (2**63 - 1),
i.e. an integer in [0, 2**63 - 1]. Library-specific seeds are derived from it
(see seed_everything) where a library needs a narrower range.

PYTHONHASHSEED
--------------
Setting os.environ["PYTHONHASHSEED"] inside a running interpreter does NOT
change that interpreter's str/bytes hash randomisation; it only affects child
processes started afterwards with that environment. This module does not rely
on hash() at all; the value is merely recorded for provenance.

Importing this module seeds nothing and does not import torch or numpy.
"""
import hashlib
import os
import random
import struct

ENCODING_VERSION = "rv1.stable_seed/1"
SEED_MAX = 2 ** 63 - 1


def _enc(p):
    if isinstance(p, bool):
        tag, body = b"B", (b"\x01" if p else b"\x00")
    elif isinstance(p, int):
        tag, body = b"I", str(p).encode("ascii")
    elif isinstance(p, str):
        tag, body = b"S", p.encode("utf-8")
    elif isinstance(p, bytes):
        tag, body = b"Y", p
    else:
        raise TypeError(f"stable_seed: unsupported part type {type(p).__name__!r} "
                        "(allowed: bool, int, str, bytes)")
    return tag + struct.pack(">Q", len(body)) + body


def encode_parts(*parts):
    """Exact byte payload hashed by stable_seed (exposed for tests/audit)."""
    return (ENCODING_VERSION.encode("ascii") + struct.pack(">Q", len(parts))
            + b"".join(_enc(p) for p in parts))


def stable_seed(*parts):
    """Integer seed in [0, 2**63-1] derived from SHA-256 of the typed encoding."""
    if not parts:
        raise ValueError("stable_seed requires at least one part")
    d = hashlib.sha256(encode_parts(*parts)).digest()
    return int.from_bytes(d[:8], "big") & SEED_MAX


def seed_everything(seed, determinism):
    """Explicitly seed Python, NumPy (legacy global) and Torch (CPU + CUDA, lazily).

    `determinism` must contain boolean keys: cudnn_deterministic, cudnn_benchmark,
    use_deterministic_algorithms, warn_only. Returns a JSON-serialisable record.
    Does not initialise CUDA; torch.manual_seed only queues CUDA seeding.
    Bitwise GPU reproducibility is NOT claimed.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= SEED_MAX:
        raise ValueError(f"seed must be an int in [0, {SEED_MAX}], got {seed!r}")
    required = {"cudnn_deterministic", "cudnn_benchmark", "use_deterministic_algorithms",
                "warn_only"}
    if set(determinism) != required or not all(isinstance(v, bool) for v in determinism.values()):
        raise ValueError(f"determinism must have exactly the boolean keys {sorted(required)}")

    derived = {"python": stable_seed(seed, "python"),
               "numpy_legacy": stable_seed(seed, "numpy") % (2 ** 32),
               "torch": stable_seed(seed, "torch")}
    random.seed(derived["python"])
    rec = {"base_seed": seed, "encoding_version": ENCODING_VERSION, "derived_seeds": derived,
           "numpy": None, "torch": None,
           "env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                   "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
           "note": "No bitwise GPU reproducibility is claimed."}
    try:
        import numpy as np
        np.random.seed(derived["numpy_legacy"])
        rec["numpy"] = {"version": np.__version__, "seeded": True}
    except ImportError:
        rec["numpy"] = {"available": False}
    try:
        import torch
    except ImportError:
        rec["torch"] = {"available": False}
        return rec
    torch.manual_seed(derived["torch"])
    torch.backends.cudnn.deterministic = determinism["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = determinism["cudnn_benchmark"]
    torch.use_deterministic_algorithms(determinism["use_deterministic_algorithms"],
                                       warn_only=determinism["warn_only"])
    rec["torch"] = {
        "version": torch.__version__, "seeded": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_initialized_after_call": torch.cuda.is_initialized(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    }
    return rec


def numpy_generator(*parts):
    """A fresh numpy Generator seeded by stable_seed(*parts) (no global state)."""
    import numpy as np
    return np.random.default_rng(stable_seed(*parts))
