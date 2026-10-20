# Stage 2B1 Report — Configuration, Seeds, Provenance

Date: 2026-09-27

## Status: **PASS**

All 13 required tests passed. A real foundation-check receipt was written and then verified. The frozen manifests and the legacy files were not changed.

This result does not authorise training.

## 1. Files created

No existing file was modified.

| Path | Purpose |
|---|---|
| `research_v1/src/rv1/__init__.py` | Package marker. Importing it has no side effects |
| `research_v1/src/rv1/config.py` | Config schema `rv1.config/1`, validation, canonical hash |
| `research_v1/src/rv1/seeds.py` | `stable_seed`, `encode_parts`, `seed_everything`, `numpy_generator` |
| `research_v1/src/rv1/provenance.py` | Receipt schema `rv1.receipt/1`: build, write (no overwrite, sidecar), verify, and a foundation-check CLI |
| `research_v1/tests/test_foundations.py` | 13 unittest tests (stdlib) |
| `research_v1/configs/foundation_check_v1.json` | Minimal config for the foundation check only |
| `research_v1/runs/foundation_checks/20260927T104607Z_6d95d9d3_8e91758f/receipt.json` + `.sha256` | Real receipt. Config SHA-256 `6d95d9d3…0add`. Verification result: OK |
| `research_v1/docs/STAGE2B1_REPORT.md` | This report |

Before writing, I checked that none of these paths already existed: `src/`, `tests/`, `configs/` and `runs/` were all absent.

## 2. Design decisions

### Config

- The config is JSON with an explicit schema. Unknown fields raise an error, and so do missing fields, wrong types and invalid enum values. Each error message names the key path (for example `config.seeds.split_seed: …`).
- `run_kind` must be `lr` or `v1`.
- `purpose` currently allows only `foundation_check`, so no settings for later stages are invented here.
- `split_seed` is required. The other seed fields (`train_seed`, `sample_seed`, `control_seed`, `bootstrap_seed`) are optional and may be int or null.
- Version identifiers are limited to the names defined in 03_rev2 §7, or null. `threshold_rule` must match the `run_kind`.
- The four determinism flags must be booleans.
- **Paths:**
  - `project_root` is resolved relative to the directory of the config file, not the current working directory.
  - All other paths are relative to that root, must exist, may not be absolute and may not escape the root.
  - Paths are stored in POSIX form.
- **Hash:**
  - The hash is the SHA-256 of canonical JSON (sorted keys, compact separators, UTF-8).
  - `project_root` is left out, so the hash does not depend on where the project lives on disk.
  - The schema has no timestamp fields.
  - `legacy_sources` is sorted before hashing.

### Seeds

- **Encoding for `stable_seed` (`rv1.stable_seed/1`):**
  - The payload is the version string, then a u64 part count, then each part as `tag + u64 length + body`.
  - Supported tags: `B` bool, `I` int (ASCII decimal), `S` str (UTF-8), `Y` bytes.
  - Floats and any other type raise `TypeError`.
  - Output range: [0, 2⁶³−1], taken from the first 8 bytes of the SHA-256 digest.
  - Python's built-in `hash()` is not used anywhere.
  - Distinct inputs give distinct payloads. Digest collisions are not impossible, but none is known.
- **`seed_everything(seed, determinism)`:**
  - Each library gets its own derived seed: Python `random`, NumPy's legacy global RNG (reduced mod 2³²), and Torch (`manual_seed`).
  - It applies the cuDNN deterministic/benchmark flags and `use_deterministic_algorithms(…, warn_only)`.
  - It returns a record containing the values it applied, whether CUDA is available, and whether CUDA was initialised after the call. In the real receipt: CUDA available, not initialised.
  - Without Torch or NumPy it records "unavailable" and does not crash.
  - It makes no claim of bitwise GPU reproducibility.
- **PYTHONHASHSEED:** the docstring explains that setting `PYTHONHASHSEED` inside a running interpreter has no effect on that interpreter. The value is only recorded. Tests pass it explicitly to subprocesses.
- **Lazy imports:** the modules import NumPy and Torch only inside functions. Importing them seeds nothing and starts no GPU work (test 9).

### Provenance

A receipt contains:
- the schema version, a unique `run_id` (UTC time + config-hash prefix + random suffix) and a UTC timestamp;
- `run_kind`, `purpose` and `status` (`completed` or `failed`);
- the resolved config (without the absolute root) and its hash;
- the path and hash of the config file;
- each manifest's hash, together with the value from its `.sha256` sidecar;
- the hash of the environment inventory, and of its sidecar;
- source hashes for every `src/rv1/*.py` file **plus** the reused legacy and tool sources listed in the config: `code/labels.py`, `code/data.py`, `research_v1/tools/*.py`;
- the seed and determinism record;
- an allowlist of environment variables: `PYTHONHASHSEED`, `PYTHONDONTWRITEBYTECODE`, `CUBLAS_WORKSPACE_CONFIG`, `CUDA_VISIBLE_DEVICES`.

Writing rules:
- The receipt is written with mode `x`, so an existing receipt is never overwritten.
- Its own digest goes only into the `.sha256` sidecar, never into the receipt itself.
- `verify_receipt` re-hashes every referenced file and also checks the sidecar.

## 3. Test command and results

```bash
set PYTHONDONTWRITEBYTECODE=1
python -B -m unittest discover -s research_v1/tests -p "test_*.py" -v
```

Interpreter: `C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe` (3.10.11), the same one used in Stage 2A.

**Result: `Ran 13 tests in 3.979s — OK`** (the time is unittest's own measurement).

| Required test | Test(s) | Result |
|---|---|---|
| 1. Same seeds across subprocesses with PYTHONHASHSEED 0 and 12345 | `T1…independent_of_pythonhashseed` (also confirms that the built-in `hash()` differs between the two) | ok |
| 2. Ambiguous-looking inputs are distinct: (1,"23"), ("1",23), ("a\|b","c"), ("a","b\|c") and 10 more; bad types are rejected | `T2` ×2 | ok |
| 3. Python, NumPy and Torch CPU sequences reproduce under the same seed, and change under a different seed | `T3` ×3 | ok |
| 4. Key order does not change the hash; changing a setting does; the hash does not depend on the current directory | `T4` ×2 | ok |
| 5. 14 invalid configurations each give a specific message | `T5` | ok |
| 6. Receipt hashes match the referenced files; the receipt's own digest is not inside it; the environment allowlist is correct | `test6` | ok |
| 7. Changes to a temporary manifest, a source file, or the receipt itself are detected | `test7` (temporary files only) | ok |
| 8. Overwriting a receipt raises `FileExistsError`, and the file is left unchanged | `test8` | ok |
| 9. Importing the modules leaves RNG states unchanged, writes no files, does not initialise CUDA, and creates no `__pycache__` | `T9` | ok |

**Real receipt:** `python -B -m rv1.provenance --config ../configs/foundation_check_v1.json --out-root ../runs/foundation_checks`, run from `research_v1/src`. Output: `verify: OK`.

**Measured elapsed time:** the test suite took 3.98 s. Wall-clock time for the other commands was not captured, because the timer command failed on a path-escaping error in the shell. This is machine time, not human effort.

## 4. Integrity of legacy and frozen files

I took SHA-256 snapshots before and after, covering **157 files**:

- `code/`, `results/`, `checkpoints/`, `audit/manifests/`, `protocol/`, `logs/`;
- `research_v1/docs/`, `research_v1/manifests/`, `research_v1/env/`, `research_v1/tools/`;
- `audit/*.*`, `cache/*_meta.json`, the top-level `*.md` and `*.docx` files, and the 4 dataset metadata files.

For the cache `.npy` files, only size and modification time were checked.

**Result:** 0 added, 0 removed, 0 changed. The `.npy` size and modification-time values are identical. All 5 manifest CSVs still match their `.sha256` sidecars. No new `__pycache__` was created.

The snapshot files were kept in the session scratchpad, not in the project. The snapshot script is the Stage 2A integrity snapshot script, extended to cover the research_v1 folders.

## 5. Limitations and deferred issues

1. The config schema covers foundation checks only. Training, preprocessing and XAI settings will be added when those stages are specified.
2. `CUBLAS_WORKSPACE_CONFIG` is recorded but not set. Deterministic cuBLAS behaviour on the GPU is therefore not configured yet, and it matters only once GPU work starts.
3. The `stable_seed` payload version (`/1`) is part of every derived seed. Changing the encoding would change every seed.

## 6. Review notes carried forward from Stage 2A

Stage 2A was not modified.

1. **HAM mask check:** in Stage 2A, `ham10000_rows_with_mask` records PASS unconditionally (its observed value is 10015). **Stage 2B2 must assert complete mask coverage and verify that the masks actually load.**
2. **Identifiers:** the Stage 2A report describes `image_uid` as path-based everywhere, but that is not how it was actually constructed:
   - HAM uses `ham10000:<image_id>`;
   - PAD uses `pad_ufes_20:<file basename>`;
   - DDI uses `ddi:<file basename>`;
   - only Derm7pt uses `derm7pt:<relative path>`.

   All of these are unique within their dataset. These identifiers must be kept as they are, and their construction must be documented before any downstream use. Uniqueness across datasets comes from the dataset prefix.
3. **Two OpenCV distributions are installed**: `opencv-python` 4.11.0.86 and `opencv-python-headless` 5.0.0.93, and the loaded `cv2` reports 4.11.0. This must be resolved before any preprocessing work. It was not repaired here.
4. **RQ5 adapt_val has 9 melanoma images**, so the protocol's Secondary B threshold endpoint remains not estimable.
5. **Primary model policy is open:** it is not yet decided whether the seed-0 primary model is the legacy checkpoint or a newly trained seed 0. This must be decided explicitly before inference or training. Nothing in this stage chose it.

**Stage 2B1 is complete. Stage 2B2 was not started.**
