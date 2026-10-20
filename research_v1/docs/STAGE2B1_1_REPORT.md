# Stage 2B1.1 Report — Receipt Integrity Verification Fix

Date: 2026-09-27

## Status: **PASS**

- All 23 tests pass.
- The new foundation-check receipt verifies with no problems.
- Protected inputs are unchanged. The only new files are the ones this task was expected to add.
- Stage 2B2 has not been started.

## 1. Exact changes

| File | Change |
|---|---|
| `research_v1/src/rv1/provenance.py` | See details below the table. |
| `research_v1/tests/test_foundations.py` | Added `write_sidecar` fixture helper. The temporary fixtures now carry valid input sidecars. The unclosed `open(p).read()` in test6 was replaced with a context manager. New class `T10_SidecarsAndConfigConsistency` (7 new tests; it also re-runs the 3 inherited receipt tests with the new fixtures). |
| `research_v1/src/rv1/seeds.py` | **Line endings only, CRLF → LF.** No content change (see note below). |
| `research_v1/configs/foundation_check_v1_1.json` (new) | Same as `foundation_check_v1.json`, except that the `card` entry is removed (see §5). The old config is kept. |
| `research_v1/runs/foundation_checks/20260927T105107Z_cd39cf9c_0ae2644b/` (new) | New receipt and its sidecar. |
| `research_v1/runs/stage2b1_1/` (new) | Integrity snapshot script, before/after snapshots and comparisons. |
| `research_v1/docs/STAGE2B1_1_REPORT.md` (new) | This report. |

### Changes in `provenance.py`

**New helpers**
- New `ProvenanceError` exception.
- New `read_sidecar(path)`:
  - raises a clear error if the sidecar is missing or empty;
  - requires the digest to be exactly 64 lower-case hex characters;
  - if an optional file-name field is present, it must match the file's basename;
  - anything else is reported as malformed. The previous `IndexError` path is gone.
- New `checked_input(path)`: hashes the file bytes and requires the sidecar digest to match them.

**`build_receipt`**
- Every manifest and the environment inventory now go through `checked_input`.
- A missing or mismatching sidecar raises an error before any receipt is written.

**`verify_receipt` now checks all of the following:**
- **The receipt's own sidecar** is validated.
- **Config consistency:**
  - the embedded config's canonical hash must equal `config_sha256`;
  - the config file must match its recorded hash;
  - the config file is re-loaded and validated, and its canonical form (with `project_root` excluded, as documented) must equal the embedded config;
  - `run_kind` and `purpose` must agree between the receipt, the embedded config and the config file.
- **Inputs that have sidecars.** For each one:
  - the actual file hash must equal the recorded hash;
  - the recorded sidecar value must equal the recorded hash;
  - the current sidecar must be valid and must equal the actual file hash.
- **Sources** are checked against their recorded hashes only. Sources have no sidecars, as specified.

**Unchanged:** the canonicalisation rules, the `stable_seed` encoding and all manifest identifiers.

### Note on `seeds.py`

`seeds.py` was changed only in its line endings. An earlier Stage 2B1 edit had written it with Windows CRLF line endings. This time I normalised it to LF while repairing a newline-escaping mistake I made in the test edits.

I verified that nothing else changed: converting the current file back to CRLF reproduces exactly the hash recorded in the old receipt (`9134d0f1…`).

## 2. Tests

```bash
set PYTHONDONTWRITEBYTECODE=1
python -B -m unittest discover -s research_v1/tests -p "test_*.py" -v
```

**Result: `Ran 23 tests in 5.100s — OK`.** The 13 original tests are included and still pass.

New tests (temporary fixtures only). Where a test targets internal consistency, it first rewrites the outer receipt sidecar so that the sidecar is valid, so the targeted check is the one that fails:

| Requirement | Test | Result |
|---|---|---|
| Valid input sidecars are accepted | `test_valid_sidecars_accepted` | ok |
| Missing, empty, malformed, all-zeros, wrong file name and upper-case sidecars are refused when the receipt is built; an empty env-inventory sidecar is also refused | `test_bad_input_sidecars_refused_at_build` | ok |
| An input sidecar changed after the receipt was created (to zeros, empty, malformed, or deleted) is detected; restoring it makes verification clean again | `test_sidecar_changed_after_receipt` | ok |
| A recorded sidecar digest that is inconsistent with the recorded hash is detected | `test_recorded_sidecar_inconsistent` | ok |
| An incorrect embedded config hash is detected | `test_wrong_config_hash` | ok |
| An embedded config that differs from the config file is detected, even when its own hash is internally correct | `test_embedded_config_differs_from_file` | ok |
| A `run_kind` or `purpose` that disagrees with the embedded config and with the config file is detected | `test_run_kind_purpose_inconsistent` | ok |

## 3. Receipt verification

| Receipt | Result |
|---|---|
| **New:** `20260927T105107Z_cd39cf9c_0ae2644b` (config `foundation_check_v1_1.json`) | `verify: OK` (no problems) |
| **Old:** `20260927T104607Z_6d95d9d3_8e91758f` (kept unchanged) | Reports problems, all of them expected: `manifest card` has no sidecar, so its recorded sidecar value (null) differs from its hash; `source: hash mismatch` for `provenance.py` (legitimately updated in this task) and for `seeds.py` (line endings only, see §1) |

The old receipt was **not rewritten**. Its differences are real and are explained above. It remains an accurate record of the code state at the time of Stage 2B1.

## 4. Integrity of protected inputs

The evidence is kept in `research_v1/runs/stage2b1_1/`:
- `integrity_before.json`
- `integrity_after_final.json`
- `integrity_comparison_final.json`
- `integrity_snapshot.py`

There is also an intermediate snapshot, `integrity_after.json`, with its comparison `integrity_comparison.json`. It was taken before the new config and receipt were created, and showed 0 changes.

**Files covered** (161 hashed before, 164 after):
- `code/`, `results/`, `checkpoints/`, `audit/manifests/`, `protocol/`, `logs/`
- `research_v1/docs/`, `research_v1/manifests/`, `research_v1/env/`, `research_v1/tools/`, `research_v1/configs/`, `research_v1/runs/foundation_checks/`
- `audit/*.*`, `cache/*_meta.json`, the top-level `*.md` and `*.docx` files, and the 4 dataset metadata files
- cache `.npy` files: size and modification time only

**Result:**
- **0 changed, 0 removed.**
- `.npy` size and modification time are identical.
- **Added**, all expected: `research_v1/configs/foundation_check_v1_1.json`, plus the new receipt and its sidecar.
- `research_v1/src/` and `research_v1/tests/` are deliberately outside this snapshot, because this task modifies them.
- The before snapshot was taken before this report existed, so the report is not in it.
- No new `__pycache__` directories were created. The ones under `code/` predate this task.

## 5. Unresolved issues

1. **`manifest_card_v1.json` has no sidecar.** Stage 2A did not create one, and the card also contains a non-deterministic elapsed-time field. The frozen manifest folder was not modified. The card is therefore left out of sidecar-backed receipts (config v1_1).
   - If the card must be tracked, one option is to record it as a source-style hash, which needs no sidecar.
   - Another option is for the researcher to authorise a sidecar in a later, reviewed step.
2. All Stage 2B1 carry-forward items remain open:
   - HAM mask coverage and loading;
   - documentation of how identifiers are constructed;
   - the two OpenCV distributions;
   - the Secondary B endpoint for adapt_val, which is not estimable;
   - the policy for the primary model at seed 0.

**Stage 2B1.1 is complete.**
