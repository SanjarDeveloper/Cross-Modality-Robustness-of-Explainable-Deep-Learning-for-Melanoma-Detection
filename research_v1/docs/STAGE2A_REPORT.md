# Stage 2A Report: Environment Inventory and Cohort Manifests v1

Date: 2026-09-27. Specification: `NEXT_TASK_SPEC_STAGE2A.md`, together with the clarifications given in the Stage 2A authorisation.

## Overall status: **PASS**, for review

- The final build passed all 73 in-script validations.
- The two independent final builds produced byte-identical CSVs.
- The legacy integrity check found no additions, removals or changes.

PASS means that the Stage 2A validations passed. **It does not mean the manifests are approved for training.** That approval requires the researcher's review.

The first build ended **BLOCKED** because one cross-check had been over-specified. It is kept as a diagnostic run and explained in §5. The specification was then applied as written, not relaxed.

## 1. Outputs

| Path | Content |
|---|---|
| `research_v1/tools/inventory_env.py` | T-A1 script |
| `research_v1/tools/build_manifests_v1.py` | T-A2 script |
| `research_v1/env/env_inventory_v1.json` + `.sha256` | Environment inventory. The file's digest is stored only in the sidecar |
| `research_v1/env/legacy_integrity_before.json`, `…_after.json` | Legacy integrity snapshots |
| **`research_v1/manifests/v1/`** | **Final manifests (run 2):** `ham10000_v1.csv`, `derm7pt_v1.csv`, `pad_ufes_20_v1.csv`, `ddi_v1.csv`, `xai_derm7pt_v1.csv`, each with a `.sha256` file, plus `manifest_card_v1.json` |
| `research_v1/manifests/run1_BLOCKED_diagnostic/` | Diagnostic run 1 (card status BLOCKED). **Not for use** |

**Versioned-output note.** Run 1 was written directly into `research_v1/manifests/`. Its files were then moved into `run1_BLOCKED_diagnostic/`. They are my own new outputs, and no legacy files were involved. The final outputs are in the `v1/` subdirectory, which differs from the flat path given in the specification.

**Final CSV SHA-256 values:**

| File | SHA-256 |
|---|---|
| ham10000_v1.csv | 41ab32de…cb1 |
| derm7pt_v1.csv | 60eb09e8…c6b |
| pad_ufes_20_v1.csv | be3a22ee…5f5 |
| ddi_v1.csv | 7a57c42f…3ba |
| xai_derm7pt_v1.csv | e67078f2…ffa |

The full digests are in the sidecar files.

## 2. Environment (T-A1)

**Python:** `C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe`, version 3.10.11. This is the system interpreter; no virtual environment or conda is active. `PYTHONHASHSEED` is unset.

| Distribution | Import name | Version | Import |
|---|---|---|---|
| torch | torch | 2.11.0+cu128 | ok |
| torchvision | torchvision | 0.26.0+cu128 | ok |
| timm | timm | 1.0.28 | ok |
| albumentations | albumentations | 2.0.8 | ok |
| opencv-python | cv2 | 4.11.0.86 | ok (cv2 reports 4.11.0) |
| **opencv-python-headless** | cv2 | **5.0.0.93** | Both distributions are installed and share the import name `cv2`. The module that loads reports 4.11.0 |
| numpy | numpy | 2.2.6 | ok |
| pandas | pandas | 2.3.3 | ok |
| scipy | scipy | 1.15.3 | ok |
| scikit-learn | sklearn | 1.7.2 | ok |
| matplotlib | matplotlib | 3.10.9 | ok |
| pyarrow | pyarrow | 24.0.0 | ok |
| captum | captum | 0.9.0 | ok |
| streamlit | streamlit | — | **absent**. Not needed for Stage 2A |

**GPU** (queried only; nothing was computed on it): NVIDIA GeForce RTX 5060 Laptop GPU, 8150 MiB, compute capability 12.0, torch CUDA build 12.8, cuDNN 91900.

**Albumentations:** `Compose` accepts a `seed` argument and has `set_random_seed` and `set_random_state` methods. Only their existence was checked; their behaviour was not tested.

## 3. Cohort counts (observed in raw metadata)

**HAM10000** (lesion-level split, split_seed 0):

| Split | Images | Melanoma images | Lesions | Melanoma lesions |
|---|---|---|---|---|
| train | 6938 | 783 | 5196 | 435 |
| val | 1529 | 160 | 1119 | 88 |
| test | 1548 | 170 | 1155 | 91 |

**Derm7pt:**

| Cohort | Pairs | Image-reference rows | Melanoma pairs |
|---|---|---|---|
| All | 1011 | 2022 | 252 |
| Primary | 1002 | 2004 | 249 |

- There are 2013 unique files.
- The 9 same-file pairs (3 of them melanoma) keep both of their rows.

**XAI manifest:** 498 pair-level rows (249 melanoma + 249 non-melanoma). These correspond to **996 image references** in the primary cohort.

**PAD-UFES-20, RQ5 partition** (actual counts):

| Role | Patients (S1 / S2 / S3 / all) | Images | Melanoma images | Melanoma patients | Melanoma images by Fitzpatrick |
|---|---|---|---|---|---|
| eval | 18 / 522 / 147 / 687 | 1150 | 26 | 18 | 1:3, 2:19, 3:4 |
| adapt_train | 13 / 366 / 103 / 482 | 805 | 17 | 13 | 2:9, 3:6, 4:2 |
| adapt_val | 5 / 156 / 43 / 204 | 343 | 9 | 5 | 2:7, 4:2 |

- Totals: 2298 images, 1373 patients, 52 melanoma images, 36 melanoma patients, 179 patients with more than one diagnosis.
- 804 images have no skin-tone value.

**DDI:** 208 / 241 / 207 images in tone groups 12 / 34 / 56, with 7 melanomas in each group.

## 4. Validation results

All results are recorded in `manifest_card_v1.json`: 73 checks, 0 FAIL.

| Group | Checks | Result |
|---|---|---|
| 1. Files exist | 4 datasets' image files; all 10015 HAM mask files | PASS (0 missing) |
| 2. Label mapping | All rows mapped by `code/labels.py`, which raises an error on unknown labels | PASS |
| 3. Counts | 26 count assertions (§3) | PASS; every count matched the expected Stage 1 value |
| 4. Disjointness | Lesion → one split; patient → one RQ5 role; each Derm7pt pair has 2 rows with the same label; `record_uid` is unique in every manifest | PASS |
| 5. Legacy consistency | HAM split equals an independent re-implementation of the legacy MD5 rule; order, labels and groups match all 7 `cache/*_meta.json` files (HAM also by image_id); `legacy_row_index` is sequential; file sets, labels and SHA-256 agree with `audit/manifests/*_v1.csv` (used as a cross-check only) | PASS |
| 6. XAI | 498 rows, 249 melanoma pairs, all from the primary cohort, unique, every record key resolves | PASS |
| 7. RQ5 | Every role has ≥ 1 S1 patient; the role counts per stratum follow the ceiling rule | PASS |
| 8. Determinism | Run 2 (`research_v1/manifests/v1`) and run 3 (a scratch directory), each in its own process: all 5 CSVs byte-identical (`cmp`). They are also identical to the run 1 CSVs | PASS |

## 5. Run 1 (BLOCKED): diagnosis and correction

**What failed.** `derm7pt_matches_audit_manifest_paths_labels` reported 108 differing row keys (54 in each direction).

**Cause.** The check compared the pair identifier as well as the path, modality and label.

- In the legacy loader (`code/data.py:derm7pt`), the pair identifier is the `case_id` column from `meta.csv` when that column is non-empty. This is the case for **27 cases**, for example case_num 102 → `1d171819`.
- In all other cases the loader uses `case<case_num>`.
- The audit manifest used `case<case_num>` for every case.

**What was compared.** The paths, modalities and labels agree in full. The v1 `pair_id` matches the groups in the legacy caches, which was verified in group 5.

**Correction.** The check was made to test what the specification requires: file sets and labels. The difference in identifier convention is now recorded under `info` in the card. No data were changed or filtered. The CSVs are byte-identical before and after the correction.

## 6. Identity, cohort flags and stable-hash definitions

**Row and file identity**

- `record_uid` is unique for every image-reference row.
- For Derm7pt, `record_uid` = `derm7pt:<pair_id>:<modality>`.
- `image_uid` = `<dataset>:<relative path>`. File identity is therefore **path-based**, and the SHA-256 of the file content is stored separately in the `sha256` column.
- Same-file pairs therefore share an `image_uid` but keep two distinct `record_uid` values.

**Cohort membership (Derm7pt)**

- `in_derm7pt_all` = 1 for every row.
- `in_derm7pt_primary` = 1 for rows outside the same-file pairs. A primary row therefore belongs to both cohorts.
- `cohort_flags` carries the same information as text, for example `derm7pt_all;derm7pt_primary`.

**cache_index** is the position within the named cache only (`cache_name`). It is not a global index.

**Stable hash**

- `sha256(s.encode("utf-8")).hexdigest()`.
- PAD patient key: `f"0|{patient_id}"`. XAI pair key: `f"20260927|{pair_id}"`.
- Ties are broken by the ID string.
- Python's built-in `hash()` is never used.
- XAI negatives are the first 249 non-melanoma primary pairs in key order. All 249 primary melanoma pairs are included.

**PAD rule**

1. S1: any MEL diagnosis. S2: otherwise, any BCC, SCC or ACK diagnosis. S3: all others.
2. Within each stratum, patients are sorted by (key, patient_id). The first ceil(0.50·n) go to eval, the next ceil(0.35·n) to adapt_train, and the rest to adapt_val.
3. Roles are assigned to patients, so all of a patient's images share the same role.

**Privacy.** No age, sex, anatomical site or other participant attribute is written. Only the fields listed in the specification are included, plus `record_uid`, `in_derm7pt_all` and `in_derm7pt_primary`.

## 7. Legacy integrity

**Coverage.** SHA-256 was computed before and after for **128 files**:

- all files under `code/` (including the existing `__pycache__`), `results/`, `checkpoints/`, `audit/manifests/`, `protocol/`, `logs/` and `research_v1/docs/`;
- the files directly in `audit/`;
- `cache/*_meta.json`;
- the top-level `*.md` and `*.docx` files;
- the four dataset metadata files.

For the 10 files `cache/*.npy`, only size and modification time were recorded; they were not hashed. Dataset image files were hashed for the manifests but were not included in the integrity comparison.

**Result:** 0 added, 0 removed, 0 changed. The `.npy` size and modification-time values are identical. No new `.pyc` files appeared under `code/`. Local modules were imported with `python -B` and `PYTHONDONTWRITEBYTECODE=1`, after I checked them for side effects: `labels.py` contains only dictionaries and functions, and at import time `data.py` only defines constants and imports torch and cv2.

## 8. Commands and measured elapsed time

These are wall-clock times for the machine. They do not measure human effort.

| Command | Elapsed |
|---|---|
| Integrity snapshot, before | not timed (seconds) |
| `python -B research_v1/tools/inventory_env.py` | not captured (the `bc` timer was missing), about seconds |
| Build run 1 → `research_v1/manifests` (BLOCKED) | 146 s (the file hashing was the first, cold read) |
| Build run 2 → `research_v1/manifests/v1` | 15.4 s |
| Build run 3 → scratchpad (for the determinism check) | 15.4 s |
| `cmp` of the 5 CSVs across runs 1, 2 and 3 | < 1 s |
| Integrity snapshot, after, and comparison | seconds |

## 9. Unresolved issues and limitations

1. **Derm7pt `case_id` semantics.** For 27 cases the legacy grouping uses a non-empty `case_id` value, for example `1d171819`, `4A45145245`. What this field means is undocumented here. It might encode image or series numbers. Whether it links cases, for example the same patient, is **unknown**. The pairing itself is unaffected.
2. **RQ5 adapt_val has only 9 melanoma images (5 patients).** Under the protocol rule, n₊ < 20 makes Secondary B (reselecting the threshold on adapt_val) **not estimable**. This is reported as a limitation, and the split was not changed. The eval partition has 26 melanoma images from 18 patients. Adapt_train has 17 images from 13 patients.
3. **Skin tone in PAD.** There are no melanoma images in Fitzpatrick types 5–6. 804 images have no skin-tone value. Fitzpatrick values are stored as the raw strings, for example "2.0".
4. **Two OpenCV distributions** (4.11.0.86 and headless 5.0.0.93) are installed side by side. The one that is actually imported is 4.11.0. Nothing was repaired. This should be recorded before preprocessing work begins in Stage 2B.
5. **Streamlit** is absent. This is not relevant until the demo stage.
6. **The integrity check does not hash the cache `.npy` arrays or the dataset images.** It uses size and modification time for the arrays, and content hashes of the dataset images are stored in the manifests.
7. **Cross-checks against the audit manifests** confirm agreement with the earlier audit. They do not validate the data independently beyond the raw metadata.

**Stage 2A is complete.** Stage 2B has not been started.
