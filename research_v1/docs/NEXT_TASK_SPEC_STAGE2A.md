# Next Task Specification — Stage 2A: Environment Inventory and Cohort Manifests

Status: **specification only. This task has not been executed.** It starts only when the researcher authorises it.

## Scope

**In scope**
1. **T-A1:** a read-only inventory of the environment.
2. **T-A2:** preparation of the v1 dataset and cohort manifests, with only the validation needed to create them correctly.

**Out of scope**
- Training loops, perturbation code, the statistics library, inference or model loading, and XAI.
- Changes to legacy files.
- Installing packages.

## Allowed writes

Only these new paths may be written:
- `research_v1/env/`
- `research_v1/manifests/`
- `research_v1/tools/`: at most two small scripts, `inventory_env.py` and `build_manifests_v1.py`. Both read-only with respect to legacy data.
- `research_v1/docs/STAGE2A_REPORT.md`

Two further rules:
- No existing file is modified.
- If an output path already exists, a versioned name is used instead.

## T-A1 Environment inventory

**Actions**
- Record the Python version and platform.
- For each of torch, torchvision, timm, albumentations, opencv-python(-headless), numpy, pandas, scipy, scikit-learn, matplotlib, pyarrow, captum and streamlit, record `find_spec` and `__version__`.
- Record `torch.cuda.is_available()`, the device name, total memory, and the CUDA and cuDNN versions. The GPU is queried only; no tensors are computed on it.
- Record the value of the environment variable `PYTHONHASHSEED`.
- Record whether `albumentations.Compose` has `seed` / `set_random_seed`. This checks that the attributes exist; it does not test their behaviour.

**Output**
- `research_v1/env/env_inventory_v1.json`, including its own SHA-256.

**Acceptance**
- Every library listed in 03_rev2 §16 is recorded as present (with version) or absent.
- Nothing was installed.

## T-A2 Manifests v1

**Inputs (read-only)**
- `DATASETS/*` metadata files: HAM `HAM10000_metadata.tab`, Derm7pt `meta/meta.csv`, PAD `metadata.csv`, DDI `ddi_metadata.csv`.
- `code/labels.py`, imported read-only for the label mapping.
- `code/data.py`, imported read-only only to reproduce the legacy HAM split assignment. Its `_stable_hash` function is MD5-based and deterministic.
- `audit/manifests/*_v1.csv`, used for cross-checking only.
- `cache/*_meta.json`, used for the cache-order cross-check only. The `.npy` pixel arrays are not loaded.

**Outputs** (each with a `.sha256` file)

| File | Content |
|---|---|
| `ham10000_v1.csv` | All 10,015 images with `split` (train/val/test), `split_seed=0`, `cluster_id=lesion_id`, `mask_path`, `cache_name`, `cache_index` |
| `derm7pt_v1.csv` | 2022 image rows (1011 pairs × 2) with `pair_id`, `same_file_pair`, `cohort_flags` ∈ {`derm7pt_all`, `derm7pt_primary`}, `legacy_row_index` |
| `pad_ufes_20_v1.csv` | All 2298 images with `cluster_id=patient_id`, `stratum_rq5`, `rq5_role`, `skin_tone` (Fitzpatrick or empty) |
| `ddi_v1.csv` | All 656 images with `skin_tone` group |
| `xai_derm7pt_v1.csv` | 249 melanoma pairs + 249 non-melanoma pairs, drawn from the primary cohort with fixed `sample_seed=20260927` (stable-hash ordering within each class, then the first k) |
| `manifest_card_v1.json` | Counts per dataset × split/role × label × stratum, plus the list of validation checks and their results |

**Columns (all image manifests)**

`dataset, image_uid, rel_path, sha256, label_mel, dx_raw, modality, cluster_id, pair_id, same_file_pair, split, split_seed, rq5_role, stratum_rq5, stratum_rule_version, mask_path, skin_tone, cache_name, cache_index, legacy_row_index, cohort_flags`

Rules for column values:
- Use empty values where a field does not apply.
- Do not write age, sex, anatomical site, or any other participant attribute.

**RQ5 partition rule** (as in 02_rev2 §7.1)
1. Assign each patient to a stratum by priority: S1 if the patient has any MEL image; otherwise S2 if the patient has any BCC, SCC or ACK image; otherwise S3.
2. Within each stratum, sort patients by `sha256(f"0|{patient_id}")`.
3. Assign the first ⌈0.50·n⌉ patients to `eval`, the next ⌈0.35·n⌉ to `adapt_train`, and the rest to `adapt_val`.

**Minimum validation.** Every check below must pass, or the task stops and reports.

1. Every referenced image file exists. File SHA-256 values are computed by streaming reads; no image is decoded.
2. Label mapping covers every row. `labels.py` raises on unknown labels, and any raise stops the task.
3. **Count assertions** against Stage 1 facts:
   - HAM: 10,015 images and 1113 melanomas; split 6938 / 1529 / 1548 images; melanomas 783 / 160 / 170.
   - Derm7pt: 1011 pairs and 252 melanomas; 9 same-file pairs (3 melanomas); primary cohort 1002 / 249.
   - PAD: 2298 images, 52 melanoma images, 36 melanoma patients, 1373 patients, 179 patients with multiple diagnoses.
   - DDI: 656 images; 21 melanomas, 7 per tone group.
4. **Disjointness:**
   - no `lesion_id` appears in more than one HAM split;
   - no `patient_id` appears in more than one PAD RQ5 role;
   - every Derm7pt pair has exactly 2 rows with the same label.
5. **Legacy consistency:**
   - the HAM split assignment is identical to the legacy loader output for seed 0;
   - `legacy_row_index` matches the legacy loader order;
   - the manifest order and labels match the `cache/*_meta.json` groups and labels for each cache;
   - the file sets and labels agree with `audit/manifests/*_v1.csv`.
6. **XAI manifest:** every pair belongs to `derm7pt_primary`; exactly 249 + 249 pairs.
7. **RQ5:** each role contains at least 1 S1 patient; the counts per role × stratum are reported.
8. **Determinism:** building the manifests twice in separate processes gives byte-identical CSV files.

**Stop and failure handling**
- Any count mismatch, overlap, missing file or label error: stop, write `STAGE2A_REPORT.md` with the discrepancy, and do not "fix" data silently.
- If a Stage 1 count turns out to be wrong, the manifest (ground truth from raw metadata) is reported and the Stage 1 documents are flagged for correction. They are not edited during Stage 2A.

**Acceptance**
- All outputs exist, with SHA-256 files.
- All 8 validation groups pass, or failures are fully reported.
- `STAGE2A_REPORT.md` lists the commands run, the elapsed time, and confirms that no legacy file was modified.
- To confirm the last point, SHA-256 values are recorded before and after the task for `code/`, `results/`, `checkpoints/`, `cache/*_meta.json` and `audit/manifests/`.

**Estimated active effort:** 7.5 h (estimate).
**Compute:** CPU only. Hashing about 15,000 image files takes minutes (not measured).

**Reviewable deliverable:** the manifest files, `manifest_card_v1.json` and `STAGE2A_REPORT.md`.
