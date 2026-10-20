# Stage 8B Report — RQ2 Prediction Collection (HAM TEST, 5 Factors × 9 Severities)

Date: 2026-10-04. All work in this stage used `research_v1/.venv_model`.

**Run:** `runs/rq2_predict_8b/20261004T080538Z_1d94708a_3474fb69/`. The spec and verify receipts verify.

**Stage records:** `runs/stage8b/`.

## Status

| Item | Result |
|---|---|
| Contract frozen before inference | `spec.json` (hashed). It records the checkpoint, threshold, Stage 4A predictions, simulator spec and module hash, manifest and cache metadata, and source hashes. Also: cohort counts, grid, seed rule, pipeline order, batch and order, tolerance, output schema, resume rule and the bundled-factor caveats |
| Model and threshold | Seed-0 ResNet-50 (`303d21e7…`). Stage 4A threshold 0.17569470405578613, **bound** to that model hash (`bind_threshold`) |
| Simulator | `perturb_msim_v1`, module SHA-256 `855cf57b…`. Unchanged; checked against the Stage 8A spec before running |
| Cohort | HAM TEST, **1548** images, **170** melanoma, **1155** lesions (`cluster_id`). Verified |
| **Baseline identity** (severity 0, each of the 5 factors) | Model inputs identical to unperturbed preprocessing. Probabilities match Stage 4A with a **maximum \|Δ\| of 0.0** (frozen tolerance 1e-6); same order and labels; predictions agree; **TP/FN/TN/FP = 157/13/911/467** ✓ |
| Runtime pilot (HAM TRAIN, 16 images) | Transform per image: blur 2.9 ms, illumination 1.2 ms, compression 1.2 ms, scale 7.7 ms, nuisance 6.2 ms. Inference 5.3 ms per image. Peak CUDA memory 293 / 318 MiB. **Estimated** full sweep 638 s; 185 GB of disk free. No metrics were computed |
| **Full sweep** | **45/45 units**, **69,660/69,660** logical rows, **69,660 unique keys**, 0 problems |
| Verification | Exact coverage and order; labels and clusters match; finite logits and probabilities in [0, 1]; pred == (p ≥ t) for every row; **all 5 severity-0 copies identical** |
| Tests | New `test_rq2_predict.py` (6 tests). `test_perturb` passes |
| Protected inputs | 0 changed, 0 removed. Caches and manifests unchanged |

**Why the full suite was not run:** no shared module changed.

## 1. Contract details

**Pipeline**
- Cache `ham_test` 256 uint8 → `perturb.apply` on the **original** cached image (factors are never chained) → `preprocess.to_model_input` → model.
- Precision `xai_precision_v1`.
- **Batch 16 in manifest order:** 97 batches with a final partial batch of 12, the same membership as Stage 4A. The batch index is stored per row.

**Grid**
- Factors: blur, illumination, compression, scale, nuisance.
- Severities 0, 0.125, …, 1.
- Seed: `stable_seed(0, image_uid, factor)`. The same realization is used across severities (Stage 8A).

**Baseline storage:** severity 0 is computed and stored **explicitly for every factor** (5 copies), and each copy is checked against Stage 4A.

**Per-row fields:**
- identifiers: record_uid, image_uid, cluster_id (lesion), label;
- grid: factor, severity, composite key, perturb_seed, full parameter JSON, geometry flag;
- outputs: logit, prob, pred;
- identities: threshold and its hash, checkpoint hash, simulator version and hash, prep, precision, batch index.

**Audit sample:** the transformed 256 px images of the first 4 test records are stored in each unit (`audit_images.npz`).

**Execution safety**
- Each unit is published by atomic rename after a hashed manifest is written.
- On resume, unit hashes, inputs and sources are verified, and `.tmp` directories are quarantined.
- No interruption occurred.

**Bundled factors** (results cannot isolate their components):
- **scale** also changes position, adds a synthetic background and creates inset seams;
- **illumination** changes brightness, contrast and colour temperature together;
- **nuisance** combines glare and hair.

**Frozen settings:** the simulator, threshold, model and grid were not changed after any prediction was seen.

## 2. Measured cost

| Phase | Time |
|---|---|
| Sweep, total | **372.9 s** (6.2 min; 08:05:56Z–08:12:17Z) |
| Transformation | 209.9 s |
| Inference | 160.1 s |
| CSV writing | 1.5 s |

- Per unit: about 4.7–6.5 s, rising to 14 s for scale and nuisance at high severity.
- The pilot estimate was 638 s and the measured time 373 s; the estimate was conservative.
- Peak CUDA memory: 293 MiB allocated / 318 MiB reserved.
- Output: 60.4 MB.

Successful execution says nothing about robustness. No performance metric was computed in this stage.

## 3. Tests (`tests/test_rq2_predict.py`, 6 tests)

| Area | What is checked |
|---|---|
| Grid | 45 units; 69,660 unique composite keys; exact severity grid |
| Batch membership | 97 batches with a final batch of 12; deterministic |
| Original image | Each factor starts from the unchanged original: the input arrays are unmodified, and blur after illumination equals blur alone (no chaining) |
| Baseline join | Order differences, label mismatches, probability differences above tolerance, and a wrong threshold all raise errors |
| Threshold binding | Rejects another model's hash and a not-estimable threshold |
| Partial output | A `.tmp` unit is never treated as complete and is quarantined |

## 4. Files

**New code, tests and config:** `src/rv1/rq2_predict.py`, `tests/test_rq2_predict.py`, `configs/rq2_predict_8b_v1.json`.

**Run outputs:**
- `spec.json` with its sidecar;
- `pilot.json`;
- `u_<factor>__<severity>/` (45 directories), each with `predictions.csv`, `audit_images.npz` and `chunk_manifest.json`;
- `progress.jsonl`;
- `verification.json`;
- the receipts.

**Integrity:** 1136 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged. No `__pycache__` was created.

## 5. Next statistical stage (not started)

From these predictions:
1. **Paired changes from baseline** in sensitivity and AUROC, per factor and severity, using a **lesion-cluster bootstrap** with the same lesions drawn for baseline and perturbed levels.
2. **Worst-case loss** per factor, recomputed **within every bootstrap replicate**.
3. **Crossing marker:** the first severity at which sensitivity falls ≥ 10 pp below baseline, found by linear interpolation and recomputed per replicate. Report the proportion of replicates with no crossing and the **censored quantiles** (">1.0").
4. **Descriptive factor ranking:** the proportion of replicates in which each factor has the largest worst-case loss.

No ANOVA and no multiplicity adjustment (02_rev2 §4).

After that: the simulated XAI arm and RQ3.

**Stage 8B is complete.**
