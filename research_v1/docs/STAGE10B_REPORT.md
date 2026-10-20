# Stage 10B Report — Simulated-HAM XAI Arm: Contract, Localisation Validation and Bounded Pilot

Date: 2026-10-04. Environment: `research_v1/.venv_model`.

**Model and scope:** the standard ResNet-50 seed 0 (`303d21e7…`). Pilot only. The full simulated arm, E4.6, training, threshold changes and simulator changes were **not** done.

**Run (valid, attempt 2):** `runs/xai_sim_10b/20261004T180422Z_ad7eef76_cdce001b/` (183 MB).

**Attempt 1** (kept, no chunks published): `runs/xai_sim_10b/20261004T180003Z_ad7eef76_97960c02/`.

**Stage records:** `runs/stage10b/`.

## Headline

| Question | Answer |
|---|---|
| Masks aligned? | **Yes, for the tested geometry.** Photometric factors keep the original mask. Scale moves and shrinks the mask exactly with the inset: centroid within 1 px, area ≈ 1/z² (tests). The alignment sheet was inspected visually (§2) |
| IG m = 512 adequate? | **NOT_READY under the frozen rule.** Pooled completeness failure was **16/220 = 7.3%** (> 5%), and **5 of 11 conditions** failed criterion (a). The other criteria passed everywhere: p95 e_rel ≤ 0.19, median \|ΔAUC\| ≤ 0.0025 |
| Full arm launched? | **No.** A next-step proposal is in §6 |

## 1. Frozen full-arm contract (`spec.json`)

**Source and status of the specification**
- The protocol (02_rev2 §6.7) prescribes **HAM 170 melanoma + 170 non-melanoma images, severities {0, 0.5, 1}**, and localisation with masks transformed with the image.
- The rev1 §11 mask rules apply.
- Everything else below was frozen in this stage before any map was computed, and is labelled **retrospective**.

**Subset**
- All 170 HAM TEST melanomas (91 lesions).
- 170 non-melanomas: the first 170 in `stable_seed('rq4_sim_nonmel_v1', record_uid)` order (165 lesions).
- Selection used identifiers only: no predictions, maps or masks.
- Repeated images of the same lesion are retained, with their cluster IDs.
- All 340 masks are usable at s = 0: 0 empty, 0 all-foreground (Addendum A3).

**Conditions:** 11 — one shared severity-0 baseline (the simulator's identity, as in the Stage 8B baseline unit), plus 5 RQ2 factors × {0.5, 1.0}. Factors are never chained.

**Planned workload:** 3,740 references. Chunk plan: per condition, blocks of 48 images.

**Methods (existing definitions)**
- Grad-CAM (layer4) and IG (Riemann midpoint, internal batch 16).
- `xai_precision_v1`; blurred baseline per chunk.
- Curves: S = 32, eval_batch 11, endpoint tolerance 1e-4.
- Controls: `controls_v1` (5 random, centre, Sobel) plus mismatch. Mismatch uses a fixed derangement within (label, condition, explainer), with the donor taken from **another lesion** and only the planned donor ever used.

**Planned analysis:** per condition, raw Δ against random, changes from s = 0, and IoU/pointing. Uncertainty by lesion-cluster bootstrap.

**Cohort note:** HAM TEST was evaluated in Stages 4A, 8B, 9C and 9G. **It is not an untouched confirmatory cohort.** The subset is class-enriched.

**Identity checks**
- Images transformed with and without a mask are identical: 220/220.
- Model inputs equal the Stage 8B path.
- Stage 8B audit digests (4 records × 5 factors × severities 0, 0.5 and 1): **60/60 match**.

## 2. Localisation definitions and tests

**Existing documents** give only the rev1 §11 mask rules and the toy-model unit test (`CAM > 0.5·max`). That test **is not used as a scientific threshold.**

**Frozen conventions** (224 × 224 model-input grid; mask = transformed 256-grid mask resized with INTER_NEAREST_EXACT)
- **IoU (primary), area-matched:** the top-|M| pixels in the curve ranking (descending value; ties broken by flat index), compared with the mask M.
- **IoU (secondary):** the map ≥ 0.5·max, with ties included. Requires max > 0.
- **Pointing:** the first ranked pixel (ties go to the lowest flat index). The tie count and "any tied maximum in the mask" are also recorded.

**Unavailable, never scored as zero**
- an empty or all-foreground mask;
- a non-finite, all-zero or constant map;
- an IG completeness failure.

**Mask handling**
- Nuisance masks represent lesion **extent**; hair and glare occlusion is not removed.
- Scale masks have a zero surround. The synthetic skin surround and the feathered seam are artefacts outside the mask.

**Tests:** `tests/test_xai_sim.py`, 11/11, covering known maps and masks, area-matched vs half-max IoU, ties, constant/zero/NaN maps, empty and full masks, photometric masks unchanged, scale alignment, selection, conditions, the readiness rule and mismatch donors.

**Disclosure about one test.** My first scale-alignment test used a > 200 brightness cutoff and failed (0.889 vs 0.9). The diagnosis showed exact alignment: centroids within 0.04 px, and the half-intensity set equals the mask (790/792). The cutoff was replaced with the plan's own criterion (03_rev2 §13: centroid within 1 px) plus half-intensity overlap ≥ 0.95. This changed only the test; no output tolerance changed.

**Visual check (`alignment_sheet.png`, inspected).** Four pilot images (2 mel, 2 non-mel) under none, blur 1, nuisance 1, scale 0.5 and scale 1:
- contours follow the lesions in every condition;
- glare and hair cover parts of the lesion inside the mask;
- ISIC_0030924's lesion extends past the original image edge, so its scaled mask has a straight inset border;
- the original masks are coarse polygons.

## 3. Pilot design and execution

**Pilot sample**
- 10 mel + 10 non-mel images, each from a unique lesion.
- Selected by identifiers only, in `stable_seed('rq4_sim_pilot_v1', record_uid)` order.
- No replacement of difficult or failed examples.

**Workload:** 11 conditions → **220 references**, one chunk per condition (20 images, frozen order).

**Computed per reference**
- Grad-CAM, IG m = 512 **and** m = 1024, with curves for all three.
- 5 random maps, centre and Sobel (curves and localisation).
- Mismatch for Grad-CAM and IG512 (curves and localisation).

**Execution history (fully disclosed)**
1. **Attempt 1** stopped in chunk 0. No chunk was published, and the logs are kept as `runs/stage10b/attempt1_*`.
   - The severity-0 check compared the XAI chunk's batch-of-20 probabilities with Stage 4A, which were computed in manifest-order batches of 16.
   - The maximum difference was 1.01e-6, against the frozen tolerance of 1e-6.
   - **Diagnosis** (`diag_s0_check.py` / `.json`): the inputs are identical. Recomputed in the 4A batch composition, all 20 differences are **0.0**. The discrepancy comes only from GPU batch composition, which Stage 2D1 documented as not bitwise batch-invariant.
   - **Fix:** the check now runs in the 4A composition, with the tolerance unchanged. The batch-20 difference is recorded as descriptive.
   - The source change required a new spec. It is identical to attempt 1 except for the `xai_sim.py` hash (verified).
2. **Attempt 2:** compute, mismatch and analyze all exited 0.
   - Severity-0 check: **20/20, max \|Δp\| = 0.0** against Stage 4A.
   - **0 endpoint failures.** Receipts verify.

## 4. IG numerical pilot (`ig_convergence_table.csv`, frozen criteria from `xai_pairs`)

**Criteria (frozen):** (a) completeness failure rate ≤ 5%; (b) p95 e_rel ≤ 0.20; (c) median \|AUC(512) − AUC(1024)\| < 0.005 on jointly eligible references.

**Readiness rule (frozen):** pooled pass **and** every condition passes. m was not raised for individual images.

| Group | n | Fails m=512 (mel / non-mel) | Fails m=1024 | p95 e_rel | Jointly eligible | Median \|ΔAUC\| ins / del | Result |
|---|---|---|---|---|---|---|---|
| **Pooled** | 220 | **16** (2 / 14) = 7.3% | 8 (3.6%) | 0.092 | 200 | 0.0009 / 0.0003 | **fail (a)** |
| none 0 | 20 | 1 | 1 | 0.046 | 18 | 0.0008 / 0.0004 | pass |
| blur 0.5 / 1.0 | 20 / 20 | 1 / 0 | 0 / 0 | 0.030 / 0.021 | 19 / 20 | ≤ 0.0008 | pass / pass |
| illumination 0.5 / 1.0 | 20 / 20 | **2 / 3** | 1 / 1 | 0.106 / 0.086 | 17 / 17 | ≤ 0.0010 | **fail / fail** |
| compression 0.5 / 1.0 | 20 / 20 | **2** / 1 | 1 / 0 | 0.111 / 0.037 | 18 / 19 | ≤ 0.0013 | **fail** / pass |
| scale 0.5 / 1.0 | 20 / 20 | 0 / 0 | 0 / 0 | 0.038 / 0.013 | 20 / 20 | ≤ 0.0011 | pass / pass |
| nuisance 0.5 / 1.0 | 20 / 20 | **3 / 3** | 2 / 2 | 0.190 / 0.176 | 15 / 17 | ≤ 0.0025 | **fail / fail** |
| class mel / non-mel | 110 / 110 | 2 / **14** | 1 / 7 | 0.027 / 0.170 | 108 / 92 | ≤ 0.0011 | pass / fail |

**Interpretation**
- At m = 512, curve AUCs are numerically stable relative to m = 1024 wherever both are eligible.
- **Completeness is not.** The failures concentrate in non-melanoma images and in the illumination, compression-0.5 and nuisance conditions. Pooled failure is 7.3%, above the 5% criterion.
- With 20 references per condition, a single extra failure moves a condition across the 1/20 boundary, so condition-level results are coarse.
- m = 1024 halves the failures (3.6% pooled), but per condition it still has 2/20 in both nuisance conditions. Its adequacy has not been assessed, because that would need an m = 2048 reference.
- Completeness success alone would not establish map convergence. Map-level change is reflected only through criterion (c).

## 5. Controls, localisation (pilot, descriptive) and cost

**Localisation availability**
- **Grad-CAM** maps are often all-zero, so localisation is unavailable for them: 4/20 at s = 0, up to 10/20 at blur 1.0 and 12/20 at scale 1.0. This is selection by model evidence.
- **IG512** is unavailable only where completeness failed.

**Area-matched IoU medians at s = 0** (pointing rate in brackets)

| Map | IoU [pointing] |
|---|---|
| IG512 | 0.28 [0.84] |
| Grad-CAM | 0.47 [0.75] |
| random | 0.14 [0.28] |
| **centre prior** | **0.71 [1.00]** |
| Sobel | 0.46 [0.50] |
| mismatch IG | 0.25 [0.74] |

**The centre prior outscores both explainers.** HAM lesions are usually centred, so a central Gaussian alone localises well. Any localisation result must be read against this control.

**Under scale 1.0** the centre prior collapses (0.02), because the lesion is offset. Note that these pilot numbers are not estimates for the arm.

**Mismatch:** every pilot group had a valid derangement. Unavailability occurred only where the planned donor map was itself ineligible; no donor was replaced.

**Reporting limitation:** in `localisation_pilot_table.csv`, the mismatch `unavailable_reasons` field shows each reason once rather than its count. The correct counts are n − available.

**Measured cost (attempt 2)**
- Compute: 2,165 s (36 min); mismatch: 70 s.
- Peak CUDA: 1,644 / 2,036 MiB. GPU temperature 57–80 °C. No competing GPU work.

**Seconds per reference**

| Component | s / ref |
|---|---|
| IG m = 512 | 2.72 |
| IG m = 1024 | 5.39 |
| Explainer curves (3 maps) | 0.44 |
| Random curves | 0.86 |
| Centre curves | 0.17 |
| Sobel curves | 0.17 |
| Mismatch curves | 0.28 |
| Grad-CAM | 0.005 |
| Localisation (all maps) | 0.03 |
| Perturbation + masks | 0.004 |

All controls were timed, including mismatch.

**Projection for the full arm (3,740 references; ESTIMATE)**

| Option | Time |
|---|---|
| m = 512 only | about 4.9 h |
| m = 1024 only | about 7.6 h |
| Both m | about 10.5 h |

Disk: about 3.2 GB. Assumes the same GPU and no competing load. The explainer-curve term is kept at its 3-map value, so the estimate is conservative.

## 6. Decision and next-step proposal (not executed)

**The full arm is NOT launched.** m = 512 does not meet the frozen readiness rule.

**Options that would need your decision**

1. **Proposal A (recommended): bounded m = 1024 vs m = 2048 pilot.**
   - Same 220 references and the same frozen rule.
   - About 1 h, based on 5.4 + 10.8 s per reference.
   - If m = 1024 qualifies, the full arm at m = 1024 is about 7.6 h.
   - Risk: nuisance may still exceed 1/20 at the condition level.
2. **Proposal B: keep m = 512 as in Derm7pt.**
   - Derm7pt proceeded at 13.9% failure, with disclosure; §6.4 treats failures as invalid maps.
   - Run the full arm with IG failures excluded and reported per condition and class.
   - This would require explicitly amending the readiness rule frozen here. It must not be done silently.
3. Either way, the analysis must report the **centre-prior localisation** and the **Grad-CAM all-zero rates** side by side with explainer results.

## 7. Other items

**`test_xai_secondary.py` fix:** the `records.json` handle is now opened with a context manager; there are no numerical changes. The prior copy is `runs/stage10b/test_xai_secondary_before_10b.py.txt`. The test passes with ResourceWarning raised as an error.

**Tests:** `test_xai_sim` 11/11. Also pass: `test_perturb`, `test_masks`, `test_xai_prod`, `test_xai_curves`, `test_xai_attr`, `test_xai_controls`, `test_xai_pairs`, `test_xai_secondary`. The full suite was not rerun, because no shared module changed.

**Integrity (`runs/stage10b/integrity_comparison.json`)**
- The before snapshot was taken at stage start, after the test-file fix (`tests/` is outside its scope).
- 4057 → 4058 files: 0 changed, 0 removed, 1 added (the config). Cache unchanged.

## 8. Files

**Code, test and config:** `src/rv1/xai_sim.py`, `tests/test_xai_sim.py`, `configs/xai_sim_10b_v1.json`.

**Run dir (attempt 2):**
- `spec.json` (full-arm and pilot manifests, chunk plans, mismatch assignment, definitions, rule);
- `alignment_sheet.png`;
- `chunk_00`–`chunk_10/`, each with `maps.npz` (masks, Grad-CAM, IG512/1024 maps, IG512 signed attribution) and `records.json`;
- `mismatch_00`–`10/`;
- `results.json`;
- `ig_convergence_table.csv`, `localisation_pilot_table.csv`, `per_reference_pilot.csv`;
- the receipts.

**`runs/stage10b/`:** `run_pilot.bat`, `status.txt`, the phase logs, the `attempt1_*` logs, `diag_s0_check.*`, `gpu_samples.csv` and the integrity files.

**RQ4 is not complete.** Remaining: the simulated arm (after a decision in §6), E4.6, and the deferred control comparisons for the secondary variants.

**Stage 10B is complete. The pilot stops here.**
