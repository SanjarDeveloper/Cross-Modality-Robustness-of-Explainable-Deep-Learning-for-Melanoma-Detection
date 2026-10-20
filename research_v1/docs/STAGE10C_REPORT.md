# Stage 10C Report — Production Simulated-HAM XAI Arm (IG m = 2048): Computation and Audit

Date: 2026-10-10. Environment: `research_v1/.venv_model`.

**Run:** `runs/xai_sim_prod_10c/20261009T073606Z_d9545ce7_386f4032/` (2.7 GB).

**Stage records:** `runs/stage10c/`.

**Scope:** computation plus a completeness and integrity audit only. **No inferential statistics and no E4.6.**

## Status

| Item | Result |
|---|---|
| Contract (`spec.json`, frozen before computation) | **Inherited unchanged from the Stage 10B spec:** cohort (340 HAM TEST images: 170 mel in 91 lesions, 170 non-mel in 165 lesions), 11 conditions, simulator, preprocessing, `xai_precision_v1`, blurred baseline, controls, localisation definitions, eligibility rules, tolerances and curve settings. **IG m = 2048**, selected by the retrospective Stage 10B.1 extension; its gate receipts verify |
| Batching (frozen) | 88 chunks: per condition (condition-major), blocks of 48 in frozen record order, with a final partial block of 4 per condition. Batch composition is recorded; batch-size invariance is not assumed, and nothing is compared bitwise with the 20-reference pilot |
| Failure policy (decided before results) | The protocol (02_rev2 §6.6) defines per-image failure and requires that "the failure rate is reported in every run". **It has no production stop rule.** The pilot's 5% readiness criterion therefore does not become one. Failed IG maps are kept with their reasons and excluded from primary analysis. They are never repaired, m is never raised per image, and no image is replaced. Hard stops are reserved for integrity violations |
| Mismatch assignment | Frozen and hashed in the spec before any scoring. **0 unassigned** (donor from another lesion, same label and condition) |
| Resume test | Passed (§2) |
| Phase exits | compute 0, mismatch 0, audit 0. The audit receipt verifies |
| Coverage | **3,740 / 3,740 references** (340 images × 11 conditions), exact identifier set and unique. Mismatch rows 3,740 / 3,740 |
| Severity-0 identity | 340/340 images: recomputed in the Stage 4A layout, **max \|Δp\| = 0.0** (tolerance 1e-6). The chunk-layout difference (2.3e-6, batch 48) is recorded as descriptive only |
| Endpoints | **0** explainer endpoint failures; 0 random/centre/Sobel failures; 0 mismatch curve failures |
| Tests | New `tests/test_xai_sim_prod.py`, 5/5: chunk naming, joint eligibility, and publisher resume safety (`.tmp` dirs, missing manifest, altered file and foreign spec are all rejected; quarantine works). `test_xai_sim`, `test_xai_sim_ext` and `test_xai_prod` pass |
| Integrity | 4240 → 4241 files, **0 changed and 0 removed**. Added: the config. Cache unchanged |

## 1. Stored per reference

**Map arrays (`maps.npz` per chunk)**
- Grad-CAM map;
- IG signed attribution (float32) and IG primary map;
- the transformed mask.

**Records (`records.json` per chunk)**
- predictions (logit, prob), completeness, eligibility and reasons;
- primary curves, plus as-computed curves for finite non-primary maps;
- 5 random, centre and Sobel curves;
- localisation for every map;
- perturbation metadata and the baseline hash.

**Mismatch (`mismatch_XXX/`):** curves and localisation for the planned donors. The baseline hash is re-checked against the compute phase for every reference.

## 2. Interruption and resume

**Planned test** (`runs/stage10c/run_resume_test.bat`)
1. Compute was started with `--max-chunks 3`.
2. After chunks 000–001 were published, the process was force-killed during chunk 002 (exit code 1). No `chunk_002` and no partial directory existed.
3. The production launcher then logged `skip verified chunk_000/001`, recomputed chunk 002, and the manifests of chunks 000 and 001 were **byte-identical** to before (`resume_test_manifests_before.txt`).

**Disclosure.** `taskkill /IM python.exe` also ended two small Python processes (the venv launcher wrapper and its parent). No other GPU work was running.

**User-requested pause** (`pause_log.txt`)
- The run was paused after 54 chunks: launcher first, then compute and the logger. Chunk 054, in progress, was discarded.
- Before resuming, the sources and inputs were re-verified against the spec, and all 54 chunks verified.
- The resumed run skipped the 54 chunks and recomputed chunk 054.
- Logs: `phase_compute_session1.txt` and `phase_compute.txt`.

## 3. Audit tables (run dir)

**Files:** `audit_results.json`, `eligibility_table.csv`, `ig_failure_table.csv`, `joint_eligibility_table.csv`, `control_availability_table.csv`, `matched_localisation_table.csv`.

**IG m = 2048 completeness failures** (§6.6; excluded from primary analysis)

| Group | Failures | p95 e_rel |
|---|---|---|
| Pooled | **31 / 3,740 (0.8%)**: mel 4, non-mel 27 | 0.013 |
| none 0 | 5 / 340 | 0.014 |
| blur 0.5 / 1.0 | 1 / 0 | 0.008 / 0.003 |
| illumination 0.5 / 1.0 | **7** / 0 | 0.016 / 0.018 |
| compression 0.5 / 1.0 | 2 / 2 | ≤ 0.011 |
| scale 0.5 / 1.0 | 1 / 2 | ≤ 0.007 |
| nuisance 0.5 / 1.0 | 4 / **7** | 0.018 / 0.028 |

The highest condition rate is 2.1%. This is reported as the measured rate; it is not a stop rule.

**Grad-CAM all-zero/constant maps** (no positive evidence; ineligible): **1,163 / 3,740**, of which 243 are mel and 920 non-mel. They rise with blur (209/340 at blur 1.0) and scale (164/340 at scale 1.0). As-computed curves are stored for these finite maps, for the protocol's sensitivity analysis.

**Controls**
- Random, centre and Sobel: available for 3,740 / 3,740.
- Mismatch Grad-CAM: 2,577 / 3,740. The 1,163 unavailable all have the reason "planned donor map not eligible".
- Mismatch IG: 3,709 / 3,740.
- No donor was replaced.

**Jointly eligible baseline-vs-condition images** (eligible at severity 0 and at the condition; all also have the random control)

| Condition | IG2048: images (mel / non-mel; lesions) | Grad-CAM: images (mel / non-mel) |
|---|---|---|
| blur 0.5 / 1.0 | 334 / 335 | 211 (142/69) / **128 (97/31)** |
| illumination 0.5 / 1.0 | 328 / 335 | 247 / 186 |
| compression 0.5 / 1.0 | 334 / 333 | 272 / 260 |
| scale 0.5 / 1.0 | 335 / 333 | 194 / 174 (131/43) |
| nuisance 0.5 / 1.0 | 332 / 328 | 269 / 254 |

IG populations are nearly complete (≥ 328 / 340, at least 246 lesions). **Grad-CAM populations shrink and are melanoma-heavy** under shift, which is selection by model evidence.

## 4. Localisation availability and matched populations (descriptive; not inference)

| Condition | Matched refs (mel / non-mel) | IG2048 IoU / pointing | Centre IoU / pointing | Random IoU | Sobel IoU |
|---|---|---|---|---|---|
| none 0 | 335 (170 / 165) | 0.280 / 0.91 | 0.739 / 0.99 | 0.154 | 0.450 |
| blur 1.0 | 340 | 0.257 / 0.84 | 0.738 / 0.99 | 0.155 | 0.528 |
| illumination 1.0 | 340 | 0.274 / 0.86 | 0.738 / 0.99 | 0.155 | 0.378 |
| compression 1.0 | 338 | 0.289 / 0.88 | 0.738 / 0.99 | 0.154 | 0.416 |
| scale 0.5 / 1.0 | 339 / 338 | 0.273 / 0.90; 0.256 / 0.89 | 0.431 / 0.78; **0.035 / 0.21** | 0.048 / 0.024 | 0.613 / 0.622 |
| nuisance 1.0 | 333 | 0.262 / 0.82 | 0.739 / 0.99 | 0.155 | 0.328 |

All 11 conditions, and the Grad-CAM comparisons on their smaller matched populations, are in `matched_localisation_table.csv`.

**Reading notes**
- **Centre prior.** HAM lesions are centred, so a centre prior localises very well (IoU about 0.74) everywhere except under scale, where the lesion is moved.
- **Area-matched IoU** uses the ground-truth mask area to choose how many pixels to keep. It is **not** an independent segmentation prediction.
- **Masks show lesion extent.** Under hair and glare (nuisance), the visible lesion is smaller than the mask.
- **Scale artefacts.** Under scale, the synthetic skin surround and the feathered seam lie outside the mask.

## 5. Measured cost

| Item | Measured |
|---|---|
| Compute | **45,302 s (12.6 h)** summed over 88 chunks. Includes the 2 resume-test chunks; excludes the discarded interrupted chunks |
| By component | IG 39,096 s (86%); random curves 3,229 s; explainer curves 1,130 s; centre 667 s; Sobel 650 s; as-computed curves 218 s; localisation 121 s; Grad-CAM 19 s; severity-0 reference 50 s |
| Mismatch | 1,053 s |
| Audit | 11 s |
| Per 48-image chunk | min 532 s, median 536 s, max 807 s |
| Wall-clock | Session 1: 12:56 → about 20:00 on 2026-10-09 (54 chunks, paused by request). Session 2: 22:08 on 10-09 → 03:14 on 10-10 |
| Peak CUDA | 1,705 MiB allocated / 2,420 MiB reserved |
| Disk | 2.83 GB (projection: about 3.2 GB) |
| GPU temperature | 47–82 °C |

**Slow chunks.** Chunks of up to 807 s coincided with GPU samples at reduced clock and power (about 1,600 MHz at about 42 W, against 2,700 MHz at about 103 W), mostly at moderate temperatures (54–63 °C), with no active throttle reason when queried. The cause is not established; it looks like power management, not temperature. It did not affect results.

The projection was about 12 h, and the measured total was 12.6 h plus mismatch.

## 6. What this stage does not do

- No inferential statistics: no Δ_ins/Δ_del tests, shift contrasts or bootstrap. **That is the next task.**
- No E4.6.
- No interpretation of explanation quality.
- The pilot's 20 images are included in the 340; they were seen in Stages 10B and 10B.1. HAM TEST is not an untouched cohort.

## 7. Files

**Code, test and config:** `src/rv1/xai_sim_prod.py`, `tests/test_xai_sim_prod.py`, `configs/xai_sim_prod_10c_v1.json`.

**Run dir:**
- `spec.json`;
- `chunk_000`–`chunk_087/` and `mismatch_000`–`087/`;
- `progress.jsonl`;
- `audit_results.json` and the five CSV tables;
- the receipts.

**`runs/stage10c/`:**
- launchers `run_resume_test.bat` and `run_prod.bat`;
- `status.txt`, `pause_log.txt` and `resume_test_*`;
- the phase logs from both sessions;
- `gpu_samples*.csv`;
- the integrity files.

**Stage 10C is complete. RQ4 is not complete:** the simulated-arm statistics, E4.6 and the deferred secondary-variant controls remain.
