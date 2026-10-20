# Stage 3B Report — Bounded Real-Model XAI Pilot (ResNet-50 seed 0, 24 Derm7pt pairs)

Date: 2026-10-01. All work in this stage used `research_v1/.venv_model`.

**Run:** `runs/xai_pilot/20261001T175921Z_9072e112_b53710a8/`. The receipt verifies.

**Stage records:** `runs/stage3b/`.

## Status

| Item | Result |
|---|---|
| Model verified before loading | Pass. File = sidecar = `best.json` = expected `303d21e7…03dd` (epoch 8); strict state-dict load |
| Pilot sample frozen before inference | 24 pairs / 48 references. Addendum D, pair and reference CSVs with sidecars, and config `xai_pilot_stage3b_v1.json` (hash `9072e112…`) |
| Tests | New `test_xai_pairs.py` (11 tests). Full suite **135/135** before the run |
| Batch-equivalence check (`eval_batch` 11 vs 1) | **Pass**: 28 maps, max \|ΔAUC\| 3.5e-7, all endpoints pass. `eval_batch` 11 used |
| Endpoint failures (all curves) | **0** |
| Constant or non-finite IG maps | **0** at every m |
| IG candidate m = 128 | **Fails** criterion (a): 15/48 completeness failures (31%) |
| IG candidate m = 256 | **Fails** criterion (a): 8/48 completeness failures (17%) |
| **Production IG m** | **UNRESOLVED**. Neither candidate meets the protocol rule |
| Controls | Computed for Grad-CAM and for **IG m = 256 as a DIAGNOSTIC ONLY** (not a production choice) |
| Wall-clock | 365.7 s in-process (6.1 min); 17:59:16Z–18:05:24Z |
| Protected inputs | 0 changed, 0 removed |

Not done: m = 1024, full XAI, RQ1 or RQ4, statistics, and any tuning.

## 1. Frozen sample and protocol conflict

**Sample:** 12 melanoma and 12 non-melanoma pairs from the frozen `xai_derm7pt_v1.csv`.
- Within each label, pairs are sorted by `(stable_seed("stage3b_xai_pilot_v1", pair_id), pair_id)` and the first 12 are taken.
- All 48 records were checked for primary-cohort membership, label, modality, `pair_id`, `image_uid` and file SHA.
- The pilot program re-derives the selection and refuses to run if it differs from the frozen CSV.
- No record was excluded.

**Conflict (Addendum D1).** 02_rev2 §6.6 planned m ∈ {16…128} on 40 pairs. This pilot uses 24 pairs and m ∈ {128, 256, 512}, by instruction. This is a documented deviation. Criteria (a)–(c) are unchanged, and their denominators were fixed in Addendum D5 before inference.

**Disclosure.** Derm7pt was inspected for the numerical XAI settings, so it is not a pristine confirmatory set for these settings. Nothing was tuned.

**Software dry run.** Before the real run, one end-to-end software check was run in the scratchpad (outside the project) with m = 2/4/8. Its outputs were not used for any decision. It exposed the same images to the same model, which is part of the same disclosure.

## 2. Batch equivalence (Addendum D6)

**Subset:** 4 pairs (case586 and case812 melanoma; case307 and case544 non-melanoma), giving 8 references.

**Maps compared:** Grad-CAM and IG m = 128/256/512.
- 28 maps were compared.
- 4 all-zero Grad-CAM maps were skipped and reported.
- Attribution batching was fixed: Grad-CAM batch 8, IG internal batch 16.

**Result:**
- Max |ΔAUC| = 3.5e-7 (tolerance 1e-3).
- All endpoints passed (tolerance 1e-4).
- **Pass**, so `eval_batch` 11 was used. The batch-1 fallback was not needed.

## 3. IG numerical checks (all 48 references)

### Per m

| m | Completeness failures | e_rel undefined | p95 e_rel | Max e_rel | All-zero / constant / non-finite maps | Endpoint failures | Primary-eligible |
|---|---|---|---|---|---|---|---|
| 128 | **15/48** (derm 2, clin 13; mel 7, non-mel 8) | 0 | 0.195 | 0.258 | 0/0/0 | 0 | 33 |
| 256 | **8/48** (derm 2, clin 6; mel 3, non-mel 5) | 0 | 0.083 | 0.099 | 0/0/0 | 0 | 40 |
| 512 | 1/48 (derm 1, clin 0) | 0 | 0.046 | 0.152 | 0/0/0 | 0 | 47 |

### Per candidate (pooled; the decision is based on these rows)

| Candidate | (a) failure rate ≤ 5% | (b) p95 e_rel ≤ 0.20 | Eligible at both m | (c) median \|ΔAUC\| insertion | (c) median \|ΔAUC\| deletion | Overall |
|---|---|---|---|---|---|---|
| 128 vs 256 | **31% ✗** | 0.195 ✓ | **31/48** | 0.0033 ✓ (max 0.021) | 0.0018 ✓ | **fail** |
| 256 vs 512 | **17% ✗** | 0.083 ✓ | **39/48** | 0.0030 ✓ (max 0.014) | 0.0009 ✓ | **fail** |

**Why (c) has fewer than 48 references.** The (c) medians use only the references usable at both m. The 9 excluded at 256 vs 512 are:
- 8 that failed completeness at 256;
- case812 dermoscopy, which failed at **512** (e_rel 0.152) but passed at 256.

Missing comparisons were **not** filled with zero.

### By modality and label (descriptive only)

**Completeness:**
- The failures are concentrated in **clinical** images: 13/24 at m = 128 and 6/24 at m = 256, against 2/24 for dermoscopy at both.
- At m = 256, all failures have e_rel between 0.052 and 0.099.

**Subgroup outcomes** (full table in `ig_convergence.json`):
- At m = 128 only melanoma × dermoscopy passes; for example, clinical p95 e_rel is 0.234.
- At m = 256 every subgroup fails (a).
- Criterion (c) passes in every subgroup at 256 vs 512.

### Decision

- **No candidate meets the pre-existing rule, so the production IG m is UNRESOLVED.**
- m = 512 looks numerically better (1/48 failures), but it **cannot be selected or called converged**, because m = 1024 was not run.
- The case812 dermoscopy result, failing at 512 after passing at 256, shows that convergence is not monotone for every image.

## 4. Grad-CAM

- 11/48 Grad-CAM maps are **all-zero** (no positive evidence after ReLU): 2 melanoma and 9 non-melanoma; 3 dermoscopy and 8 clinical.
- These are **not integration failures**. Under the existing rule they are ineligible for the primary map analysis, and they are reported, not replaced.
- No endpoint failures.

## 5. Controls and paired eligibility (IG = m 256 DIAGNOSTIC)

- **Mismatch assignments** were written before the control curves were evaluated.
  - There are 96 assignments (2 explainers × 48), all assigned with shift 1.
  - Invalid donors were kept and marked unavailable.
- **Random, centre and Sobel:** every map was valid and every curve passed its endpoints.
- **Random control:** the 5 random maps give one within-image mean.

**Complete pairs** (both modalities eligible; for a contrast, the control must also be available in both):

| Explainer | Explainer pair | Random | Centre | Sobel | Mismatch |
|---|---|---|---|---|---|
| Grad-CAM | **15/24** (mel 10, non-mel 5) | 15 | 15 | 15 | **9** (mel 9, non-mel **0**) |
| IG m = 256 (diagnostic) | **16/24** (mel 9, non-mel 7) | 16 | 16 | 16 | **11** (mel 7, non-mel 4) |

**Exclusion reasons (`pair_eligibility.csv` lists each pair):**
- Grad-CAM: all-zero maps in 9 pairs.
- IG m = 256: completeness failure in 8 pairs.
- Mismatch: in addition, a planned donor map was not eligible (Grad-CAM 7 entries, IG 8). For non-melanoma Grad-CAM, the all-zero donors leave **no** complete mismatch pair.

No modality was dropped, and no case was resampled or replaced. No scores are interpreted.

## 6. Time, memory and projection

### Measured

CUDA-synchronised time and peak memory (allocated / reserved, MiB) for 48 references:

| Operation | Time | Peak memory |
|---|---|---|
| Baseline | 0.09 s | — |
| Grad-CAM | 0.35 s | — |
| IG m = 128 | 33.6 s | — |
| IG m = 256 | 67.2 s | — |
| IG m = 512 | 134.3 s | 1750 / 2406 |
| Batch-equivalence check | 20.0 s | — |
| Explainer curves (157 maps) | 27.9 s | — |
| Control generation | 0.25 s | — |
| Control curves (413 maps) | 73.5 s | — |
| Persistence | 3.9 s | — |

- Total: 365.7 s.
- One insertion-plus-deletion pair costs **0.178 s** per map.
- The GPU was at 56 °C before the run and 63 °C after.

### Projection (not a measurement)

**Scope:**
- Assumes 996 references, the same GPU, the seed-0 ResNet-50, unperturbed Derm7pt, and `eval_batch` 11.
- Per reference: 11 curve pairs (Grad-CAM, IG, 5 random, centre, Sobel, 2 mismatch), plus attribution, control generation and persistence.

| IG m | Projected time |
|---|---|
| 128 | ≈ 0.76 h |
| 256 | ≈ 0.95 h (3.45 s per reference) |
| 512 | ≈ 1.34 h |

A future m = 1024 check is not included.

**Storage:**
- Production maps in float32, uncompressed (Grad-CAM, IG signed attribution, IG primary, Sobel): 1.2 MB per reference, **≈ 1.2 GB**.
- Curves JSON: about 25 kB per reference, **≈ 0.025 GB**.
- Random and centre maps are regenerated from their seeds, not stored.
- The pilot's compressed `maps.npz` holds all three m values and is 108 MB.

This projection does **not** cover other architectures, seeds or simulated shifts.

## 7. Outputs

**Run directory:**
- `batch_equivalence.json`, `ig_convergence.json` (rule, decision, all subgroup assessments, per-reference records);
- `mismatch_assignment.json`, `maps.npz`, `curves.json`;
- `per_image_records.csv` (maps, eligibility, completeness, endpoints, AUCs and `p_full`, which is stored but not analysed);
- `scores.csv`, `pair_eligibility.csv`, `summary.json` (includes the projection);
- `receipt.json` with its sidecar.

**`runs/stage3b/`:** integrity before, after and comparison; stdout; wall-clock timestamps; GPU snapshots.

**Code (new only):**
- `src/rv1/xai_pairs.py`, `xai_pilot_freeze.py`, `xai_pilot.py`;
- `tests/test_xai_pairs.py`.

No existing source file was modified.

**Integrity:**
- 487 files before, 493 after.
- **0 changed, 0 removed.**
- The only additions are the addendum D, the config, and the pair and reference CSVs with their sidecars.
- The cache `.npy` files are unchanged (size and modification time). No `__pycache__` was created in `src` or `tests`.

## 8. Unresolved (decisions for the user)

1. **Production IG m.** Both candidates fail completeness (a). Options, each requiring authorisation:
   - (i) run m = 1024 on the same 48 references, so that 512 can be assessed against 1024;
   - (ii) accept an explicitly documented deviation.

   The tolerances must not be changed after seeing these results.
2. **Grad-CAM all-zero maps.** These remove 9/24 pairs, and every non-melanoma Grad-CAM mismatch pair. This is an eligibility property of the model and data, and how the RQ4 population is reported needs a decision. It is not something to fix.
3. **Clinical-image IG difficulty.** Completeness failures are concentrated in clinical images. Reporting by modality must stay explicit.
4. **Sample size.** This pilot used 24 pairs, not the 40 planned (Addendum D1).

**Stage 3B is complete.** Full XAI and RQ1 were not started.
