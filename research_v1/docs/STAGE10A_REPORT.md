# Stage 10A Report — Secondary IG Map Variants on the Derm7pt XAI Cohort

Date: 2026-10-04. Environment: `research_v1/.venv_model`.

**Run:** `runs/xai_secondary_10a/20261004T171602Z_a97243d1_c22139d2/` (185 MB).

**Stage records:** `runs/stage10a/`.

**Scope:**
- Standard ResNet-50 seed 0 (`303d21e7…`) only.
- The frozen 498-pair / 996-reference cohort.
- The stored Stage 7A signed IG attributions (m = 512).
- New insertion/deletion curves.

Not done: IG re-integration, training, a change of m, perturbations or new models. The primary results and the 12-test Holm family are unchanged.

## Status

| Item | Result |
|---|---|
| Spec (`spec.json`, hashed, frozen before any curve) | Variants, smoothing conventions, reconstruction tolerance, curve settings, eligibility, controls, analysis definitions and seeds. It also binds: Stage 7A audit receipt outputs and all 21 chunk file hashes, Stage 7B receipt outputs, and source hashes |
| Variants | **A. `ig_abs`** = Σ_RGB \|attr\| (the stored `ig512_absolute`). **B. `ig_pos_smooth`** = Gaussian(σ = 2) applied **after** ReLU(Σ_RGB attr), i.e. the stored `ig512_primary`, then smoothed |
| Smoothing conventions | **Retrospective, frozen in 10A.** The earlier documents give only σ = 2 px. Kernel: sampled Gaussian, radius int(4σ + 0.5) = 8 (17 taps), normalised to 1, separable. Padding: symmetric half-sample reflection (identical to `scipy.ndimage.gaussian_filter(mode='reflect', truncate=4)`, tested to 1e-12). float64 |
| Stored-map validation | Every one of the 996 primary and absolute maps was reconstructed from the signed attribution: **max \|Δ\| = 3.0e-8 (primary) and 5.2e-8 (absolute)**, against a frozen tolerance of 1e-5 × map max. `abs(Σ RGB)` provably does **not** match the stored absolute map (tested) |
| Inputs and baseline | Same chunk membership as 7A (21 chunks). The blurred baseline was recomputed per chunk, and **all 996 SHA-256 values equal the 7A records** |
| Random control | The Stage 7A random curves were reused (same model, x, baseline, S = 32, eval_batch 11, `xai_precision_v1`). Identity check: new-curve p_full and p_base equal the 7A random-curve values, **max \|Δ\| = 0.0** (tolerance 1e-6). The same 5 AUCs per image are averaged (`control_scores`) |
| Other controls | Centre, Sobel and mismatch were **not used for the variants**: the mismatch curves belong to primary maps and are not relabelled. Deferred |
| Curves | Both variants for all 858 eligible images. **0 endpoint failures**, all finite. Each record keeps the full curves (`var_XX/records.json`) and the smoothed maps (`maps_smooth.npz`, float32 copy; the float64 SHA-256 is recorded) |
| Tests | New `tests/test_xai_secondary.py`, **9/9 pass**. Covered: sum\|·\| vs \|sum\|, smoothing order, kernel (impulse variance = σ², scipy equality, boundary mass and constants), shape, finiteness and no mutation, completeness exclusion, populations and direction, shared draws, 7A reconstruction, random-control fields. `test_xai_prod`, `test_rq4_stats`, `test_xai_curves`, `test_xai_attr` and `test_xai_controls` pass. The full suite was not rerun, because no shared module changed |
| Runtime and resources | Compute 617 s (curves 302 s), analyze 71 s. Peak CUDA 298 / 390 MiB. GPU 55–73 °C. Exit codes 0 / 0. Receipts verify |
| Integrity | 3967 → 3969 files, **0 changed and 0 removed**. Added: the config and `STAGE9G2_CLARIFICATION_NOTE_v1.md`. Cache unchanged |

## 1. Eligibility (`eligibility_table.csv`)

| Variant | Dermoscopy mel / non-mel | Clinical mel / non-mel | Exclusion reason |
|---|---|---|---|
| ig_abs | 243 / 221 | 207 / 187 | Only `ig_completeness_failure` (6 / 28 / 42 / 62). Map-invalid: 0 |
| ig_pos_smooth | 243 / 221 | 207 / 187 | Same |

**Populations**
- Own matched population: **373 pairs** for each variant (204 mel / 169 non-mel).
- Intersection with the original primary IG population: **373**.
- Variant-only and original-only pairs: **none**.
- The 138 completeness-failed images remain excluded; the transformation does not repair them.

**Reproduction check:** the original IG statistics on that population reproduce the Stage 7B family table (HL and median), **max \|Δ\| = 0.0**.

## 2. Results

**Files**
- `own_population_stats.csv`;
- `intersection_comparison.csv` (original, variant, and variant − original, per modality and for D = clinical − dermoscopy);
- `intersection_per_pair.csv`;
- `per_image_variant_scores.csv`;
- `bootstrap_replicates.npz`.

**Thesis table:** `runs/stage10a/thesis_results_rq4_secondary_ig.md` (Table S1, with `.sha256`).

**HL on the same 373 pairs** (Δ = random-referenced advantage; positive means better than random)

| | Original | Absolute | Abs − orig | Smoothed | Smooth − orig |
|---|---|---|---|---|---|
| Δ_ins dermoscopy | 0.116 | 0.080 | −0.029 [−0.041, −0.017] | 0.059 | −0.054 [−0.075, −0.035] |
| Δ_ins clinical | 0.072 | 0.018 | −0.044 | 0.007 [−0.002, 0.017] | −0.058 |
| Δ_ins D | −0.033 | −0.055 | −0.025 [−0.040, −0.011] | −0.053 | −0.018 [−0.037, 0.001] |
| Δ_del dermoscopy | 0.190 | 0.061 | −0.105 | 0.121 | −0.060 |
| Δ_del clinical | 0.079 | 0.010 | −0.065 | 0.034 | −0.039 |
| Δ_del D | −0.089 | −0.047 | +0.027 [0.015, 0.039] | −0.065 | +0.018 [0.011, 0.025] |

**Reading**
- **Better than random:** both variants were better than random for dermoscopy. For clinical images the margin was small, and for smoothed insertion it was compatible with zero.
- **Below the original:** both variants were **below** the original positive IG in every modality and curve, with paired intervals excluding zero.
- **Modality shift:**
  - D stayed negative for every map: clinical was lower than dermoscopy.
  - With the variants, the deletion shift was smaller in magnitude. The reason is the larger loss in dermoscopy, not better clinical scores; both modality values are shown above.
  - For insertion, the shift was larger in magnitude for the absolute map.
- **Not an explanation-quality ranking:**
  - The absolute map includes melanoma-opposing contributions, so its lower scores against a melanoma-probability target do not mean it explains worse.
  - Smoothing changes the spatial ranking that the curves test.
  - These are descriptive, conditional results for one model. No new tests were run.

## 3. Stage 9G2 clarification

`docs/STAGE9G2_CLARIFICATION_NOTE_v1.md` (new; the 9G2 files are unchanged) covers two points:
1. **Most consistent, not exclusive.** Benefits were most consistent for the overlapping families, but they were not exclusively confined to them. Some non-overlapping cases were favourable, for example EfficientNet-B3 s1 nuisance, +12.9 pp [+6.8, +19.1].
2. **Digest scope.** The digest comparisons directly covered only the recorded audit sample (4 images per unit). The other images rely on the shared deterministic implementation, plus the severity-0 baseline checks.

## 4. Files

**Code, test and config:** `src/rv1/xai_secondary.py`, `tests/test_xai_secondary.py`, `configs/xai_secondary_10a_v1.json`.

**Run dir:**
- `spec.json`;
- 21 `var_XX/` dirs (`records.json`, `maps_smooth.npz`, manifest);
- `results.json` (populations, measured cost, reconstruction and identity checks);
- the CSV tables and `bootstrap_replicates.npz`;
- the receipts.

**`runs/stage10a/`:** `run_phases.bat`, `status.txt`, the phase logs, `gpu_samples.csv`, the thesis text and the integrity files.

## 5. Remaining RQ4 work (not started)

- E4.6.
- The simulated HAM arm.
- Centre, Sobel and mismatch comparisons for the secondary variants (deferred).
- Optional XAI for additional seeds (separate).

**Stage 10A is complete.**
