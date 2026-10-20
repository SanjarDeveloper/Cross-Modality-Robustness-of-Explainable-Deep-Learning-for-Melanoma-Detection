# Stage 12A Report — Secondary IG Variants against Centre, Sobel and Mismatch Controls

Date: 2026-10-10. Environment: `research_v1/.venv_model`.

**Run:** `runs/xai_secondary_ctrl_12a/20261010T062521Z_d2bedde4_bd079a5e/`.

**Stage records:** `runs/stage12a/`.

**Scope:** Derm7pt, seed-0 ResNet-50, the Stage 10A variants (`ig_abs`, `ig_pos_smooth`). No training, extra seeds or simulated-HAM work. Descriptive, with no tests; the 12-test Holm family is unchanged.

## 1. Frozen specification and verification (`spec.json`, hashed before scoring)

**Inputs bound by hash:** the Stage 7A audit outputs and all 21 chunks; the Stage 10A analysis outputs and all 21 `var_XX` dirs; the frozen 7A mismatch assignment.

**Variant fidelity**
- Stage 10A ranked pixels with **float64 maps built from the stored float32 7A arrays**: `ig_abs` = cast `ig512_absolute`; `ig_pos_smooth` = `smooth()` of the cast `ig512_primary`.
- Both maps were reconstructed with exactly that recipe. **996/996 hashes match** the 10A `map_sha256_float64` for each variant.
- The 10A `maps_smooth.npz` float32 copy does **not** match those hashes (tested), so it is not used for ranking.
- No reconstruction from the signed attributions was needed: 10A froze the stored maps as its source, and they were validated against the signed attributions there.

**Centre/Sobel reuse.** These are the 7A control curves. They depend only on the model, image, baseline, S, eval_batch and precision. Their p_full/p_base equal those of the 10A variant curves: **max |Δ| = 0.0 over 1,716 checks**.

**Mismatch**
- The frozen 7A **ig512** derangement is reused: same recipient, same planned donor, from another case with the same label and modality.
- A donor is eligible only if its 10A variant record is "available", i.e. the variant is valid and inherited IG completeness passed.
- The curve is computed from the **donor's** variant map on the **recipient's** image and baseline.
- Primary-IG mismatch curves were **not** relabelled.

## 2. Execution

**Tests:** `tests/test_xai_secondary_ctrl.py`, 4/4:
- 10A hash fidelity, and the float32 copy rejected;
- donor key equals the frozen ig512 assignment;
- the complete-pair population rule;
- contrast directions and unavailable controls.

Also pass: `test_xai_secondary`, `test_xai_prod`, `test_xai_controls`, `test_rq4_stats`.

**Bounded check first (chunk 00)**
- Recipient p_full/p_base identity is 0.0, with 0 endpoint failures.
- Every donor is from a different case.
- 3 donors per variant were unavailable (IG completeness failure).

**Full run (21 hashed, resumable chunk dirs)**
- Runtime: 281 s (274 s of curves). Exit codes: mismatch 0, analyze 0. Receipts verify.
- **Mismatch unavailable:** 138 images per variant, all "planned donor variant map not eligible (ig_completeness_failure)". None replaced.

**Consistency checks**
- **Stage 10A random-control summaries reproduced exactly** (HL, medians, CIs, n; max |Δ| = 0.0).
- **Original-IG centre / Sobel / mismatch values** equal the Stage 7B descriptive table, for example centre ins dermoscopy 0.1075 and mismatch n = 280.
- Variant − original differences are identical across the random, centre and Sobel controls, which is expected because the control cancels within each image.

## 3. Populations (`population_counts.csv`, `intersection_counts.csv`)

| Comparison | Pairs (mel / non-mel) | Image exclusions |
|---|---|---|
| Each variant × random / centre / Sobel | **373** (204 / 169) | IG completeness failure: 138 images |
| Each variant × mismatch | **280** (164 / 116) | 138 completeness + 109 recipients whose planned donor failed completeness |
| Variant vs original IG, same control | Identical sets: 373 / 373 / 373 / 280 | 0 variant-only, 0 original-only |

## 4. Results (`variant_control_stats.csv`, `variant_vs_original.csv`)

The full tables are in `runs/stage12a/thesis_results_rq4_secondary_controls.md` (Table SC1).

**Against the centre prior**

| Variant | Insertion advantage | Deletion advantage |
|---|---|---|
| Absolute IG | dermoscopy 0.076, clinical 0.032 | **dermoscopy −0.007 [−0.017, 0.003]; clinical −0.032 [−0.039, −0.026]** |
| Smoothed IG | dermoscopy 0.053, clinical 0.023 | dermoscopy 0.039; **clinical −0.009 [−0.013, −0.004]** |

**Against Sobel**
- Insertion is positive for both variants.
- Deletion: absolute IG dermoscopy 0.010 and **clinical −0.012**; smoothed IG 0.059 / 0.010.

**Against mismatch (point estimates only)**

| Variant | Insertion (dermoscopy / clinical) | Deletion (dermoscopy / clinical) |
|---|---|---|
| Absolute IG | 0.033 / **−0.007** | 0.078 / 0.027 |
| Smoothed IG | 0.062 / 0.018 | 0.062 / 0.017 |

**Shift:** D < 0 (clinical lower than dermoscopy) for every variant, control and curve.

**Variants vs the original map:** on identical pairs, both variants are below the original positive IG. Smoothed deletion: −0.060 for dermoscopy and −0.039 for clinical.

**Interpretation (descriptive)**
- The secondary maps' advantage over simple spatial priors is smaller than the primary map's. For deletion against the centre prior, it vanishes for absolute IG and is negative for clinical images.
- No variant is preferred and no test was done.
- The absolute map measures magnitude of either sign.

## 5. Integrity and files

**Integrity:** see `runs/stage12a/integrity_comparison.json`.
- **0 changed and 0 removed**; added only the config.
- File count: 4,798 → 4,799.
- The run dir (4.5 MB) and the stage records are outside the protected scope.

**Files**
- Code, test and config: `src/rv1/xai_secondary_ctrl.py`, `tests/test_xai_secondary_ctrl.py`, `configs/xai_secondary_ctrl_12a_v1.json`.
- Run dir: `spec.json`, `mm_00`–`mm_20/`, `results.json`, the 4 CSV tables and the receipts.
- `runs/stage12a/`: logs, status, the thesis text and the integrity files.

## 6. Updated RQ4 completion inventory

| Required item (02_rev2) | Status | Authoritative outputs |
|---|---|---|
| E4.1 / E4.2 primary family, Holm (§6.3) | Done | `runs/rq4_stats_7b/*/rq4_family_table.csv`, `docs/STAGE7B_REPORT.md` |
| As-computed, normalised, E4.3 (§6.4–6.5) | Done | Stages 7B / 7C (`runs/rq4_secondary_7c/`) |
| E4.4, E4.5 (§6.7) | Done | Stages 7C / 7D (`runs/rq4_explore_7d/`) |
| Secondary variants vs random (§6.1) | Done | `runs/xai_secondary_10a/20261004T171602Z_a97243d1_c22139d2/` |
| **Secondary variants vs centre / Sobel / mismatch (§6.1, controls b–d)** | **Done (this stage)** | `runs/xai_secondary_ctrl_12a/20261010T062521Z_d2bedde4_bd079a5e/` |
| Simulated HAM arm with localisation (§6.7) | Done | `runs/xai_sim_prod_10c/…`, `runs/xai_sim_stats_10d/…` (Stages 10B–10D) |
| E4.6 (§6.7) | Done | `runs/e46_design_11a/…`, `runs/e46_cv_11b/…`, `runs/stage11b1/` |

**All required RQ4 items are accounted for. RQ4 is complete.**

**Optional, not done**
- Additional-seed XAI (seeds 1–2 on a 100-pair subset, §6.1, "only if the budget allows").
- XAI or DDI for non-primary RQ3 models.

**Stage 12A is complete.**
