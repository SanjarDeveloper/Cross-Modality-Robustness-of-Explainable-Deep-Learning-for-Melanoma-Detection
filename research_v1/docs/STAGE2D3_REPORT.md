# Stage 2D3 Report — XAI Control Maps and Per-Image Control-Referenced Diagnostic Scores

Date: 2026-09-27. All work in this stage used `research_v1/.venv_model` and the `xai_precision_v1` policy.

This is a software diagnostic. It used the Stage 2C3 pilot checkpoint and the same 8 HAM TRAIN images as Stages 2D1–2D2.1 (the model was trained on these images). The IG setting m = 256 with internal batch 16 is a diagnostic choice only. **No score here is a finding.** No statistics, confidence intervals, localisation, main training or full RQ4 were run.

## Status

| Item | Result |
|---|---|
| Control spec `controls_v1` | Frozen before computation: `configs/xai_controls_v1_spec.json` |
| Guard fixes (non-finite curve points; contradictory eligibility records) | Implemented and tested |
| XAI tests | **33/33 pass** (25 earlier + 8 new in `test_xai_controls.py`) |
| Full suite | **117/117 pass** |
| Diagnostic run | Completed; receipt verifies |
| Unavailable controls / ineligible records | **None** in this sample (§4) |
| Protected inputs | 0 changed, 0 removed |

## 1. Guard fixes (`xai_curves.py`; pre-change copy in `runs/stage2d3/`)

**Non-finite values at any curve point.** If any curve point, or either endpoint prediction, is non-finite, the curve is marked `curve_status = "nonfinite"`. In that case:
- all AUC and mean values are set to `None`;
- `endpoint_status` is set to "fail";
- the number of non-finite points is recorded.

`analysis_curves` then labels the result `invalid_nonfinite`, so such a curve is **never** primary. Curves without non-finite values carry `curve_status = "ok"`.

**Eligibility at the analysis boundary.** `analysis_curves` now:
- rejects IG unless `ig_completeness_pass is True`, whatever `eligible_for_primary_analysis` claims;
- **recomputes** map validity and rejects the record if it disagrees with `map_shape_valid`;
- rejects the record if `eligible_for_primary_analysis` contradicts the recomputed validity and completeness.

Each case raises `IneligibleMap("inconsistent eligibility metadata …")` or a completeness error.

## 2. Control definitions (`controls_v1`, `src/rv1/xai_controls.py`)

All maps are float64 on the 224 grid. Pixel (i, j) has its centre at (i + 0.5, j + 0.5). No control uses labels, masks or model outputs, except that the mismatch control uses another case's attribution map, as the protocol requires.

| Control | Exact definition |
|---|---|
| **Random** (5 per image) | For k = 0…4: `numpy.random.default_rng(stable_seed("ctrl", k, image_uid)).random((224,224), dtype=float64)`. Uniform on [0, 1), PCG64, a local generator per map, and no global RNG. Each map is evaluated separately and the AUCs are **averaged within the image**. Maps are never averaged, and the seeds are not treated as independent observations |
| **Centre** | exp(−((i + .5 − 112)² + (j + .5 − 112)²) / (2σ²)) with σ = min(H,W)/4 = **56 px**. It is radially symmetric, so equal-valued pixels follow the standard tie-break (ascending flat index) |
| **Sobel** | Convert back to the pre-normalisation RGB on the 224 grid (x·STD + MEAN, not re-clipped). Take luminance Y = 0.299R + 0.587G + 0.114B (BT.601). Apply 3×3 Sobel kernels gx = [[−1,0,1],[−2,0,2],[−1,0,1]] and gy = gxᵀ by cross-correlation, with 1-pixel edge-replicate padding. The map is √(gx² + gy²). A constant or non-finite map is flagged and the control is **unavailable** |
| **Mismatch** | Groups are (label, modality, explainer, condition). Within a group, records are sorted by `stable_seed("mismatch_v1", record_uid)`. The donor is found by a cyclic shift, using the smallest s ≥ 1 for which no recipient gets its own map or a map from the same case (`cluster_id`, or `pair_id` for Derm7pt). This is a derangement. If a group has fewer than 2 records, or no valid shift exists, the control is **unavailable**. If the donor's map is ineligible, the planned mapping is kept and the control is unavailable; no other donor is chosen. The mapping is **written to disk before any metric is computed** |

**Contrasts.** Each control type is reported separately; there is no combined score. Positive means the explainer did better than that control on that metric.
- insertion advantage = AUC_ins(explainer) − AUC_ins(control);
- deletion advantage = AUC_del(control) − AUC_del(explainer).

Explainers and controls use the same image, the same baseline, S = 32, `eval_batch` = 11, and `xai_precision_v1`.

## 3. Tests (`tests/test_xai_controls.py`)

| Test | Result |
|---|---|
| Random maps are identical across processes with `PYTHONHASHSEED` 0 and 4242, and independent of access order. There are 5 distinct maps, float64 in [0, 1), and the global NumPy RNG is untouched | Pass |
| Centre map: flip and transpose symmetry, the 4 central peak pixels, and exact analytic values | Pass |
| Sobel: a known vertical step gives magnitude 4 exactly at the two edge columns and 0 elsewhere. A constant image is flagged `constant`. The input is not mutated | Pass |
| Mismatch: no self-donor, no same-case donor, same label/modality/explainer/condition, a bijection within each group, deterministic and independent of order. A singleton group and an all-same-case group are unavailable. An invalid donor keeps the planned donor and is marked unavailable | Pass |
| Scores: the within-image mean of 5 control AUCs, correct insertion and deletion signs (positive and negative cases), and unavailable handling | Pass |
| Guards: a NaN in mid-curve gives `nonfinite`, AUC None and `invalid_nonfinite`. A contradictory IG record (completeness failed but eligible=True) is rejected. Metadata claiming a valid map for a constant map, and an inconsistent eligible flag, are both rejected | Pass |

Precision policy and immutability of inputs and model are covered by the earlier guard and curve tests, which all still pass. No identity of the form E[g(X)] = g(E[X]) is used.

## 4. Diagnostic run

Output directory: `runs/xai_control_diag/20260927T154612Z_fb16a45a_7ea768a3/`.

| File | Content |
|---|---|
| `mismatch_assignment.json` | Written before any metrics |
| `maps.npz` | Inputs, baseline, Grad-CAM map, IG attributions and primary maps, 5 random maps per image, centre map, Sobel maps |
| `curves.json` | Every raw curve, including each random control's curve |
| `scores.csv` | 64 rows = 8 images × 2 explainers × 4 control types |
| `validity.json` | Validity records |
| `summary.json` | Summary |
| `review_sheet.png` | Visual check |

`verify_receipt` → OK.

**Availability**
- Grad-CAM and IG (m = 256) were **primary for all 8 images**; all IG maps passed completeness.
- All control types were **available** for all 8 images. No Sobel map was constant, and every mismatch donor was eligible.
- There were **0 endpoint failures** across all 176 curves.

**Mismatch mapping** (Grad-CAM; IG uses the same shift because the groups and order are the same):

| Label | Recipient → donor |
|---|---|
| 1 | 0033369 → 0029453 → 0024400 → 0033470 → 0033369 |
| 0 | 0025117 → 0034194 → 0030049 → 0025001 → 0025117 |

**Scores:** both signs occur for every control type across images and explainers. For example, the IG insertion advantage over the centre prior ranges from −0.21 to +0.21. These are diagnostic numbers from a pilot model on its own training images. **They are not interpreted.** No recipe was changed in response to them.

**Visual review**
- Random maps are uniform noise.
- The centre map is a central Gaussian.
- Sobel follows lesion borders and also the **dark dermoscope vignette ring**.
- Mismatch shows the donor image's map.

Two things to note for later interpretation:
- **Centred lesions** make the centre prior resemble lesion-focused maps.
- The **vignette edge** dominates Sobel on dermoscopic images.

These are properties of the images, recorded as such, not findings.

## 5. Time and memory (CUDA-synchronised, 8 images)

| Phase | Time | Peak allocated / reserved |
|---|---|---|
| Attribution (baseline, Grad-CAM, IG m = 256) | 9.95 s | 1.55 / 1.85 GiB |
| Control generation (random, centre, Sobel) | 0.04 s | — |
| Curve evaluation (176 curves; eval_batch 11) | 14.98 s | 0.29 / 1.85 GiB |
| Persistence | 1.14 s | — |
| Whole run | 30.8 s | — |

Curve evaluation now costs more than attribution, because each image needs 11 maps × 2 curves. This is **not** a full-RQ4 runtime estimate.

## 6. Files and integrity

**New files**
- Code: `src/rv1/xai_controls.py` and `src/rv1/xai_control_diag.py`.
- Tests: `tests/test_xai_controls.py`.
- Configs: `configs/xai_controls_v1_spec.json` and `configs/xai_control_diag_v1.json`, which records the spec as a supporting input.
- The run directory.
- `runs/stage2d3/`: the pre-change copy of `xai_curves.py`, a preview crop of the review sheet, and the integrity evidence.

**Changed:** `xai_curves.py`.

**Integrity:** 413 files before, 415 after.
- **0 changed, 0 removed.**
- The only additions are the two new configs.
- Earlier outputs, checkpoints and caches are unchanged. For the cache `.npy` files, size and modification time were compared.
- No `__pycache__` was created.

## 7. Remaining requirements for the real-model XAI pilot

1. **The real v1-trained primary model**, which needs main training authorised. Pilot or smoke weights must not be used.
2. **The XAI manifest sample** (Derm7pt primary cohort, 249 + 249 enriched pairs or a pilot subset of them) and both modalities.
   - The mismatch groups then become (label, modality, explainer, condition). The derangement must exclude the same **pair** across modalities, and the grouping already separates modalities.
3. **Production IG m**, re-evaluated on that sample (candidate 256 against 512, possibly 1024).
4. **The analysis layer**: matched eligible pairs, E4.1/E4.2 contrasts, Holm correction, and the sensitivity analyses defined in 02_rev2 §6. None of it is implemented yet.
5. **Cost measurement at pilot scale**, and a decision on `eval_batch` (1 as reference, 11 validated as equivalent on this sample).
6. Whether the Sobel **vignette sensitivity** and the **centre-prior/lesion-position coincidence** need documenting as interpretation caveats. They must not change the control definitions after results are seen.
7. Other architectures (target layers), and the simulated-shift arm with transformed masks.

**Stage 2D3 is complete.**
