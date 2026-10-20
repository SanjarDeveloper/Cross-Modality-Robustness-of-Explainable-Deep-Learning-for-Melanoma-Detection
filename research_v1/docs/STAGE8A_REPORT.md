# Stage 8A Report — RQ2 Perturbation Simulator (perturb_msim_v1): Implementation and Validation

Date: 2026-10-04. All work in this stage used `research_v1/.venv_model`.

**Scope:** image transformations and tests only. No model, no inference, no XAI and no full dataset transformation.

## Status

| Item | Result |
|---|---|
| Specification frozen before review | `configs/perturb_msim_v1_spec.json` (SHA-256 `8a5d096b…`). It pins the module hash of `perturb.py` (`855cf57b…`); the review run refuses to start if the module differs |
| Module | `src/rv1/perturb.py`: `apply(img256, factor, severity, image_uid, split_seed=0, mask256=None)` returns `(img, mask, meta)` |
| Tests | New `test_perturb.py` (**12** tests). The relevant suites (`perturb`, `preprocess`, `masks`, `foundations`) pass |
| Visual review | `runs/perturb_review_8a/20261004T075815Z_534dc79b_e67bce5b/`. 5 sheets, inspected (§5). The receipt verifies |
| Data used | 4 HAM TRAIN images only. No HAM TEST, no model predictions |
| Protected inputs | 0 changed, 0 removed. The only addition is the Stage 7D note |

**Why the full suite was not run:** no shared module changed.

## 1. Sources (nothing invented where a source defines it)

| Source | What it fixes |
|---|---|
| 02_rev2 §4 | The five factors; severity grid {0, 0.125, …, 1}; `perturb_seed = stable_seed(split_seed, image_uid, factor)`; claim limits (radius_max 7 px, JPEG quality_min 18, zoom_out_max 2.4) |
| 03_rev1 §4/§6/§10 and 03_rev2 §8/§14 | Legacy `msim` ops with an explicit rng; `scale_v1` applying the same geometry to the mask with zero fill; severity-0 identity; two-process determinism |
| Legacy `code/msim/simulator.py` (read-only) | Formulas and defaults, reused |
| Legacy `code/data.py` | The shift is applied to the **cached 256 px** image **before** resize and normalisation |

## 2. Specification (summary)

**Pipeline**
- Image: cached 256×256 uint8 RGB → `perturb.apply` → existing `preprocess.to_model_input` (INTER_LINEAR to 224, /255, ImageNet normalisation).
- Mask: `mask_interp_v1` → `mask_to_cache_grid` (INTER_NEAREST_EXACT to 256) → `apply` → INTER_NEAREST_EXACT to 224.
- This is unchanged from the verified cached256 path. There is no switch to a raw→224 path.

| Factor | Formula and parameters | Geometry | Mask |
|---|---|---|---|
| blur | r = 7s px; r < 0.3 → identity; Gaussian with k = 2·round(3r)+1, σ = r, BORDER_REPLICATE | no | unchanged |
| illumination | Sign ±1 drawn once per image. bright = 0.28s·sign, contrast = 1 − 0.25s·sign, temp = 0.22s·sign. x = (img/255 − 0.5)·contrast + 0.5 + bright; R·(1+temp); B·(1−temp). float32, clip, `astype(uint8)` (truncation, as legacy) | no | unchanged |
| compression | s = 0: **bypass**. Otherwise JPEG quality q = round(95 − 77s) via `cv2.imencode` with OpenCV defaults (OpenCV 4.11.0), then decode | no | unchanged |
| scale | z = 1 + 1.4s; offset 0.18s × a random direction ∈ {−1, 0, 1}² (clipped). INTER_AREA inset; legacy synthetic low-frequency skin-tone surround; feathered alpha | **yes** | `scale_v1`: the same inset geometry, INTER_NEAREST_EXACT, zero fill |
| nuisance | Glare ellipse (intensity 0.95s; random centre and angle) plus round(16s) of 16 pre-drawn semi-transparent hair strands | no | unchanged. **Occlusion is not masked out** |

**RNG**
- One local `Generator(PCG64(stable_seed(0, image_uid, factor)))` per call. The global RNG is untouched.
- Results are independent of record order, process and PYTHONHASHSEED.
- **Retrospective decision:** every draw is unconditional and made in a fixed order, so all severities share one realization: the same illumination sign, scale offset direction, glare placement and hair strands (higher severity adds strands).
- This replaces two legacy behaviours: content-hash seeding through Python `hash()`, and a glare draw skipped at low intensity.

**Metadata:** every call returns the full parameter metadata (seed, radius and ksize, sign and coefficients, JPEG quality and bytes, inset geometry, glare and hair parameters).

**HAM all-white masks:** ISIC_0026042 and ISIC_0029819 keep their flags (Addendum A3). Nothing is repaired.

**Claim:** these are controlled synthetic changes. They are **not** a reproduction of real smartphone acquisition, and they are not calibrated to it.

## 3. Tests (`tests/test_perturb.py`, 12 tests)

| Test | What is checked |
|---|---|
| Severity 0 | Bit-identical image and mask for every factor, and model input identical to `preprocess.to_model_input`. JPEG at s = 0 is a bypass |
| Output validity | Shapes, dtypes, finite model inputs and mask values {0,1} across the whole grid. Inputs are not mutated and the global numpy RNG state is unchanged |
| Invalid inputs | Severities < 0, > 1, NaN, bool or str; unknown factor; wrong image shape; mask not {0,1}; wrong mask shape. All rejected |
| Determinism | Record-order independence; seeds differ by image; **byte-identical outputs in two processes with PYTHONHASHSEED 0 and 12345** |
| Shared realization | The same glare position, illumination sign and scale direction across severities |
| Blur | Laplacian variance falls across s = 0, 0.25, 0.5, 1; ksize 43 at s = 1; identity at r < 0.3 |
| Illumination | Exact formula on a constant image |
| JPEG | Quality 85 / 56 / 18 at s = 0.125 / 0.5 / 1; encoded size decreases |
| Scale | Mask area ratio = inset area / frame area (within 0.03) for an unclipped disk; centroid at the analytic inset position (within 1 px); dark lesion pixels > 90% inside the transformed mask |
| Clipped object | An edge-touching object gets zero padding outside the inset. **No unclipped-area claim is made for it** |
| Photometric factors | Mask unchanged |

**Parameter progression vs metric monotonicity:** the tests check parameter progression. They do not assume that image statistics or model metrics worsen monotonically for every image.

## 4. Visual review sample

HAM TRAIN. For each combination of label and mask PNG mode (L or P), the first record in stable-seed order was taken:

| Record | Label | Mask mode |
|---|---|---|
| ISIC_0028283 | non-melanoma | L |
| ISIC_0024438 | non-melanoma | P |
| ISIC_0030754 | melanoma | L |
| ISIC_0024700 | melanoma | P |

None of these carries an `all_foreground` flag.

Each sheet shows the original and severities 0, 0.5 and 1, with the mask boundary overlaid in green on the 256 grid.

## 5. Observations from inspecting the sheets (technical, not clinical review)

| Factor | Observations |
|---|---|
| **blur** | Progressive loss of fine structure; nothing unexpected. Mean \|Δ\| at s = 1 is 3.3–6.3 grey levels |
| **illumination** | The direction is per image: 3 of 4 became bright and warm, 1 became dark and cool (purple cast). At s = 1, strong **saturation**: about 30–33% of channel values clip at 255, the lesion's contrast is markedly reduced, and the lesion looks washed out |
| **compression** | Barely visible at review scale (mean \|Δ\| 3.4–5.2 at q = 18). 8×8 block artefacts appear mainly in smooth regions |
| **scale** | The lesion shrinks and moves off-centre. **The mask stays aligned with the inset.** Artefacts: a visible **rectangular inset seam** and an unnatural smooth, blotchy synthetic surround at s = 0.5–1. The feathered image border fades lesion pixels near the inset edge while the mask edge stays hard. For lesions touching the frame, the mask inherits **straight frame-edge segments** |
| **nuisance** | Glare clips to white and can **cover part of the lesion** (e.g. ISIC_0030754), and the mask is not reduced accordingly. Hair strands overlay the lesion semi-transparently |

None of these settings was changed after the review. The parameters are the frozen legacy and protocol values.

## 6. Files

**Code, tests and config**
- New: `src/rv1/perturb.py`, `src/rv1/perturb_review.py`, `tests/test_perturb.py`.
- New: `configs/perturb_msim_v1_spec.json`, `configs/perturb_review_8a_v1.json`.

**Review run:** five `review_<factor>.png` sheets, `review_sample.json`, `perturbation_params.json` (60 parameter records) and the receipt.

**Stage 7D note:** `docs/STAGE7D_INTERPRETATION_NOTE_v1.md`. The ~10% reference is an independent-uniform top-k expectation, not a spatially matched null model, and low agreement does not identify the correct method.

**Integrity:** 1124 files before, 1125 after. **0 changed, 0 removed**, 1 added (the note). The cache `.npy` files are unchanged. No `__pycache__` was created.

## 7. Remaining before RQ2 model evaluation (not started)

1. Freeze the RQ2 evaluation specification: HAM TEST 1548 images, all 5 factors × 9 severities, the seed-0 model and frozen threshold, lesion-cluster bootstrap, the worst-case-loss and crossing-marker definitions, and the ranking.
2. A runtime pilot (02_rev2 §4 marks the grid as [PILOT]).
3. RQ2 inference and statistics.
4. The simulated XAI arm (§6.7) after that.
5. RQ3.

**Stage 8A is complete.**
