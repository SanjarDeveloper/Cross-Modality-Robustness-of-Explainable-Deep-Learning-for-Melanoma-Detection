# Stage 7D Report — Exploratory E4.4 (Class Subgroups) and E4.5 (Grad-CAM–IG Agreement)

Date: 2026-10-04. This stage was a CPU analysis of the verified Stage 7A, 7B and 7C artifacts only.

**Run:** `runs/rq4_explore_7d/20261004T074741Z_b9653a16_e281cd99/`. The spec and analyze receipts verify.

**Stage records:** `runs/stage7d/`.

## Status

| Item | Result |
|---|---|
| Stage 7C interpretation note | `docs/STAGE7C_INTERPRETATION_NOTE_v1.md` (frozen 7C files unchanged) |
| Specification frozen before computation | `analysis_spec.json` (hashed). It records the input hashes of every 7A, 7B and 7C receipt output, populations, formulas, tie rules, seeds (20261040+) and interval methods. New conventions are labelled RETROSPECTIVE |
| Donor-mapping audit | All 8 groups (class × modality × explainer, condition none) are **one-to-one derangements**: 249 recipients each, maximum donor multiplicity 1, shift 1, no self-donation |
| E4.4 populations | Grad-CAM 275 and IG 373 pairs, verified from the 7B per-pair inputs |
| E4.5 population | **227** pairs eligible for both explainers in both modalities (151 mel / 76 non-mel). Grad-CAM-only: 48; IG-only: 146 |
| Tests | New `test_rq4_explore.py` (5 tests). Relevant suites pass |
| Runtime | 77 s |
| Protected inputs | 0 changed, 0 removed |

**Why the full suite was not run:** no shared module changed.

**Not changed:** the primary RQ4 results and the 12-test Holm family.

## 1. Clarifications and donor audit

**Clarifications** (from `STAGE7C_INTERPRETATION_NOTE_v1.md`):
- On the identical subset, Grad-CAM insertion D went from a raw HL of −0.062 to a normalized HL of −0.068, so normalization did **not** move every contrast towards zero.
- A CI that includes zero does not demonstrate equality or a lack of benefit.
- The claim "mismatch intervals are likely too narrow" is withdrawn as not established.

**Donor audit** (`donor_mapping_audit.csv`)
- **Within a mapping:** each donor serves exactly one recipient.
- **Across analyses:** the same maps are reused, as explainer maps in E4.1 and E4.2 and as donor maps in the mismatch control.
- These shared artifacts can create dependence. The direction and size of any CI error have **not** been assessed, and the bootstrap was not redesigned.

## 2. E4.4 — melanoma vs non-melanoma (exploratory; not a formal interaction test)

**Method**
- D = random-referenced contrast (clinical) − contrast (dermoscopy), per pair.
- Class-stratified case bootstrap (RETROSPECTIVE): each replicate resamples within each class, with B = 2000 and multiplicity kept.
- The class difference is computed **within each replicate**. It is not inferred by comparing separate within-class intervals.

**Eligible vs excluded pairs**

| Explainer | Melanoma | Non-melanoma |
|---|---|---|
| Grad-CAM | 175 / 74 | 100 / 149 |
| IG | 204 / 45 | 169 / 80 |

**Results**

| Explainer, curve | Estimator | Melanoma [CI] | Non-melanoma [CI] | Mel − non-mel [CI] |
|---|---|---|---|---|
| Grad-CAM, insertion | median | −0.074 [−0.138, −0.021] | −0.067 [−0.130, −0.023] | −0.007 [−0.094, 0.072] |
| | HL | −0.090 [−0.129, −0.047] | −0.082 [−0.120, −0.044] | −0.008 [−0.064, 0.051] |
| Grad-CAM, deletion | median | −0.027 [−0.078, 0.006] | −0.008 [−0.042, 0.022] | −0.018 [−0.074, 0.032] |
| | HL | −0.033 [−0.065, 0.001] | −0.016 [−0.048, 0.015] | −0.016 [−0.062, 0.029] |
| IG, insertion | median | −0.034 [−0.063, −0.011] | −0.020 [−0.033, −0.009] | −0.015 [−0.045, 0.014] |
| | HL | −0.039 [−0.061, −0.017] | −0.026 [−0.040, −0.014] | −0.013 [−0.037, 0.012] |
| IG, deletion | median | −0.130 [−0.146, −0.092] | −0.032 [−0.050, −0.017] | **−0.098 [−0.125, −0.051]** |
| | HL | −0.126 [−0.153, −0.100] | −0.049 [−0.069, −0.032] | **−0.077 [−0.109, −0.043]** |

**Reading**
- For IG deletion, the clinical penalty was larger for melanoma than for non-melanoma pairs.
- For the other three combinations, the class differences were small, with CIs including 0. That does not establish equality.
- The two class populations have different eligibility rates, so the comparison is between selected subgroups.

## 3. E4.5 — Grad-CAM vs IG spatial agreement

**Method**
- Stored primary maps at 224×224, unaltered.
- **Spearman correlation:** Pearson correlation of average ranks.
- **Top-10% overlap (RETROSPECTIVE definition):** k = ⌈0.10·50176⌉ = 5018 pixels, taken by descending value with ties broken by ascending flat index; overlap = |A∩B|/k. **This is not Jaccard.**
- No map in the population was constant or non-finite. Such maps would have been unavailable, not scored as zero.

| Metric | n pairs | Dermoscopy, median [CI] | Clinical, median [CI] | Clin − derm, median [CI] | Class medians (mel / non-mel), derm; clin |
|---|---|---|---|---|---|
| Spearman ρ | 227 | 0.039 [0.030, 0.047] | 0.019 [0.015, 0.028] | −0.018 [−0.028, −0.008] | 0.047 / 0.021; 0.021 / 0.012 |
| Top-10% overlap | 227 | 0.150 [0.141, 0.165] | 0.144 [0.132, 0.161] | −0.008 [−0.023, 0.004] | 0.150 / 0.154; 0.145 / 0.144 |

HL values and their CIs are in `results.json`.

**Zeros and ties**
- Median fraction of exact-zero pixels: Grad-CAM 0.57; IG 0.50.
- **128 of 454 Grad-CAM maps** have a tie at the top-k cutoff; IG maps have none.
- Many pixels are tied at zero, and the deterministic index-based tie break affects overlap. Both can depress or distort ρ and overlap.

**Reading**
- The two methods highlight **largely different regions**. Overlap is only modestly above the roughly 0.10 expected by chance.
- Agreement is slightly lower on clinical images for ρ. The overlap difference has a CI that includes 0.
- **Not done:** no pixel-level tests, and pixels are not counted as N.
- Agreement does not establish correctness, localization or faithfulness, and low agreement does not identify which method is right.

## 4. Tests (`tests/test_rq4_explore.py`, 5 tests)

| Area | What is checked |
|---|---|
| Spearman | Identical maps → 1, reversed → −1, tied values match `scipy.spearmanr`, a constant map → unavailable (not 0) |
| Top-k | k is exact, including the ⌈⌉ case; ties are broken by ascending index; tie-at-cutoff is flagged |
| Overlap | A known example gives 0.5, which is not Jaccard (that would be 5/15) |
| Class contrast | Direction, reproducibility, and per-class resampling with multiplicity |
| Donor audit | A known derangement passes; a duplicated donor is detected |

## 5. Limitations

- Seed 0 only.
- Class-enriched cohort.
- Evidence-dependent exclusions.
- IG completeness failures (13.9%).
- 24 pilot pairs previously inspected.
- Retrospectively developed study.
- The E4.5 population is the intersection of both explainers' eligible pairs, so it differs from each single-explainer population.

## 6. Files

**New code, tests and config:** `src/rv1/rq4_explore.py`, `tests/test_rq4_explore.py`, `configs/rq4_explore_7d_v1.json`.

**New document:** `docs/STAGE7C_INTERPRETATION_NOTE_v1.md`.

**Run outputs:**
- `analysis_spec.json` with its sidecar;
- `results.json`;
- `donor_mapping_audit.csv`;
- `e44_subgroup_table.csv`;
- `e45_per_image_agreement.csv`;
- `bootstrap_replicates.npz`;
- `thesis_results_rq4_e44_e45.md`, which has its own sidecar;
- the receipts.

**Integrity:** 1103 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged.

## 7. Remaining (not started)

**RQ4**
- Secondary IG variants: |attr| maps and positive maps with σ = 2 smoothing. These need new curves.
- **E4.6:** error prediction from clinical-image features.
- The **simulated HAM arm** (§6.7), which depends on the RQ2 simulator.
- Optional additional-seed XAI on a 100-pair subset.

**Other research questions:** **RQ2** and **RQ3**.

**RQ4 is not complete. Stage 7D is complete.**
