# Protocol Addendum G — 2026-10-01: v1 threshold and prediction tables (Stage 4A)

This addendum is frozen before any HAM VAL, HAM TEST or Derm7pt inference in this stage.

## G1. Fixed settings

| Item | Setting |
|---|---|
| Model | Stage 3A `best.json` → `ckpt_epoch8.pt`, SHA-256 `303d21e7…03dd` |
| Preprocessing | `prep_v1_cached256` |
| Precision | `xai_precision_v1` |
| Logit | One logit; p = sigmoid(logit) in float32, carried as an exact Python float. **No rounding** |
| Inference batch | **16**, as in the Stage 3A validation |
| Order | Manifest order within each cache: `ham_val` in the order of `configs/main_ham_val_full_v1.csv`; `ham_test`, `derm7pt_dermoscopy` and `derm7pt_clinical` in manifest order |
| Final partial batches | Recorded per cache (n mod 16) |

## G2. Threshold

The rule is 02_rev2 §1.1, `v1_highest_t_img_sens95`, implemented in `rv1/threshold.py`:
- k = ⌈95·n₊/100⌉, computed with integer arithmetic;
- t = the k-th largest melanoma probability on HAM VAL;
- predict melanoma iff p ≥ t;
- **NOT ESTIMABLE** (no fallback) if n₊ < 20, or if any score is non-finite.

**Check before use:** the HAM VAL predictions must reproduce the stored Stage 3A epoch-8 validation outputs, with the same record IDs and labels and a maximum probability difference of 0. Otherwise the stage stops.

**Freezing:** `threshold.json` and its sidecar are written **before** any HAM TEST or Derm7pt inference, and are verified by hash before every use.

## G3. Same-file Derm7pt pairs (tolerance declared before inference)

The 9 same-file pairs keep both rows. Their two rows come from different caches, in different batch positions.

**Declared consistency criteria:**
- the two cached 256 images are byte-identical;
- |Δ probability| ≤ 1e-5;
- the prediction class is identical.

Any differences are reported. Batch-shape effects are about 1e-6 (Stage 3B.1).

## G4. Data roles

| Data | Role |
|---|---|
| HAM VAL | Already used for checkpoint selection, and now for the threshold. **Not** an independent estimate |
| HAM TEST | Evaluation only |
| Derm7pt | Evaluation only, but **not pristine**: it was inspected in legacy work and in the Stage 3B numerical XAI pilots (48 references) |

**Populations:**
- The RQ1 population is the frozen primary cohort: 1002 pairs, 249 melanoma.
- The 498-pair XAI sample is not used.
- No filtering by XAI eligibility.

**Out of scope:** PAD and DDI are not evaluated. Only descriptive counts are produced: no tests, CIs, DeLong, bootstrap or Holm.
