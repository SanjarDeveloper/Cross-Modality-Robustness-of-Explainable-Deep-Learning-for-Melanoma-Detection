# Protocol Addendum E — 2026-10-01: Sequential IG extension, candidate m = 512 against m = 1024 (Stage 3B.1)

This addendum is written and frozen **before** any m = 1024 computation. Addendum D and the Stage 3B run are preserved unchanged.

## E1. Nature of the extension

In Stage 3B, candidates m = 128 and m = 256 failed criterion (a). This stage is a **sequential numerical extension** made after that result. It is **not** a new or untouched pilot.

**Unchanged from Stage 3B (Addendum D):**
- the 24 pairs and 48 references;
- the model: Stage 3A `best.json` → `ckpt_epoch8.pt`, SHA-256 `303d21e7…03dd`;
- preprocessing `prep_v1_cached256`;
- the blurred baseline;
- the Riemann midpoint rule with `internal_batch` 16;
- `curves_v1` with S = 32 and `eval_batch` 11, which passed the Stage 3B equivalence check;
- `xai_precision_v1`;
- `controls_v1`;
- every tolerance and criterion.

**Limits:**
- m = 1024 is computed for **all 48** references.
- There is no escalation beyond 1024, and m = 1024 is **not** called converged.

## E2. Reuse of Stage 3B results

The Stage 3B m = 512 maps, completeness records and curves, and the m-independent control curves (random, centre, Sobel, Grad-CAM), are reused only after these checks:
- the Stage 3B receipt and every output file hash verify;
- the recorded settings and precision policy match exactly;
- the pilot pair and reference lists are identical.

**Compatibility recheck**
- m = 512 is recomputed for the frozen batch-check subset (4 pairs, 8 references).
- Acceptance (fixed now):
  - identical completeness failure flags;
  - |Δe_rel| ≤ 1e-6;
  - maximum |Δ| of the primary map ≤ 1e-6 × max|map|;
  - |ΔAUC| ≤ 1e-6 for insertion and deletion;
  - identical curve and endpoint statuses.
- The computation is deterministic, so exact agreement is expected.
- If the recheck fails, the run **stops** and the discrepancy is reported. Cached and new outputs are then not combined.

## E3. Assessment

The rule is `xai_pairs.assess_candidate`, unchanged from Addendum D5.
- The candidate is 512 and the comparison is 1024.
- Completeness denominator: all 48 references.
- e_rel is assessed where it is defined.
- The (c) medians cover references usable at both m values, with that count reported.
- Results are pooled, with subgroups by modality and by label as descriptive results.

**Outcome:**
- If the candidate passes the pooled rule, m = 512 is frozen as the production IG setting **only** for the seed-0 ResNet-50 and the current unperturbed Derm7pt pipeline. Every later image still undergoes its own completeness and eligibility checks, and other models and conditions need their own assessment.
- If it fails, or cannot be assessed, the result is **UNRESOLVED**.

## E4. If candidate 512 passes

**Recipient→donor mapping**
- The planned mapping is unchanged. The deterministic rule gives the same donors as Stage 3B, and this is asserted.
- The mapping is persisted under the explainer label `ig512` before any mismatch curve is evaluated.
- Donor eligibility is taken from the m = 512 maps.

**What is recomputed**
- Only the IG mismatch curves and the IG scores at m = 512.
- The m = 256 scores remain diagnostic-only and are not relabelled.

**Grad-CAM**
- The eligibility rule is unchanged: all-zero maps are ineligible.
- There is no absolute-value or alternative-target substitution.
- If a cell has zero eligible pairs, that contrast is **NOT ESTIMABLE**; it is not a zero effect.
- Primary paired comparisons describe eligible pairs only.
