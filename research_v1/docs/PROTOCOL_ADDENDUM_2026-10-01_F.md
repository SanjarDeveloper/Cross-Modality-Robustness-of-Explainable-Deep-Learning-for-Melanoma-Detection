# Protocol Addendum F — 2026-10-01: Fresh m = 512 / m = 1024 recomputation (Stage 3B.2)

This addendum was frozen **before** computation. Addenda D and E, and the Stage 3B and 3B.1 runs, are preserved.

## F1. Why fresh recomputation

**Response to Stage 3B.1.** That stage stopped because its cache-compatibility recheck failed: e_rel differed by up to 5.5e-6 against a tolerance of 1e-6. The chosen response is to **recompute m = 512 and m = 1024 from scratch for all 48 references** and use only those fresh results. Cached Stage 3B completeness values are not combined with the new attributions.

**Evidence kept.** The failed recheck is preserved as evidence that results depend numerically on batch shape.

## F2. Settings

**Unchanged from Addenda D and E:**
- the 24 pairs and 48 references, in the frozen order;
- the checkpoint, `303d21e7…03dd`;
- `prep_v1_cached256` and the blurred baseline;
- the Riemann midpoint rule;
- IG `internal_batch` 16;
- curves with S = 32 and `eval_batch` 11;
- `xai_precision_v1`;
- `assess_candidate` and every threshold.

**Outer batch:** the **outer batch** is a separate, recorded setting. It is **48 references in frozen order, a single batch with no partial batch**, and it is used for:
- baseline construction;
- logit(x) and logit(baseline) for completeness;
- both IG step counts.

**Endpoint-logit agreement:** the completeness endpoint logits from the two fresh IG calls must agree under the unchanged recheck tolerance, |Δ| ≤ 1e-6 on logits. If they do not, the run stops.

## F3. Reuse of m-independent outputs

The following Stage 3B outputs are reused **only if** fresh recomputation reproduces them exactly:
- the random, centre and Sobel control curves;
- the Grad-CAM maps and curves;
- the Grad-CAM mismatch curves.

The checks are:
- Grad-CAM maps (batch 8, as in Stage 3B);
- Sobel maps;
- baseline logits from the same outer batch.

The Stage 3B receipt and output hashes must also verify. If any check fails, nothing is reused, and the run stops.

## F4. Decision

- The rule is unchanged. The candidate is m = 512; the reference is m = 1024.
- If m = 512 passes the pooled rule, it is frozen **only** for the seed-0 ResNet-50 and the unperturbed Derm7pt pipeline.
- If it fails, the decision is UNRESOLVED and step counts are not escalated further.
- m = 1024 is never called converged.
- Diagnostic m = 512 pair eligibility is reported in either case, and is labelled as diagnostic if m = 512 is not selected.

## F5. Numerical limitation (recorded)

Identical repeated calls with the same batch shape are deterministic. Different batch shapes can differ numerically, by about 1e-6 in the logits.

Future production runs must therefore specify how the completeness endpoint batches are formed, including the final partial batch.

**No batch-size independence is claimed.**
