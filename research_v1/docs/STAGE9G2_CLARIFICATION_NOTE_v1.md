# Stage 9G2 Clarification Note v1 — 2026-10-04

This note qualifies two statements in Stage 9G2. These files are left unchanged:
- `docs/STAGE9G2_REPORT.md`;
- `runs/stage9g2/thesis_results_rq3_heavy_rq2.md` (and its `.sha256`);
- the run outputs.

## 1. "Family-specific" means most consistent, not exclusive

The robustness benefits of heavy augmentation were **most consistent** for the RQ2 families that overlap the training recipe: blur, illumination (AUROC) and compression (AUROC). They were **not exclusively** confined to those families.

**Non-overlapping or partly overlapping factors also had some favourable matched pairs**

| Factor | Favourable result | Where (`matched_worst_case.csv`) |
|---|---|---|
| Scale (partial overlap) | Positive sensitivity benefit in 3/9 pairs | EfficientNet-B3 s2, +35.9 pp |
| Scale | Positive AUROC benefit in 6/9 pairs | — |
| Nuisance (not in the recipe) | Positive sensitivity benefit | EfficientNet-B3 s1, +12.9 pp, interval above zero; DenseNet-121 s0, +7.1 pp |

**Correct reading:** benefits were concentrated in, and most reproducible across seeds for, the overlapping families. Results for non-overlapping factors were mixed and more often unfavourable. Neither pattern establishes robustness to unseen real-world shifts.

## 2. Scope of the transformed-image digest comparison

"Identical perturbations across strategies" was **directly verified only on the recorded audit sample**: the SHA-256 digests of the first 4 transformed images of every unit, which were checked against Stage 8B and the Stage 9C2 manifests in all 405 units.

The identity of the other transformed images follows from the shared deterministic implementation:
- the same `perturb.py` hash;
- seeds that depend only on `image_uid` and factor;
- the same cached inputs.

Those other images were **not** individually digested. The severity-0 baseline checks (max |Δp| = 0.0) cover every image, but only at severity 0.
