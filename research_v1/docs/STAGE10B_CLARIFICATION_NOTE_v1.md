# Stage 10B Clarification Note v1 — 2026-10-08

This note corrects and qualifies Stage 10B. The following are all unchanged:
- `docs/STAGE10B_REPORT.md`;
- the runs (`runs/xai_sim_10b/…`), the thresholds and the frozen readiness rule;
- the eligibility and localisation definitions.

## 1. Correction: m = 1024 could not have qualified

The frozen readiness rule (Stage 10B `spec.json → ig_readiness.rule`) requires the pooled assessment **and every one of the 11 conditions** to pass all criteria. Criterion (a) is a completeness failure rate ≤ 5%, i.e. at most 1 failure in 20 per condition.

The saved Stage 10B results (`ig_convergence_table.csv`, column `fail_1024`) already show **2/20 m = 1024 completeness failures (10%) in both nuisance 0.5 and nuisance 1.0**.

Therefore **m = 1024 cannot meet the unchanged rule**, whatever a later comparison against m = 2048 shows. Proposal A in the Stage 10B report ("m = 1024 vs m = 2048; if m = 1024 qualifies, full arm at m = 1024") was wrong to present m = 1024 as a possible candidate.

**The corrected next step is Stage 10B.1:** candidate m = 2048 with reference m = 4096, gated on criteria (a) and (b) at m = 2048 before any m = 4096 computation.

## 2. Units of the pilot

The pilot's **220 references are 20 images (10 melanoma / 10 non-melanoma, unique lesions) under 11 conditions**. They are not 220 independent cases. Pooled rates and medians mix repeated images, and any uncertainty statement must account for this.

## 3. Centre-prior vs IG localisation: matched populations

The Stage 10B localisation table reported each map over its own available references. A centre-vs-IG comparison must use a matched population: references with available localisation for every map compared.

Recomputed from the saved records (`runs/stage10b1/matched_localisation_10b.json`; no new maps):

| Condition | Matched refs (images; mel / non-mel) | IG512 IoU / pointing | Centre IoU / pointing | Random IoU | Centre > IG (refs) |
|---|---|---|---|---|---|
| none 0 | 19 (19; 10 / 9) | 0.282 / 0.84 | 0.720 / 1.00 | 0.137 | 19 / 19 |
| scale 1.0 | 20 (20; 10 / 10) | 0.268 / 0.85 | 0.018 / 0.15 | 0.022 | 3 / 20 |
| all 11 conditions (repeated images) | 204 (20 images; 108 / 96) | 0.273 / 0.82 | 0.665 / 0.89 | 0.121 | 183 / 204 |

**Comparisons including Grad-CAM** are restricted further by Grad-CAM all-zero maps, to 136 references from 18 images, of which only 42 are non-melanoma. That population is selected by model evidence.

**What the conclusion depends on.** The conclusion that the centre prior localises better than IG at severity 0 holds on the matched population. It reflects the **centred acquisition of HAM lesions**. Under scale 1.0 the lesion is moved away from the centre and the order reverses.
