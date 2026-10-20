# Stage 13B Clarification Note v1 — Interpretation of the Legacy Reproduction (2026-10-10)

This note qualifies several wordings in Stage 13B. The following are **unchanged**:
- `docs/STAGE13B_REPORT.md`;
- `thesis_materials/APPENDIX_LEGACY_REPRODUCTION.md`;
- `thesis_materials/UPDATE_13B_COMPLETION_AND_LIMITATIONS.md`;
- the LR run outputs.

Where this note differs from those documents, this note governs.

## 1. Chronology: what the timestamps show and what they do not

**What they show.** The present cache files carry modification times of 13 Sep 2026, 22:10–22:12 (+05). That is later than the legacy checkpoint (19:20) and the historical result files (19:20–19:46).

**What they do not prove.** File timestamps alone **do not prove that no earlier cache existed**. An earlier cache could have been created, used and later overwritten or replaced.

**Wording to replace.** In Stage 13B, the wording "the historical results could not have used these cache files" should be read as: *the present cache files post-date the historical results according to their file timestamps*.

**Related record.** `audit/cache_pixel_verification_20260920.json` (20 Sep 2026) verifies the present cache against raw decoding on a sample. It does not date any earlier cache.

## 2. What the exact reproduction establishes

**Established**
- The legacy raw-JPEG decoding pathway (`code/data.py:LesionDataset`, full image → Resize 224, fp16, batch 64) reproduces **all 30 historical aggregate checks exactly**:
  - integer counts;
  - AUROC;
  - legacy ECE;
  - DeLong and McNemar p-values.
- This **supports raw decoding as the pathway used for this reconstruction** of the historical aggregates.

**Not established**
- It does **not** establish historical **per-image** prediction identity, because no historical per-image outputs exist to compare against.
- The aggregate agreement is strong but indirect evidence. It does not exclude, in principle, other pathways that would yield identical aggregates.

## 3. The legacy *training* pathway is not inferred

**Preferred wording.** The LR result concerns the **evaluation** pathway that regenerates the historical aggregates. No conclusion is drawn about the legacy **training** input pathway from the current file timestamps. Stage 13B's description of the legacy model as "trained before the cache existed" is withdrawn.

**What the training log does and does not show.** It lacks the image-source line that the current `train.py` prints. This shows only that a different code version produced the log. It does not establish which input pathway that version used.

## 4. Meaning for v1

**What remains true for v1**
- The finding does **not** undermine v1's documented preprocessing consistency.
- v1 training, validation, thresholding and evaluation all use the same documented `prep_v1_cached256` path (Stage 2B2C; Addenda C, G, I, J).

**What the finding does not establish**
- It does **not** establish that v1 results are invariant to other preprocessing choices.
- Stage 2B2C already showed that raw→224 and cached256→224 inputs differ. The legacy-model probability differences observed in LR show that such differences can matter at the per-image level.
- v1's sensitivity to alternative preprocessing has not been studied.

## 5. Unchanged

- LR neither validates nor invalidates v1 by numerical agreement alone.
- The completion status "completed (diagnostic appendix)" stands, with the qualifications above.
