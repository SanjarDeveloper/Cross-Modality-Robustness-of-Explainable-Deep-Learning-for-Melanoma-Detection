# Stage 14E Report: Discussion Corrections, Focused Related-Work Search, Positioning

**Scope**
- No experiments, no tuning and no change to scientific results.
- No Introduction and no full literature-review chapter.
- All earlier drafts are preserved; the integrity check confirms this.

## 1. Deliverables (new)

| File | Content |
|---|---|
| `thesis_materials/DISCUSSION_DRAFT_v2.md` | v1 with 12 logged corrections (10 substantive, 2 header). Built by `runs/stage14e/make_discussion_v2.py` from the same value files, so no number changed |
| `thesis_materials/DISCUSSION_CORRECTION_LOG_v2.md` | Old → new text for each correction, with its reason |
| `thesis_materials/DISCUSSION_EVIDENCE_LEDGER_v2.md` | Updated interpretation rows and a regenerated per-value map (54 keys) |
| `thesis_materials/VERIFIED_REFERENCES_v3.bib` | v2 entries unchanged, plus 8 related-work entries (37 in total, each with an `x-status`) |
| `thesis_materials/RELATED_WORK_EVIDENCE_MATRIX.md` | Search log (queries, sources, date, selection rationale), study-level matrix, topics without an inspected study |
| `thesis_materials/RESEARCH_POSITIONING_NOTE.md` | Contributions, relation to the inspected literature, unsupported novelty claims |
| `runs/stage14e/check_discussion_v2.py`, `check_related_work.py`, `integrity_*` | Verification and integrity |

## 2. Discussion corrections (all confirmed against records)
- **Heavy recipe vs synthetic scale.** The overlap is **partial**: the recipe zooms in or rescales by 0.75–1.25, while the synthetic factor zooms out to 2.4×. Source: `rq3h_rq2_9g2` spec, `factor_overlap_with_heavy_recipe`. Hair and glare are **not** training augmentations (Addendum K).
- **Seed range.** The "widened range" claim was removed. All nine standard-recipe gaps lie within the ResNet-50 seed span of 11.2–43.4 pp; this was re-derived.
- **RQ5 thresholds.** The Secondary A threshold was re-estimated by the fixed rule; only Secondary B was not estimable.
- **E4.6 comparator.** The wording now says that comparator choice matters, with no claim of inflated benefit for arbitrary features.
- **E4.6 status.** It is described as exploratory, specified before fitting but after earlier exposure, and not preregistered.
- **Severity and ranking.** The unsupported claim that rankings transfer better than severities was removed.
- **Paired design.** It controls lesion identity and shared characteristics, not all confounding.
- **DDI identifiers.** The text now reads "no usable patient identifier was found in the local release inspected".
- **Surrounding prose** was rechecked by pattern search (group H of the check): no remaining instance was found.

## 3. Focused narrative search (2026-10-11)

**Method**
- Five topics, with verbatim queries.
- Searched with web search (extended), the Europe PMC search and full-text API, Crossref and the arXiv API.
- The matrix records the inclusion and exclusion reasons.

**Included sources (new)**
- **Read in full text:**
  - Vollmer et al. 2026 — a **paired same-lesion** study of close-up vs dermoscopic images, including a threshold analysis;
  - Rios-Duarte et al. 2024 — paired images, mostly from the 7-point database, with models *trained* on each modality. Its Methods state that the imaging-system specifications were not provided;
  - Combalia et al. 2022 — ISIC 2019 external validation;
  - Young et al. 2021 — stress tests.
- **Abstract only:** Maron et al. 2021, Daneshjou et al. 2021, Hauser et al. 2022, Hendrycks & Gimpel 2017.

**Not found in this search** (recorded, and **not** treated as a literature gap):
- explanation faithfulness under real dermatology domain shift;
- explanation features adding error-prediction value beyond a probability baseline;
- primary studies of fine-tuning on smartphone clinical images.

## 4. Positioning
- The contribution is framed as a structured, auditable evaluation. No claim is made of being first, novel, superior or clinically ready.
- "First paired dermoscopic/clinical comparison" is explicitly **contradicted** by Vollmer et al. and Rios-Duarte et al.
- Our RQ1 direction (lower sensitivity, higher specificity and lower AUROC on non-dermoscopic images) is **consistent in direction** with Vollmer et al. No numerical comparison is made, because the models, tasks and cohorts differ.

## 5. Verification
- **`check_discussion_v2.py`: 139 checks, 0 failures.** It includes all 117 Stage 14D checks applied to v2, plus group H, which checks:
  - that the corrections are present and the overstatements absent;
  - the seed span, the scale-overlap record, Addendum K's hair/glare statement and the RQ5 threshold status, each against its source;
  - that Discussion v1 is unchanged.
- **`check_related_work.py`: 45 checks, 0 failures.**
  - Every number and quotation in the matrix is found in the inspected full text (Vollmer, Rios-Duarte, Combalia, Young) or in the inspected abstract (Maron, Daneshjou 2021, Hauser, Hendrycks).
  - All keys resolve in bib v3.
  - There are no un-negated "first", "novel", "superior" or "state-of-the-art" statements, and the disclaimers about the nature of the search are present.
- **Limits.** These are pattern and quotation checks. Whether a source supports the exact thesis claim was judged manually; the matrix column "thesis claim it can support" records that judgement.

## 6. Remaining unresolved
- **Bibliographic fields:**
  - the complete author list for Hauser et al.;
  - the end page of Daneshjou et al. 2021;
  - the ICLR proceedings page for Hendrycks & Gimpel;
  - the full text of Maron et al. (the publisher link returned HTML).
- **Unchanged from earlier stages:** the Derm7pt acquisition details (now additionally noted as undocumented by Rios-Duarte et al.), ethics, dataset permissions, supervisor approval and the AI-use declaration.

## 7. Integrity
The reference is the Stage 14D end snapshot (4,928 files). The present snapshot has 4,935 files:
- 0 files changed and 0 removed;
- the cache is unchanged;
- 7 files were added: the 6 new 14E files, plus `docs/STAGE14D_REPORT.md`, which was written after the 14D snapshot.

## 8. Stop point
Stage 14E is complete. No Introduction or literature-review chapter was drafted.
