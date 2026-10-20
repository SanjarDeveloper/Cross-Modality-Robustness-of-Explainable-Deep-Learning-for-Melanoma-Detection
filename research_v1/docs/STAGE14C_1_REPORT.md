# Stage 14C.1 Report: Citation Gaps Closed and Methods v2

**Scope**
- No experiments, no tuning and no Discussion.
- No scientific estimate was changed and no interval was recomputed.
- Nobody was contacted, no agreement was accepted and no access was requested.
- All earlier files are preserved; the integrity check confirms this.

## 1. Deliverables (new)

| File | Content |
|---|---|
| `thesis_materials/METHODS_DRAFT_v2.md` | Methods with verified citation keys `[@key]`, kept separate from the audit tags `[Sxx]`. It applies all Methods corrections and ends with a 28-item change log. Generator: `runs/stage14c1/make_methods_v2.py`, with the change list in `methods_v2_changes.json` |
| `thesis_materials/VERIFIED_REFERENCES_v2.bib` | 29 entries, each with an `x-status`: META / PASSAGE / ABSTRACT / UNRESOLVED. v1 is preserved |
| `thesis_materials/CITATION_CLAIM_MAP_v2.md` | Updated claim map (new rows A0a–b, A10a, C9–C10; updated C7, C8, E1, E2) with a per-reference status summary (§G) |
| `thesis_materials/DATASET_TERMS_AUDIT_CLARIFICATION_v1.md` | Versioned correction of the Stage 14C audit |
| `runs/stage14c1/check_methods_v2.py` | Verification |
| `runs/stage14c1/integrity_*` | Integrity |

## 2. Dataset audit corrections
- **DDI patients.** The official page states "656 images representing 570 unique patients". The local metadata (`ddi_metadata.csv`: DDI_ID, DDI_file, skin_tone, malignant, disease) contain **no patient identifier**, so a usable mapping does not exist.
  - Analyses remain image-level. No patient groups were inferred, which would also conflict with the re-identification clause.
  - Methods v2 states that the DDI intervals do not account for within-patient dependence.
- **Exact terms vs interpretation**
  - The DDI agreement restricts copies of the dataset and derivative works, but has **no clause about publishing results or analyses**.
  - Each artefact category is marked **unresolved**: aggregate metrics, prediction files, cached arrays, perturbed images, explanation maps, figures.
  - Derm7pt: "permission **not established**" for reproducing images, which is not a legal conclusion of prohibition.
- **Dates.** The inferences of download dates and agreement acceptance from file timestamps are **withdrawn**. Acquisition dates and the DDI agreement version are unresolved.

## 3. Reference verification (finished as far as accessible)

| Reference | New evidence (2026-10-11) | Status |
|---|---|---|
| Guo et al. 2017 | PMLR PDF read: §2 defines ECE with M equally spaced bins and attributes it to Naeini et al. 2015 | Passage inspected |
| Adebayo et al. 2018 | NeurIPS PDF read: §1 contributions; the model-parameter and data randomisation tests (§3–4); edge detectors (§5.4). The official BibTeX has an empty pages field | Passage inspected; pages unresolved (none invented) |
| Hooker et al. 2019 | NeurIPS PDF read: abstract; §1 (ROAR; a drop without retraining may reflect distribution shift). Official BibTeX pages empty | Passage inspected; pages unresolved |
| Petsiuk et al. 2018 (RISE) | Venue **verified** in the BMVC 2018 proceedings listing on the BMVA archive (paper 1064; the PDF matches) | Verified; no page numbers exist on that record |
| Holm 1979 | Verified from the JSTOR-generated cover page of a hosted copy: Scand J Statist 6(2): 65–70. The live JSTOR page did not render for automated access | Metadata verified |
| Naeini et al. 2015 | AAAI publisher page: abstract (BBQ). No page range on the page. The ECE definition was not inspected in this paper | Abstract only; attribution via Guo |
| DeLong, Wilcoxon, Fitzpatrick, Efron | Publisher records give only the first page or no pages | End pages **unresolved**; not invented |

**Keys cited in Methods v2.** Methods v2 cites 28 of the 29 bibliography keys (34 citations). selvaraju2017gradcam is kept as an alternative version but is not cited.

**Identification vs validation.** Every citation is worded as identification of a method or dataset. Our implementation choices are explicitly labelled as ours, notably:
- the blurred IG baseline (the paper's example is a black image) and our completeness tolerance;
- the Grad-CAM layer;
- the reference-map controls and contrasts;
- the positive-class `ece_pos_v1` (Guo et al. define ECE for top-label confidence);
- the cluster bootstrap scheme and the spline knots;
- EfficientNet-B3 at 224 px.

Hooker et al. is cited as a **limitation** of insertion/deletion without retraining.

## 4. Checks (`check_methods_v2.py`): 218 checks, 0 failures
- Every `[@key]` resolves in the v2 bib, and every entry has an `x-status`. No open page ranges.
- Every cited key appears in the v2 claim map.
- **Support adjacency (pattern-based).** Each citation's line contains a term of the claim that the source was verified to support. Metadata-only keys appear only where the named method is attributed.
- **Required content**
  - "single-institution" removed;
  - the DDI 570-patient limitation present;
  - Derm7pt never called smartphone;
  - an explicit sentence that no ethics approval or exemption is asserted;
  - audit tags retained;
  - `[CITATION NEEDED]` kept only for the software citations.
- **Preserved files unchanged against the 14C snapshot:** Methods v1 and its ledger, bib v1, claim map v1, the audit, Results v1 and MASTER v2.

**Limits.** The adjacency check confirms topical placement, not scientific adequacy. Whether a sentence over-reads a source was judged manually, against the passages listed in the claim map.

## 5. Still inaccessible or unresolved
- **Sources**
  - The Kawahara et al. 2019 full text (closed access), so the Derm7pt clinical-image acquisition remains undocumented.
  - The live Derm7pt site terms (no readable page).
  - The DDI AIMI portal page (no readable terms; the project-page agreement was used).
  - The book sections in Davison & Hinkley and in Harrell.
  - Missing pages for DeLong, Wilcoxon, Fitzpatrick, Efron, Naeini, Adebayo and Hooker.
  - Harrell's subtitle and edition.
  - Software citations (PyTorch, timm, Albumentations, OpenCV, NumPy, SciPy).
- **Institutional and author questions** (not externally verifiable)
  - Ethics approval or exemption (A1).
  - Dataset permissions for each artefact category, and the non-commercial status (A2; clarification §2).
  - Supervisor or protocol approval (A3).
  - University format and the AI-use declaration (A4).
  - Dataset acquisition dates and the accepted DDI agreement version.

## 6. Integrity
The reference is the Stage 14C end snapshot (4,920 files). The present snapshot has 4,925 files:
- 0 files changed and 0 removed;
- the cache is unchanged;
- 5 files were added: the 4 new 14C.1 files, plus `docs/STAGE14C_REPORT.md`, which was written after the 14C snapshot.

## 7. Stop point
The cited Methods draft (v2) and the source verification are complete. No other chapter was drafted.
