# Stage 14C Report: Citation and Dataset-Source Verification; Versioned Results Corrections

**Scope**
- No experiments, no chapter drafting and no Discussion.
- No agreements were accepted, no access was requested and nobody was contacted.
- No historical artifact was modified.

## 1. Deliverables (all new)

| File | Content |
|---|---|
| `thesis_materials/MASTER_RESULTS_v2.csv` / `.md` | **Authoritative master table from now on**; v1 is preserved |
| `thesis_materials/MASTER_RESULTS_CORRECTION_LOG_v2.md`; `runs/stage14c/master_v2_changed_rows.json` | Exact changed rows (24; units column only) |
| `runs/stage14c/make_master_v2.py` | Generator. It asserts that all non-unit columns are byte-identical to v1 |
| `thesis_materials/RESULTS_CORRECTION_NOTE_v1.md` | Corrections C1–C5 to Results v1 and the 14B report |
| `thesis_materials/VERIFIED_REFERENCES.bib` | 29 entries, each with its verification route and depth |
| `thesis_materials/CITATION_CLAIM_MAP.md` | Claim → source → locator → read status → limits and our choices |
| `thesis_materials/DATASET_TERMS_AUDIT.md` | Terms, citations, redistribution and acquisition facts per dataset; unresolved items |
| `thesis_materials/METHODS_CITATION_INSERTION_PLAN.md` | Where each citation goes in Methods v2, with wording that separates the original method from our choices |
| `runs/stage14c/integrity_*` | Integrity |

## 2. Corrections

**Master table**
- Heavy-minus-standard Derm7pt AUROC-gap rows (18): "pp-scale proportion" → "AUROC units (difference of AUROC gaps)".
- E4.5 rows (6): "Spearman rho or overlap fraction" → "Spearman rho" / "overlap fraction (0-1)".
- No estimate, interval, p-value or count changed; the script asserts this.

**Results correction note**
- **C1.** A curve AUC lies in [0, 1], but a difference of two curve AUCs lies in [−1, 1]. The same holds for AUROC.
- **C2.** Uncertainty is described separately for:
  - fixed-model analyses: case, lesion or patient resampling, with no training or threshold variability;
  - E4.6: a bootstrap of fixed out-of-fold predictions, which excludes refitting and fold-assignment uncertainty.
- **C3.** The ranking share is 0.575 (1,150/2,000). "0.58" is a correct rounding, so the 14B report should not have called it a discrepancy. "0.57" was a float-formatting artefact.
- **C4.** Pointer to the corrected authoritative table.
- **C5.** E4.6 M0 is a spline of logit(*p*), not of *p* (found during this stage).

## 3. Source verification: methods and what was read

**Verification routes**
- **Registries and proceedings:** Crossref API, the arXiv API, PMLR and NeurIPS proceedings pages, the Harvard Dataverse and Mendeley Data APIs.
- **Full text:** Europe PMC for HAM10000, PAD-UFES-20 and DDI; arXiv PDFs, extracted locally, for Grad-CAM (IJCV version), Integrated Gradients and RISE.

**Read in full text, supporting passages located:**
- HAM10000 (Methods; Abstract);
- PAD-UFES-20 (§1.2.1; Experimental design);
- DDI (Materials and Methods; Results; Data availability);
- Grad-CAM (§3, Eqs. 1–2);
- IG (§3, Proposition 1, "Computing Integrated Gradients");
- RISE (§4.1).

**Abstract or proceedings page only:**
- Kawahara et al. 2019 (Derm7pt; the full text is closed);
- Adebayo et al. 2018;
- Hooker et al. 2019;
- Guo et al. 2017.

**Metadata only (cite for attribution of a named method only):**
- McNemar, DeLong, Wilcoxon, Hodges–Lehmann, Efron, Davison & Hinkley, Wilson, Harrell, Naeini et al.;
- ResNet, DenseNet, EfficientNet, ImageNet;
- Fitzpatrick 1988;
- Tschandl et al. 2019.

**Not fully verified (reported as such)**
- **RISE venue.** BMVC 2018 is confirmed via the authors' project page. The BMVC-hosted PDF could not be opened (expired TLS certificate) and was not bypassed.
- **Holm 1979.** There is no DOI; the record was verified via secondary sources and a third-party scan of the first page. The JSTOR page itself was not opened.
- **End pages.** DeLong, Wilcoxon, Fitzpatrick and Efron are entered with their first page only (or no pages); the author must complete them. Harrell's subtitle and edition were not taken from the registry.
- **Adebayo and Hooker page ranges.** These were not checked on the publisher pages.
- **Book sections.** The Davison & Hinkley section on clustered resampling and the Harrell restricted-cubic-spline section were not read.
- **Not verified at all (left as [CITATION NEEDED]):**
  - software citations: PyTorch, timm, Albumentations, OpenCV, NumPy, SciPy;
  - general domain-shift background;
  - melanoma and dermoscopy background;
  - further limitations of insertion/deletion metrics.

**Implementation vs original method (recorded in the claim map and the insertion plan)**
- Exact McNemar.
- Cluster bootstrap.
- Paired HL with bootstrap CIs.
- Blurred IG baseline (the paper's example is a black image).
- Midpoint rule and m = 512 / 2,048 with our own completeness gates.
- Positive-part IG map.
- Grad-CAM layer and upsampling.
- 32-step curves with a blurred baseline.
- Contrasts against random, centre, Sobel and mismatch maps, and the modality contrast. These are **our additions**, not part of RISE.
- Retrospective `ece_pos_v1`.
- EfficientNet-B3 at 224 px.

## 4. Dataset terms (details in `DATASET_TERMS_AUDIT.md`)

| Dataset | Terms | Redistribution of images | Status |
|---|---|---|---|
| HAM10000 (Dataverse v4) | CC BY-NC 4.0. Verified in both the local 2026-09-13 snapshot and the live API | Allowed non-commercially, with attribution | Unresolved: the institution decides whether thesis use and publication count as non-commercial |
| Derm7pt | Local README only: academic research use; "may not be redistributed"; rights retained by G. Argenziano; CC BY-NC-ND 4.0 | **No**; ND forbids adapted material | The live site could not be read. Unresolved: whether example images or derived images may appear in the thesis |
| PAD-UFES-20 (Mendeley v1) | CC BY 4.0. Verified in both the snapshot and the live API | Allowed with attribution | Verified |
| DDI | Stanford Research Use Agreement, read live on the project page: non-commercial research only; no redistribution; **no derivative works**; no link sharing | **No** | Unresolved: the agreement version accepted on 2026-09-13 was not saved, and whether derived files count as derivative works is open |

**Ethics.** The PAD source programme's ethics approval belongs to the data collectors. **No ethics conclusion is drawn for this thesis** (A1 remains unresolved).

## 5. Image acquisition (supported statements only)
- **PAD-UFES-20:** different smartphone devices; varying resolution and lighting; app cropping. Device models are not reported.
- **DDI:** clinical photographs taken at about 15 cm, "typically" with a clinic-issued smartphone camera.
- **HAM10000:** dermatoscopic images from MoleMax HD and DermLite FOTO systems, plus scanned slides, at two sites.
- **Derm7pt clinical images:** the device and conditions are **not verifiable** from accessible sources.
  - The earlier project phrase "close-up photographs" is therefore marked **unverified**.
  - Recommended wording: "clinical (non-dermoscopic) images; acquisition device not documented in accessible sources". This is planned for Results and Methods v2; the existing files are unchanged.
- "Smartphone" is used only for PAD, and for DDI with "typically".

## 6. Items kept separate from verifiable facts
- Institutional ethics determination (A1).
- Supervisor or protocol approval (A3).
- University format and the AI-use declaration (A4).
- The dataset-terms decisions above.

All of these remain with the author and the institution.

## 7. Integrity
The reference is the Stage 14B end snapshot (4,911 files). The present snapshot has 4,920 files:
- 0 files changed and 0 removed;
- the cache is unchanged;
- 9 files were added:
  - the 8 new 14C files in `thesis_materials`;
  - `docs/STAGE14B_REPORT.md`, written at the end of 14B and edited once (an E-check count) before 14C began.

The full-text extracts were kept in the scratchpad, outside the project.

## 8. Stop point
Stage 14C is complete. The Discussion was not started.
