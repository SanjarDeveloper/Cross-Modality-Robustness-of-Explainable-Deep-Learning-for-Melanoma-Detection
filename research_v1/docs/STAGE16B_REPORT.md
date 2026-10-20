# Stage 16B Report: Editorial and Content-Preservation Review

**Scope**
- Editorial and bibliographic work only: no experiments, analyses or changes to estimates.
- No university template; no Word or PDF production.
- **The thesis is not submission-ready.**

## 1. Deliverables (new; nothing historical modified)

| File | Content |
|---|---|
| `thesis_materials/THESIS_READING_DRAFT_v2.md` | Edited reading draft (generator `runs/stage16b/make_reading_v2.py`) |
| `thesis_materials/EDITORIAL_CHANGELOG_16B.md` | All 68 edits, old → new, with category and reason (`runs/stage16b/editorial_log.json`) |
| `thesis_materials/CONTENT_COVERAGE_MAP_16B.md` | Section map (70 source headings); reverse number coverage; 39 numerical claims checked in context; required elements |
| `thesis_materials/VERIFIED_REFERENCES_v4.bib` | v3 + 6 software entries; full Hauser author list; Efron pages. 43 entries, all with `x-status` |
| `thesis_materials/CHAPTER_SOURCE_MANIFEST_v2.json` | The unchanged chapter-source hashes, plus reading drafts v1/v2 and bib v4 hashes, and word counts |
| `thesis_materials/SUBMISSION_OPEN_ITEMS_v2.md` | Institutional items, substantive issues, bibliographic gaps (located vs absent) and formatting tasks, kept separate |
| `runs/stage16b/check_coverage.py`, `coverage.json`, `integrity_*` | Verification and integrity |

## 2. Full read
The whole of reading draft v1 (1,565 lines) was read before editing. Problems found and fixed in v2:

| Kind | Finding | Fix |
|---|---|---|
| **Content loss** | The Stage 16A assembly had truncated the "As executed" column of the v0 → v1 table (Section 3.14.2) | Restored from the frozen source |
| **Rendering** | Unescaped "\|IG\|" broke four rows of the secondary-IG table | Escaped; every table now has a consistent column count (checked) |
| **Wording** | "Clinical close-up images" (RQ1 answer; acquisition not verified) | Replaced by "clinical (non-dermoscopic)" |
| **Wording** | "No patient identifier released" (DDI; three places in Methods) | Replaced by "no usable patient identifier was found in the local release inspected" |
| **Wording** | Heading "The deficit is partly an operating-point phenomenon" (asserted an untested decomposition) | Now "The operating point shifted as well" |
| **Wording** | Appendix B "resolved" / "explained" / "historical input pathway was identified" | Bounded to "a raw-decoding evaluation pathway was used" |
| **Stale statement** | "Concluding chapter and appendices … not drafted yet" | Replaced by the actual structure |
| **References** | "Salom~ao" accent bug; empty venues ("**.") | References regenerated from bib v4 (42 cited keys) |
| **Internal references** | "02_rev2", "§6.7", "(Appendix)", "stage" | Replaced by reader-facing wording |

## 3. Editorial changes

**Duplication reduced**
- The Methods RQ table was replaced by a cross-reference to Section 1.4.
- The Discussion 5.3 list was condensed into prose with the same four points and qualifications.

**Transitions.** Chapters 3–6 now open with a transition paragraph.

**Placement and numbering**
- Moved to a new Appendix D: Table 4.4 (heavy models, 18 columns), Table 4.7 (point estimates only), Table 4.12 (mostly counts) and Figure 4.4 (localisation). Each is replaced in the text by a pointer.
- Numbering is now consecutive: Tables 4.1–4.8 and D.1–D.3; Figures 4.1–4.3 and D.1. Every reference is updated and checked.

**Implementation identifiers.** 21 identifiers were removed from the main text (e.g. `prep_v1_cached256`, `xai_precision_v1`, `ece_pos_v1`, `perturb_msim_v1`, timm weight names) and listed in the new Appendix C with the section where each applies. Appendix B keeps the technical identifiers its diagnostic needs.

**Master table.** The 549-row master table remains a linked supplementary file (Appendix A).

**Figure captions.** Model, population, panel and axis labels and abbreviations were added, after checking each against the image content: "compression" = JPEG; R/E/D = architectures; "random_mean"; m = 2,048. The plots themselves were not altered.

**Limits of the prose editing.** Some list-heavy passages remain (Methods 3.2 and 3.8, Results §4.1). They are listed as editorial task D3. No limitation needed to interpret a result was removed, and no statement was strengthened; the coverage check confirms that the required qualifications are present.

## 4. Content preservation (`check_coverage.py`): 154 checks, 0 failures
- **Section map.** All 70 source headings are retained, merged or intentionally omitted with an explanation; there are 0 unmatched headings.
- **Reverse number coverage.** Every number in each source's retained content appears in v2, except 4 explained cases: two internal protocol section references, and two old table numbers changed by renumbering.
- **Numbers in context.** 39 numerical claims were checked with their model, cohort, estimator or denominator in the same paragraph or table row (for example, 43.4 [36.5, 50.2] pp with the 249 paired Derm7pt melanomas, and the Secondary B "not estimable" row with 9 < 20).
- **Required elements.** Answers to RQ1–RQ5 in Results, Discussion 5.1 and Conclusion 6.1; all four labelled hypotheses; 18 required qualifications (causal boundary, robustness scope, E4.6, forgetting, retrospective additions, safeguards, DDI dependence, fixed-OOF intervals, threshold uncertainty, and others).
- **Structure and rendering**
  - UTF-8 with no mojibake; accents correct;
  - no identifiers or citation placeholders in the main text;
  - reference list = cited keys (42); every key resolves in bib v4;
  - consecutive numbering, with every table and figure reference defined and every image path present;
  - section cross-references resolve; table column counts are consistent.
- **Preservation.** v1 and all frozen chapter sources are unchanged.

**Limits.** Section matching is lexical, and the equivalence of non-numeric wording was checked by a full manual read rather than an automated sentence-level comparison.

## 5. Bibliography (v4)

**Software citations, from official citation instructions**
- PyTorch: CITATION.cff preferred citation, Ansel et al., ASPLOS '24; pages from Crossref.
- timm: CITATION.cff, Zenodo DOI.
- Albumentations: README "Citing" section.
- NumPy: numpy.org/citing-numpy.
- SciPy: scipy.org/citing-scipy.
- OpenCV: Bradski 2000, via the CiteOpenCV wiki entry seen on a fork and the Q&A site, because opencv.org returned HTTP 403. Marked as a secondary route.

**Other corrections**
- Hauser et al.: full 33-author list from Crossref.
- Efron 1979: pages 1–26, verified on Project Euclid.

**Not located vs absent** (no page range was invented)
- **Not located:** end pages of DeLong, Wilcoxon, Fitzpatrick and Daneshjou 2021; Naeini pages.
- **Genuinely absent from the official record:** Adebayo and Hooker (the NeurIPS BibTeX has empty page fields).
- **Abstract-only limitations retained.**

**Every cited key verified** (42 of 42 in bib v4).

## 6. Substantive issues found (reported, not silently changed)
1. **Assembly truncation of the v0 → v1 table.** Fixed in v2. The 16A consistency check did not catch it; the new reverse-coverage check would have.
2. **Environment inventory.** `env/env_inventory_v1.json` was captured with the global Python interpreter (`in_virtualenv: false`), not the project environment. The Methods versions are nonetheless confirmed for the project environment: torch, NumPy, Albumentations and OpenCV in the run receipts; timm and SciPy read from `.venv_model` in this stage. The inventory's listing of scikit-learn reflects the global interpreter, consistent with the earlier E4.6 note that scikit-learn is not importable in the project environment. Recorded as open item B2.
3. **Pillow** is used in the code but not cited (open item C9).

## 7. Word counts (v2, approximate; tables included)

| Part | Words |
|---|---|
| Abstract | 316 (with its heading and labels; the body is 298) |
| 1 Introduction | 1,258 |
| 2 Related Work | 1,476 |
| 3 Methods | 4,846 |
| 4 Results | 4,656 |
| 5 Discussion | 2,976 |
| 6 Conclusion | 827 |
| Appendices A–D | 1,882 |
| References | 1,692 |
| **Main text (Abstract to Conclusion)** | **≈ 16,355** |

## 8. Status: reading draft vs submission
- **Complete as an edited reading draft:** coherent and content-preserving, with consecutive numbering and verified citations.
- **Not submission-ready.** The following are open (`SUBMISSION_OPEN_ITEMS_v2.md`):
  - institutional ethics; dataset permissions; supervisor approval; the AI-use declaration;
  - the university template; the bibliographic gaps in §5;
  - a further style pass;
  - final Word/PDF production.

## 9. Integrity
The reference is the Stage 16A end snapshot (4,959 files). The present snapshot has 4,969 files:
- 0 files changed and 0 removed;
- the cache is unchanged;
- 10 files were added: the 16B deliverables, plus the 16A report and documents written after the 16A snapshot.

## 10. Stop point
Stage 16B is complete. Final Word or PDF production was not started.
