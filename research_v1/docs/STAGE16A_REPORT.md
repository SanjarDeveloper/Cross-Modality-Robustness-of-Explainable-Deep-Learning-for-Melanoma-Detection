# Stage 16A Report: Thesis Reading Draft

**Scope**
- Assembly and wording only: no experiments, no change to estimates, and no university template.
- Final Word or PDF formatting was not started.
- All historical files are preserved.

## 1. Corrected individual drafts (new versions; earlier versions preserved)

| New file | Changes |
|---|---|
| `RESULTS_DRAFT_v2.md` | Applies correction notes C1–C5: difference scales; fixed-model vs E4.6 intervals; spline-of-logit comparator. Also applies the Derm7pt wording; the counts 1,002 pairs / 2,004 images; the 18-model matrix (three seeds per combination, primary model included); non-causal heavy-recipe wording; embedded figure paths. The drafting-only §4.6 was removed. Estimates, intervals and populations are unchanged (same `values.json`) |
| `ABSTRACT_DRAFT_v2.md` | 2,004 images; the 18 models include the primary model, with three seeds per combination; the interval types distinguished; spline probability baseline. 298 words |
| `CONCLUSION_DRAFT_v2.md` | The causal "mainly by moving operating points" replaced by the observed coexistence of gap narrowing and lower specificity; spline baseline; interval types; seeds per combination |
| `DISCUSSION_DRAFT_v3.md` | Same operating-point and seed wording fixes; spline baseline |
| `INTRODUCTION_DRAFT_v3.md` | Matrix description (18 models, three seeds per combination); spline baseline |
| `METHODS_DRAFT_v3.md` | Derm7pt 2,004 images; matrix wording, including the primary model |
| `APPENDIX_LEGACY_REPRODUCTION_v2.md` | Applies the Stage 13B clarification (timestamps; per-image identity; training pathway not inferred); internal file paths removed |

Generators: `runs/stage16a/make_results_v2.py`, `make_versions.py`, `make_appendix_v2.py`. Logs: `results_v2_changes.json`, `version_changes.json`.

## 2. Frozen sources and assembly
- **`CHAPTER_SOURCE_MANIFEST.json`.** The exact file and SHA-256 for the abstract, each chapter, the protocol history, the appendix, the master table and the bibliography (`VERIFIED_REFERENCES_v3.bib`, keys preserved). It also lists the four figures and the two value files.
- **`THESIS_READING_DRAFT_v1.md`.** Assembled by `runs/stage16a/assemble.py`, which refuses to run if any source hash differs from the manifest.
  - Provisional numbering: Abstract; Chapters 1–6; References; Appendix A (master table, supplementary); Appendix B (legacy reproduction).
  - The protocol history is integrated **once** as Section 3.14 and cross-referenced from Chapters 1, 3, 4 and 5.
  - Removed from the reading text: internal tags, generator comments, change logs, drafting conventions and drafting-only sections. They are indexed in `AUDIT_PACKAGE_INDEX.md`.
  - Simple bullet blocks of complete sentences were joined into paragraphs, and bold-labelled bullets became paragraphs. Tables and comparison lists were kept.
  - References are generated from the bibliography for all 36 cited keys, in a provisional format.

## 3. Word counts (reading draft)

Approximate counts by whitespace after removing markdown symbols; tables are included.

| Part | Words |
|---|---|
| Abstract | 317 in the assembled part (including its heading and labels); the abstract body alone counts 298 |
| 1 Introduction | 1,243 |
| 2 Related Work | 1,479 |
| 3 Methods (including 3.14 protocol history) | 4,775 |
| 4 Results | 5,047 |
| 5 Discussion | 2,971 |
| 6 Conclusion | 833 |
| References | 1,253 |
| Appendices A–B | 1,136 |
| **Total (excluding references)** | **≈ 17,800** |

## 4. Consistency
`check_consistency.py` ran **144 checks with 0 failures**; see `CROSS_CHAPTER_CONSISTENCY_REPORT.md`. It covers:
- source hashes; UTF-8; removal of internal material;
- citations; tables and figures; image paths; section cross-references;
- counts (pairs, images, lesions, patients); the threshold; model scope; seeds;
- RQ numbering; terminology; units and interval types;
- the required, not-estimable and optional categories;
- "no new numbers introduced by assembly";
- preservation of earlier files.

## 5. Reading-draft completion vs submission readiness
- **The reading draft is complete.** All chapters are present, consistent and free of internal audit material, and every number traces to frozen sources.
- **The thesis is not ready for submission.** The open matters are listed in `SUBMISSION_OPEN_ITEMS.md`:
  - ethics determination; dataset permissions (Derm7pt and DDI images and derived artefacts; HAM10000 non-commercial status); supervisor approval;
  - the AI-use declaration; the university template and front matter;
  - bibliographic gaps and software citations; the citation-style conversion;
  - a manual editing pass for list-heavy sections.

None of these was resolved or presumed resolved.

## 6. Integrity
The reference is the Stage 15B end snapshot (4,945 files). The snapshot taken before writing these reports has 4,959 files:
- 0 files changed and 0 removed;
- the cache is unchanged;
- 14 files were added: the new drafts, the manifest, the reading draft, 4 figure copies, and `docs/STAGE15B_REPORT.md`.

The consistency report, the open-items list, the audit index and this report were written after that snapshot.

## 7. Stop point
The reading draft is assembled and checked. Final Word or PDF formatting was not started.
