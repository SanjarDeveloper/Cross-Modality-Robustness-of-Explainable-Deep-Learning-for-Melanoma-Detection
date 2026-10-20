# Stage 16C Report: Reproducibility Clarification and Supervisor-Review Handoff

**Scope**
- No experiments, no change to estimates, no broad rewriting.
- No Word or PDF output.
- Nothing was sent to anyone.
- **Review draft, not submission-ready.**

## 1. Deliverables (new; historical files preserved)

| File | Content |
|---|---|
| `thesis_materials/THESIS_READING_DRAFT_v3.md` | Supervisor review draft (generator `runs/stage16c/make_reading_v3.py`) |
| `thesis_materials/EDITORIAL_CHANGELOG_16C.md` | 9 edits: 2 substantive and 7 copy-edits, listed separately (`runs/stage16c/editorial_log.json`) |
| `thesis_materials/CHAPTER_SOURCE_MANIFEST_v3.json` | The unchanged chapter-source hashes, the bib v5 hash, the environment-evidence file hashes, reading drafts v1–v3 and word counts |
| `thesis_materials/VERIFIED_REFERENCES_v5.bib` | v4 plus the full contributor list of the PAD-UFES-20 dataset record |
| `thesis_materials/SUPERVISOR_REVIEW_BRIEF.md` | One-page brief: aim, findings, design and limitations, decisions needing input, section references |
| `thesis_materials/SUBMISSION_OPEN_ITEMS_v3.md` | Updated open items |
| `runs/stage16c/check_coverage.py`, `coverage.json`, `integrity_*` | Verification and integrity |

## 2. Substantive changes

**1. Environment provenance.** Contemporaneous evidence was located: package freezes saved on 27 Sep 2026 for both project environments, run-start records, specification fields and receipts. Each environment is now described separately:

| Environment | Used for | Evidence | Verified versions / limits |
|---|---|---|---|
| Global interpreter (initial inventory, 27 Sep) | 5 early audit receipts. It is also the "environment inventory" hashed in all receipts | The inventory itself (`in_virtualenv: false`) | The inventory includes scikit-learn 1.7.2 and two OpenCV distributions. It describes the global interpreter, **not** the v1 environment |
| Preparation environment | 3 receipts | Stage 2B2C freeze | CPU only; no PyTorch |
| Project environment (v1 experiments) | 123 receipts | 27 Sep freeze (after installing timm); timm 1.0.28 in all 19 training run-start records (to 4 Oct); SciPy 1.15.3 in the RQ4 statistics specification (4 Oct) | PyTorch 2.11.0+cu128, torchvision 0.26.0, timm 1.0.28, NumPy 2.2.6, SciPy 1.15.3, Pillow 12.3.0, Albumentations 2.0.8, opencv-python-headless 4.11.0.86; scikit-learn not installed. **Unverified:** package versions for runs from 4 to 10 Oct (only the interpreter and Python version are recorded). The current environment matches, but this is current evidence only |
| Global interpreter (legacy reproduction, 10 Oct) | Appendix B | LR specification written at run time | Recorded there; the original v0 versions are unknown |

The Methods environment paragraph (§3.11) was rewritten briefly; the detailed table is in Appendix C.

**2. Pillow.** Its role (lesion-mask decoding; rendering of the figures, verified in the plotting code) and its version (12.3.0, from the freeze and the data-audit record) were added. No official citation guidance was found (no CITATION.cff; nothing on the documentation "About" page), so **no citation was invented**.

## 3. Copy-editing (claims preserved)
- **Converted from nested lists to paragraphs:** Methods §3.2 (datasets), Methods §3.8 (explainers; IG validity) and Results §4.1.
- **Checks on each edited block, old vs new.** No number lost: the only differences are punctuation-attached tokens and the "[95% CI]" label, which the chapter conventions define. No citation lost. Only bold labels became sentence text.
- **References** were regenerated from bib v5. The PAD-UFES-20 dataset contributors now come from the registry record; previously "Pacheco … others".
- **No other bibliographic field was changed:** no further authoritative evidence was found. Absent or unlocated pagination stays blank and is listed in the open items.

## 4. Verification (`runs/stage16c/check_coverage.py`): 177 checks, 0 failures

All Stage 16B content-preservation checks were run on v3:
- section map: 70 headings, 0 unmatched;
- reverse number coverage: only the 4 explained cases;
- 39 numerical claims checked in context;
- required RQ answers, hypotheses and qualifications;
- numbering, cross-references, image paths, table column counts, encoding;
- the 42 cited keys resolve in bib v5;
- no identifiers in the main text.

**New checks**
- The environment paragraph and the Appendix C provenance table contain the required distinctions.
- The freeze file contains every version stated.
- The PAD contributor list appears in the References.
- The header states the review-draft status.
- Reading drafts v1–v2, bib v4 and all chapter sources are unchanged.

**Limits.** These are presence, consistency and verbatim checks. The edited prose was additionally re-read in full.

## 5. Word counts (v3; approximate; tables included)

| Part | Words |
|---|---|
| Abstract | 316 (body: 298) |
| 1 Introduction | 1,258 |
| 2 Related Work | 1,476 |
| 3 Methods | 4,931 |
| 4 Results | 4,692 |
| 5 Discussion | 2,976 |
| 6 Conclusion | 827 |
| Appendices A–D | 2,168 |
| References | 1,756 |

## 6. Remaining open items (`SUBMISSION_OPEN_ITEMS_v3.md`)
- **Institutional** (the supervisor brief §4 lists the decisions): ethics; dataset permissions; protocol approval and terminology; whether the optional analyses are needed; the AI-use declaration; the university template.
- **Substantive:**
  - Derm7pt acquisition undocumented;
  - project-environment package versions unverified for runs from 4 to 10 Oct;
  - preprocessing sensitivity not studied.
- **Bibliographic:**
  - unlocated end pages (DeLong, Wilcoxon, Fitzpatrick, Daneshjou 2021, Naeini);
  - pages absent from the official NeurIPS records (Adebayo, Hooker);
  - OpenCV via a secondary route;
  - Hendrycks proceedings page;
  - abstract-only sources;
  - unread book sections;
  - no Pillow citation;
  - citation style.

**No further polishing cycles** are planned without a specific defect or a reviewer request.

## 7. Integrity
The reference is the Stage 16B end snapshot (4,969 files). The present snapshot has 4,976 files:
- 0 files changed and 0 removed;
- the cache is unchanged;
- 7 files were added: the 16C deliverables, plus `docs/STAGE16B_REPORT.md`.

This report was written after the snapshot.

## 8. Stop point
The supervisor-review package is prepared and **not sent**. Final Word or PDF formatting awaits the university requirements.
