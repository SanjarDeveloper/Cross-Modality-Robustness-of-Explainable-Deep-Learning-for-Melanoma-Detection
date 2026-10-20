# Stage 15A Report: Introduction and Related Work Drafts

**Scope**
- Writing only: no experiments, and no change to scientific results.
- No Abstract and no Conclusion were drafted.
- All earlier files are preserved; the integrity check confirms this.

## 1. Deliverables (new)

| File | Content |
|---|---|
| `thesis_materials/INTRODUCTION_DRAFT_v1.md` | §1.1 problem and motivation; §1.2 dermoscopic vs clinical images and shift; §1.3 why performance and explanations need separate evaluation; §1.4 aim and RQ1–RQ5 (protocol wording), with what was fixed in advance vs retrospectively; §1.5 scope and contributions as executed; §1.6 provisional structure |
| `thesis_materials/RELATED_WORK_DRAFT_v1.md` | Organised by concept and design:<br>• §2.1 three designs for modality shift (one fixed model on both modalities; separate models per modality; external validation);<br>• §2.2 acquisition and population shift;<br>• §2.3 synthetic robustness and augmentation;<br>• §2.4 explanation methods and evaluation limits;<br>• §2.5 confidence-based error prediction;<br>• §2.6 limited adaptation;<br>• §2.7 positioning |
| `thesis_materials/INTRO_RELATED_WORK_EVIDENCE_LEDGER.md` | Source versions and exact inspected passages (with section titles) for Vollmer et al. and Rios-Duarte et al.; a claim → source map; statements left unsupported |
| `runs/stage15a/check_intro_rw.py`, `integrity_*` | Verification and integrity |

## 2. Sources
- Methods v2.
- Results v1 with its correction note.
- Discussion v2.
- MASTER v2; no numbers from it were needed in these chapters.
- Bib v3, the related-work matrix and the positioning note.
- Protocol 02_rev2 §3–7, for the RQ wording, H1₀, the 12-test RQ4 family and the planned E4.6.

**No new literature was added.** All 18 cited keys come from bib v3, with passage- or abstract-level support.

## 3. Discipline applied
- **RQs**
  - They are stated as in the protocol; only the Derm7pt "clinical photographs" became "clinical (non-dermoscopic)", following Methods v2.
  - They are not reframed around observed results.
  - The retrospective decisions are listed separately, and none is presented as an original hypothesis.
- **Designs.** Training separate models per modality (Rios-Duarte et al.) is explicitly distinguished from evaluating one fixed model on both modalities (Vollmer et al.; our RQ1). External validation (Combalia et al.) is kept separate from both.
- **Review periods** are stated where review findings are used: Hauser et al., January 2017 – October 2021; Daneshjou et al. 2021, 2015 – November 2020. They are not presented as claims about the field in 2026.
- **No headline numbers from other studies** appear in the drafts; the ledger records them for audit only.
- **No "first", novelty, superiority or clinical-readiness claim.** Topics that returned no inspected study are stated as "not found in this focused search", with no gap inferred.
- **Scope.** The study is described as a retrospective computational evaluation with prior data exposure, not a prospective clinical validation.

## 4. Verification (`check_intro_rw.py`): 173 checks, 0 failures
- **Citations.** All 38 citations resolve in bib v3. Each sits in a sentence containing the term its source was verified to support, and only PASSAGE or ABSTRACT sources are used.
- **Wording.** No un-negated first, novel, superior, state-of-the-art or clinical-readiness wording; no inferred gap.
- **Foreign numbers.** None of the other studies' headline numbers appears in either draft.
- **Disclaimers present:** review periods, the nature of the search, and the per-modality vs fixed-model distinction.
- **RQ consistency with the protocol.** Checked: RQ1 wording; the five RQ2 factors; the RQ3 architectures; "no directional hypothesis" for RQ4; the 12-test family; E4.6 as planned; the scope statements.
- **Ledger quotations.** All 14 quotations for Vollmer et al. and Rios-Duarte et al. were found verbatim in the inspected Europe PMC full text.
- **Preserved files.** Methods v2, Results v1, Discussion v2, MASTER v2, bib v3, the matrix and the positioning note are unchanged.

**Limits.** The checks confirm placement, presence and verbatim matches. Whether each sentence states only what its source supports was judged manually; the judgements are recorded in the ledger.

**Correction made during drafting.** An unverified cohort description ("German multicentre") was removed from the ledger before the checks; a check confirms it is absent.

## 5. Remaining unsupported statements
- **Melanoma epidemiology and care-setting availability of dermoscopy** (Introduction §1.1): [CITATION NEEDED].
- **Not found in the focused search:**
  - faithfulness evaluation under real dermatology domain shift;
  - explanation features vs a probability baseline for error prediction;
  - fine-tuning studies on smartphone clinical images.
- **Carried over:**
  - bibliographic gaps: Hauser author list; Daneshjou 2021 end page; Hendrycks proceedings page; Maron full text;
  - ethics, dataset permissions, supervisor approval and the AI-use declaration.

## 6. Integrity
The reference is the Stage 14E end snapshot (4,935 files). The present snapshot has 4,939 files:
- 0 files changed and 0 removed;
- the cache is unchanged;
- 4 files were added: the 3 new 15A files, plus `docs/STAGE14E_REPORT.md`, which was written after the 14E snapshot.

## 7. Stop point
The Introduction and Related Work drafts are complete and verified. The Abstract and final Conclusion were not drafted.
