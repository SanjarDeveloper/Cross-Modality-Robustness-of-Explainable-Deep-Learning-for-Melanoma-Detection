# Stage 15B Report: Conclusion, Abstract and Introduction Clarification

**Scope**
- Writing only: no experiments, no literature expansion, and no change to scientific results.
- No thesis assembly or formatting.
- All earlier files are preserved; the integrity check confirms this.

## 1. Deliverables (new)

| File | Content |
|---|---|
| `thesis_materials/INTRODUCTION_CLARIFICATION_v1.md` | Versioned clarification: 5 changes, each with the old text, the new text and the reason |
| `thesis_materials/INTRODUCTION_DRAFT_v2.md` | Introduction with the clarification applied (generator: `runs/stage15b/make_intro_v2.py`). v1 is preserved |
| `thesis_materials/CONCLUSION_DRAFT_v1.md` | Concise answers to RQ1–RQ5; contributions as executed; principal limitations; 3 future-work priorities taken from Discussion §5.5; a bounded closing statement |
| `thesis_materials/ABSTRACT_DRAFT_v1.md` | Structured abstract (Background, Methods, Results, Conclusions), without citations |
| `thesis_materials/CONCLUSION_ABSTRACT_EVIDENCE_LEDGER.md` | Non-numeric empirical claims mapped to evidence; every number mapped to its source file and row or key |
| `runs/stage15b/*` | Templates, the fill script, the ledger/check script and integrity files |

## 2. Introduction clarification
1. **Paired design.** "Isolates the change of image type" became a within-lesion comparison that still includes acquisition-related differences, and "does not isolate image type alone".
2. **Classifier development.** "Typically developed on dermoscopic images" was narrowed to the classifier actually studied.
3. **Dermoscope availability.** "Dermatoscopes are not available in every setting" was replaced by an attributed statement from a verified passage: Vollmer et al., Summary → Background and Objectives.
4. **Epidemiology placeholder.** The [CITATION NEEDED] placeholder was removed by narrowing scope: the thesis makes no burden or availability claims. No epidemiology was added.
5. Version header.

## 3. Abstract
- **Word count:** 299 by whitespace tokens (297 by the word regex), within the **provisional** 250–300 target. This target is not a university rule.
- **Selected results only**, formatted from the unrounded source values:
  - the paired modality gap (249 melanomas, Derm7pt PRIMARY, ResNet-50 seed 0) with the two further seeds;
  - the heavy-augmentation trade-off (blur benefit against specificity and calibration costs);
  - XAI faithfulness, with the inconclusive E4.6 increment;
  - the adaptation AUROC decrease, with only 9 adapt-val melanomas.
- **Corrected during drafting.** An early version said "the same evaluations were repeated for 18 models". The RQ3 models were **not** evaluated on DDI or with explanations, so the text now reads "Classification (except DDI) and perturbation analyses, without explanations, were repeated for 18 models".

## 4. Validation (`make_ledger_and_check.py --check-only`): 93 checks, 0 failures

**Run on the actual generated text**
- The drafts equal their filled templates.
- Every value key appears in the ledger.
- Every MASTER-sourced value equals the corresponding row of `MASTER_RESULTS_v2.csv` at full precision.
- All intervals in the text are consistent.
- The E4.6 and forgetting intervals include zero, and the adaptation AUROC interval excludes zero, matching the prose.
- The abstract word count lies within 250–300; there are no citations in the abstract.
- **Wording**
  - there is no un-negated first, novel, superior, state-of-the-art, clinical-readiness or causal wording in the abstract, the conclusion or Introduction v2. One "blur caused" was reworded to "produced" after the check flagged it;
  - the required distinctions are present in the conclusion: modality-associated vs causal; sensitivity change vs overall improvement; perturbation-specific vs general robustness; faithfulness and localisation vs clinical correctness; no demonstrated increment vs no information; this adaptation vs adaptation in general.

**Re-derived from source files**
- Blur benefit in all 9 matched models.
- Narrowing always accompanied by lower dermoscopic specificity.
- 18 RQ3 models.
- The centre prior beats IG except at scale 1.0.
- All four E4.2 contrasts negative.
- 1,002 pairs.

**Consistency with other chapters**
- Methods v2 (primary model).
- Discussion v2 (Secondary A/B; partial scale overlap).
- Completion matrix (Secondary B not estimable; DDI and XAI not run for the RQ3 models).
- Introduction v2: clarifications applied; the Vollmer passage is verified in the source text.
- Preserved files (Introduction v1, Related Work v1, Methods v2, Results v1, Discussion v2, MASTER v2, bib v3) are unchanged.

**Limits.** The wording checks are pattern-based. Whether each sentence is fair to the evidence was judged manually, against the ledger.

## 5. Remaining submission requirements (not addressed in this stage)
- **University rules:** word limits, abstract format, chapter template, citation style, front matter.
- **Declarations and approvals:** the AI-use declaration; the institutional ethics determination; dataset-permission questions (especially images and derived artefacts for Derm7pt and DDI); supervisor approval of the protocol and the drafts.
- **Bibliographic gaps:** the Hauser author list; the Daneshjou 2021 end page; the Hendrycks proceedings page; missing end pages for some statistics references; software citations.
- **Pending version updates:** Results v2 (applying correction note C1–C5 and the Derm7pt wording) has not yet been generated as a file. The current Results draft remains v1 plus its correction note.
- **Thesis assembly:** cross-references, final table and figure numbering, appendices.

## 6. Integrity
The reference is the Stage 15A end snapshot (4,939 files). The present snapshot has 4,945 files:
- 0 files changed and 0 removed;
- the cache is unchanged;
- 6 files were added: the 5 new 15B files, plus `docs/STAGE15A_REPORT.md`, which was written after the 15A snapshot.

## 7. Stop point
The Conclusion and Abstract drafts are complete and verified. Thesis assembly and formatting were not started.
