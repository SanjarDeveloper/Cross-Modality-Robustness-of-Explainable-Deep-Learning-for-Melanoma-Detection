# Stage 14D Report: Discussion Draft

**Scope**
- Writing only: no experiments, no tuning, no optional extensions.
- Introduction, Abstract and Conclusion were not written.
- All earlier drafts are preserved.

## 1. Deliverables (new)

| File | Content |
|---|---|
| `thesis_materials/DISCUSSION_DRAFT_v1.md` | Discussion, §5.1–5.6:<br>• answers to RQ1–RQ5;<br>• interpretation;<br>• contributions;<br>• limitations and alternative explanations;<br>• implications;<br>• unresolved claims |
| `thesis_materials/DISCUSSION_EVIDENCE_LEDGER.md` | Interpretation → our evidence (each row marked observation or **hypothesis**); literature use → verified passage; per-value source map (53 keys) |
| `runs/stage14d/build_extra.py` → `values_extra.json` | 9 additional values from source files: IG completeness failures, Grad-CAM zero-map counts, PAD adapt-val melanomas, DDI metadata columns |
| `runs/stage14d/discussion_template.md`, `fill_discussion.py` | The draft is generated from the template, so no number is typed by hand |
| `runs/stage14d/check_discussion.py` | Verification |
| `runs/stage14d/integrity_*` | Integrity |

## 2. Sources used
- `METHODS_DRAFT_v2.md`.
- `RESULTS_DRAFT_v1.md`, with `RESULTS_CORRECTION_NOTE_v1.md` (C1–C5) applied.
- `MASTER_RESULTS_v2.csv`: the values were built from v1 rows, and the check confirms that the v2 estimates are identical.
- `UPDATE_13B_COMPLETION_AND_LIMITATIONS.md`.
- `VERIFIED_REFERENCES_v2.bib` and `CITATION_CLAIM_MAP_v2.md`.
- `DATASET_TERMS_AUDIT_CLARIFICATION_v1.md`.

## 3. Interpretation boundaries kept
- **RQ1**
  - The deficit is described as modality-associated, not causal.
  - Threshold transport, specificity and calibration are discussed alongside AUROC.
  - Seed variation (11.2–43.4 pp) and the architecture range are presented, and "43.4 pp" is explicitly not *the* effect size.
- **RQ2/RQ3**
  - The severity scale is arbitrary.
  - Benefits concentrate in families that overlap the augmentation recipe.
  - Absolute sensitivity and specificity trade-offs are discussed.
  - No universal robustness and no best architecture are claimed.
- **RQ4**
  - Faithfulness, localisation and clinical correctness are kept separate.
  - Grad-CAM zero maps are explained as a consequence of the ReLU in the published definition, and the resulting melanoma-enriched population is discussed.
  - Also discussed: IG completeness exclusions, selected simulated-arm populations, the centre prior's superiority in localisation on centred lesions, and the different meaning of |IG|.
  - E4.6 is described as no demonstrated increment over the specified spline baseline.
- **RQ5**
  - One recipe, one seed, one partition, with 9 adapt-val melanomas from 5 patients.
  - Adaptation is not called generally harmful.
  - Forgetting intervals are not read as "no forgetting".
- **Observations versus hypotheses.** Four explicitly labelled hypotheses, none of them tested:
  - lower clinical-image probabilities with a dermoscopy-tuned threshold;
  - seed differences arising through operating points;
  - the replacement scheme interacting with image statistics;
  - overfitting or class-balance mismatch in adaptation.
- **Limitations included**
  - prior data exposure and retrospective decisions;
  - fixed-model and fixed-OOF uncertainty;
  - three seeds;
  - DDI's 656 images from 570 patients analysed at image level;
  - pointwise intervals;
  - no inference of ethics approval from dataset availability.

## 4. Literature discipline
- **Citations.** 13 citations across 8 keys, all with an inspected passage or, for Derm7pt, the abstract. Each is used only for what its passage states (see ledger §2).
- **No new sources were needed.** No new comparison was introduced that required one.
- **Headline scores** are not compared with any cited paper. For DDI, the paper's finding is described qualitatively and the incomparability is stated.
- **No novelty, priority, superiority or clinical-readiness claims.** Contributions are stated as provisions, and novelty is marked UNRESOLVED.
- **Unresolved (§5.6):**
  - prior dermoscopic-vs-clinical deep-learning comparisons;
  - general medical-imaging domain-shift literature;
  - Derm7pt acquisition;
  - novelty;
  - ethics and dataset permissions.

## 5. Verification (`check_discussion.py`): 117 checks, 0 failures

| Group | What is checked |
|---|---|
| A | The draft equals the filled template; no unfilled tokens |
| B | Every MASTER-sourced value matches the same row in `MASTER_RESULTS_v2.csv` |
| C | Every interval satisfies lo ≤ estimate ≤ hi. The "includes zero" and "excluding zero" wording matches the interval. The inconclusive results (forgetting ×2, E4.6, \|IG\| vs centre) are confirmed to include zero |
| D | No un-negated confirmatory, significant, novel, first, superior or clinically ready; "equivalence" and "no forgetting" appear only negated; hypothesis labels present; RQ1 section discusses seed, threshold and AUROC; RQ5 and E4.6 generality caveats present |
| E | Each citation resolves, appears in the claim map, sits next to its verified passage term, and has PASSAGE or ABSTRACT status |
| F | Literals re-derived from sources: 43.4 pp, about four-fold, four more melanomas, seven per DDI group, DDI 656 images with no patient column, six features, lower thresholds for seeds 1–2, narrowing always with lower specificity, melanoma enrichment of the Grad-CAM population |
| G | Methods v2, Results v1, MASTER v2, bib v2 and claim map v2 are unchanged; the audit tags `[R4.x]` / `[M3.x]` are separate from `[@key]` |

**Limits**
- The wording and adjacency rules are pattern-based.
- Whether an interpretation is reasonable, and whether a hypothesis is fairly labelled, were judged manually and are documented in the ledger.
- The hypothesis-label check counts five occurrences: four hypotheses plus the definition in the conventions block.

## 6. Integrity
The reference is the Stage 14C.1 end snapshot (4,925 files). The present snapshot has 4,928 files:
- 0 files changed and 0 removed;
- the cache is unchanged;
- 3 files were added: the two new 14D files, plus `docs/STAGE14C_1_REPORT.md`, which was written after the 14C.1 snapshot.

## 7. Stop point
The Discussion draft and its verification are complete. Introduction, Abstract and Conclusion were not started.
