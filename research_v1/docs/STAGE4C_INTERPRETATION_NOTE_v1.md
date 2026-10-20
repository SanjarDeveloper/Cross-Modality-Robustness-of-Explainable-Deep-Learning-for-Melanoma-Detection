# Stage 4C Interpretation Note v1 — 2026-10-03

The frozen `STAGE4C_REPORT.md`, its run outputs and the thesis draft are unchanged. This note refines the wording, mainly of the conversational summary.

1. **PAD: report detection counts, not "moderate performance".** The summary should state the raw results:

   | Cohort | Melanoma detected | Specificity | AUROC [95% CI] |
   |---|---|---|---|
   | PAD all | 31 of 52 images (59.6%) | 1619/2246 | 0.684 [0.599, 0.765] |
   | PAD eval | 16 of 26 images (61.5%) | 834/1124 | 0.720 [0.612, 0.828] |

   Qualitative labels such as "moderate" are not used.

2. **DDI AUROC and chance.** The DDI AUROC was 0.460 [0.341, 0.571]. The interval includes 0.5, so the data do not show ranking better than chance. This **does not establish equivalence to chance**: the interval is wide, it also includes values below 0.5, and no equivalence test or margin was specified. Wording such as "indistinguishable from chance" should be read as "not shown to differ from 0.5 on these data".
