# Stage 8C Interpretation Note v1 — 2026-10-04

The frozen `STAGE8C_REPORT.md`, its run outputs and the thesis draft are unchanged. This note clarifies three points.

1. **Blur AUROC also declined.** AUROC fell from −0.011 at s = 0.25 to −0.066 at s = 1.0, and its CIs exclude 0 from s = 0.5 onwards. Blur therefore changed ranking performance as well as the operating point. The statement that "the ordering was largely preserved" should be read as "ranking declined much less than sensitivity", not as "ranking was unaffected".
2. **JPEG effects were not limited to quality 18.** Sensitivity was raised at several levels, for example +4.1 pp [0.9, 8.1] at s = 0.25, and false positives increased progressively (from 467 to 599). Only the AUROC decrease was concentrated at the strongest level.
3. **"No crossing" is limited to what was tested.** It refers only to the tested severity grid (0–1) and to the bootstrap replicates of this cohort. It is not a statement about severities, images or acquisition conditions outside them.
