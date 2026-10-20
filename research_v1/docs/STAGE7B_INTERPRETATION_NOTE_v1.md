# Stage 7B Interpretation Note v1 — 2026-10-04

The frozen `STAGE7B_REPORT.md`, its run outputs and the thesis draft are unchanged. This note clarifies their interpretation.

1. **Median vs Hodges–Lehmann.** The sample median and the HL pseudomedian are different estimators. A difference between them in one sample, such as the Grad-CAM deletion D, does **not** by itself establish that the population distribution is asymmetric. Sampling variability alone can produce such differences.

2. **What the signed-rank test means when symmetry is doubtful.** The Wilcoxon signed-rank test assumes a distribution symmetric about its centre. If that assumption fails, its p-values cannot simply be relabelled as a valid general test of the pseudomedian, or of the median. The prespecified Wilcoxon results are preserved unchanged, together with this assumption. No estimator or test is substituted after seeing the results.

3. **Status of these analyses.** The E4.1 and E4.2 results are the protocol-designated primary analyses of a study that was developed retrospectively, through staged decisions, numerical pilots and prior exposure of Derm7pt. The study is **not** untouched and confirmatory, and the word "confirmatory" in the 7B tables should be read in that light.

4. **Random is the weakest reference.** Outperforming the random-map control does **not** imply outperforming the centre prior, Sobel edges or the mismatched-map controls. The 7B descriptive tables show, for example, Grad-CAM deletion close to or below the centre and Sobel controls. Stage 7C reports uncertainty for those contrasts.
