# Stage 7C Interpretation Note v1 — 2026-10-04

The frozen `STAGE7C_REPORT.md`, its run outputs and the thesis draft are unchanged. This note corrects three statements.

1. **Normalization did not move every contrast towards zero.** On the identical δ = 0.05 subset, Grad-CAM insertion D went from a raw HL of −0.062 to a normalized HL of **−0.068**. That is a slightly *larger* magnitude. The general statement that rescaling "moves the shift contrasts towards zero" applies to Grad-CAM deletion and to both IG curves, not to Grad-CAM insertion.

2. **A CI that includes zero does not show equality.** It also does not show that there is no benefit. For example, Grad-CAM deletion against Sobel (HL 0.005 [−0.012, 0.023]) means that the data are compatible with small positive or negative differences. It does not mean that Grad-CAM and Sobel are equivalent. No equivalence margin or test was specified.

3. **Mismatch intervals.** The statement that mismatch intervals are "likely too narrow" is **not established**, and it is withdrawn.
   - The Stage 7D audit shows that the frozen mapping is a **one-to-one derangement** within every class × modality × explainer × condition group: each donor appears exactly once, with shift 1 in all 8 groups.
   - Maps are reused across analyses (as explainer maps and as donor maps), and the realized assignment is held fixed. Such shared artifacts can create dependence between recipients' contrasts.
   - The direction and size of any resulting error in the recipient-case bootstrap CIs has **not** been assessed. No claim is made about whether these CIs are too narrow or too wide.
