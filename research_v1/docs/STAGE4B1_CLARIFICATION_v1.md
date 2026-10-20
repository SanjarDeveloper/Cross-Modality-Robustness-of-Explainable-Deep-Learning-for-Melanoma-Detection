# Stage 4B1 Clarification Note v1 — 2026-10-01

This note corrects and clarifies the interpretation of Stage 4B1. The frozen report `STAGE4B1_REPORT.md`, the run outputs and the thesis draft file are **not** modified. Where wording differs, this note takes precedence.

1. **What was rejected.** The exact McNemar result (p = 4.19 × 10⁻²⁵) rejects the **null hypothesis** H1₀ of equal discordance, that is, of equal melanoma detection between modalities. It does not "confirm the research hypothesis".
2. **Avoided wording.** Phrases such as "the difference is not random" (including the Uzbek summary's "farq … tasodifiy emas") are withdrawn. A small p-value means the observed discordance would be very unlikely if H1₀ and the model assumptions held. It does not show that chance played no role.
3. **Confidence intervals.** The 95% CI [36.5, 50.2] pp is the output of a procedure that, under its assumptions, covers the true value in about 95% of repeated samples. It is **not** a 95% probability that the fixed true effect lies in this particular interval. The Uzbek summary's phrase "haqiqiy farq taxminan 37–50 foiz punkt oralig'ida" should be read as "the values compatible with these data, under the bootstrap assumptions".
4. **Same-file pairs.** These are 9 pairs, both slots holding the same file. They are outside the primary cohort. Because their predictions are identical, they change the **denominators and the concordant cells only**, never the discordant counts b and c.
5. **What the paired design controls.** Within the paired comparison, case composition is matched: the same cases are seen under both modalities. However, acquisition and framing differences (camera, distance, field of view, lighting) are **inseparable** from modality in Derm7pt. The difference is therefore not a pure "modality effect".
6. **What a small p-value does not do.** It does not remove the other limitations:
   - Derm7pt's prior exposure in legacy work and the Stage 3B XAI pilot;
   - the single model and threshold;
   - the unknown dependence between cases, since there are no patient identifiers.
