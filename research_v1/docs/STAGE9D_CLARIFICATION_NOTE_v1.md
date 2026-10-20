# Stage 9D Clarification Note v1 — 2026-10-04

The frozen `STAGE9D_REPORT.md`, Addendum K, the spec and the run outputs are unchanged. The recipe itself (including GaussNoise std 0.2–0.44) is unchanged.

1. **Provenance**
   - The legacy transform **order** and **explicit parameters and probabilities** were preserved.
   - The **implicit defaults** were resolved for the installed Albumentations **2.0.8**.
   - The legacy code uses 2.x keyword names (for example `quality_range`). That API compatibility alone does **not** establish the exact historical environment, and it does not prove that the legacy runs used these implicit defaults.
2. **Withdrawn claim: "10–30× stronger" noise.** The Stage 9D statement that the 2.x GaussNoise default is "about 10–30 times stronger" than older defaults is **withdrawn**. It was not backed by an exact historical version, source, quantity and calculation. What remains is factual: the frozen recipe uses std 0.2–0.44 of the maximum value, and the review showed strong visible noise when GaussNoise was selected.
3. **Timing figures.** These are two different numbers, and **neither is a total training slowdown**:
   - **Two-repeat mean timing ratio** heavy/std = **1.60**. The first `aug_std_v1` repeat included warm-up.
   - **Warmed-repeat throughput ratio:** using the second repeats only, 301 vs 158 images/s gives about **1.90**.
