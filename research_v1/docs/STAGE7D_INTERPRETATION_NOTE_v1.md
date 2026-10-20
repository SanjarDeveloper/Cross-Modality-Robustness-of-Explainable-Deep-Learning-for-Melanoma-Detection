# Stage 7D Interpretation Note v1 — 2026-10-04

The frozen `STAGE7D_REPORT.md`, its run outputs and the thesis draft are unchanged. This note qualifies the E4.5 reference value.

1. **The ~10% overlap reference is not a matched null model.** About 0.10 is the expected top-10% overlap for two **independent, uniformly random** top-k pixel sets. It is **not** a spatially matched null model for these explanations. Grad-CAM maps are smooth and upsampled from a 7×7 grid, IG maps are sparse, and both depend on image content, so a structure-preserving null could give a different reference. Overlap of about 0.15 should therefore not be read as "only slightly better than chance" in any calibrated sense.
2. **Low agreement does not identify the correct method.** Low measured pixel agreement does not establish which method, if either, is correct, faithful or clinically meaningful.
