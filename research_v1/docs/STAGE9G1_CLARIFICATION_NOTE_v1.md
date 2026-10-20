# Stage 9G1 Clarification Note v1 — 2026-10-04

This note corrects and qualifies two Stage 9G1 texts. Both files are kept unchanged:
- `docs/STAGE9G1_REPORT.md`;
- `runs/stage9g1/thesis_results_rq3_heavy_unperturbed.md` (and its `.sha256`).

## 1. Correction: the sensitivity-gap narrowing count is 8/9, not 7/9

Source: the Derm7pt PRIMARY rows `gap_sens_derm_minus_clin` in `runs/rq3_heavy_eval_9g1/20261004T152428Z_2cea4004_64df5fc0/matched_table.csv`, read programmatically in Stage 9G2.

**Change in the gap, heavy minus standard** (negative = narrower)

| Arch | s0 | s1 | s2 |
|---|---|---|---|
| ResNet-50 | −0.157 | +0.197 | −0.129 |
| EfficientNet-B3 | −0.004 | −0.357 | −0.273 |
| DenseNet-121 | −0.016 | −0.016 | −0.072 |

The gap narrowed in **8 of 9** models; only ResNet-50 s1 widened. Some of the narrowings are very small (−0.004, −0.016, −0.016), and their intervals include zero.

The report's consistency table (2 + 3 + 3) was correct. The interpretation paragraph in `STAGE9G1_REPORT.md` and the Results section of the thesis text both said "seven", which is wrong.

## 2. The narrowing is real, but it is not evidence of robustness

The narrowing is a correct measurement, not a false result. It went together with large specificity losses: dermoscopic specificity fell in 9/9 models and clinical specificity in 7/9. It therefore reflects a shift of operating points under each model's own threshold.

The threshold-free AUROC gap did not narrow consistently:
- 3/3 narrower for EfficientNet-B3;
- 0/3 for DenseNet-121;
- 1/3 for ResNet-50.

## 3. ECE scope

The ECE statements in Stage 9G1 ("higher dermoscopic ECE for every heavy model") are specific to:
- Derm7pt PRIMARY;
- the `ece_pos_v1` definition (bins and class used there).

They do not establish general miscalibration on other cohorts, or under other ECE definitions or recalibration.
