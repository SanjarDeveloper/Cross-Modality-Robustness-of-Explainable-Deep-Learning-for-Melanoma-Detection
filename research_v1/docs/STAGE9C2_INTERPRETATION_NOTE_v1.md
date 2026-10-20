# Stage 9C2 Interpretation Note v1 — 2026-10-04

The frozen `STAGE9C2_REPORT.md`, its run outputs and the thesis draft are unchanged. This note qualifies two statements.

1. **Blur's leading sensitivity loss is not equally certain in every model.** Blur had the largest observed worst-case sensitivity loss in all nine models, but its bootstrap ranking share varied:
   - 1.00 for ResNet-50 s0 and s1, and for DenseNet-121 s1 and s2;
   - 0.88 for ResNet-50 s2 and 0.97 for EfficientNet-B3 s2;
   - only **0.58 for EfficientNet-B3 s1**, where illumination was nearly tied (31.2 vs 30.6 pp).

   "Largest in all nine models" refers to the point estimates. It does not mean the ranking is equally certain everywhere.
2. **JPEG can cause substantial sensitivity loss.** JPEG compression did not reduce sensitivity in most models, but it produced worst-case losses of **17.1 pp (EfficientNet-B3 s1)** and **10.6 pp (EfficientNet-B3 s2)**, with 10-pp crossings at 0.90 and 0.99. The statement that compression "rarely reduced sensitivity" should keep these exceptions visible.
