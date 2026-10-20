# Demo test images

These examples are for interface demonstration, not performance evaluation.
Image 04 was deliberately selected to have a non-melanoma dataset label AND a non-melanoma model prediction. This selection is not random.

| File | Dataset label | Diagnosis code |
|---|---|---|
| 01_ISIC_0024351.jpg | Melanoma | mel |
| 02_ISIC_0024449.jpg | Melanoma | mel |
| 03_ISIC_0024317.jpg | Non-melanoma | nv |
| 04_ISIC_0024357.jpg | Non-melanoma | nv |
| 05_ISIC_0024337.jpg | Non-melanoma | bkl |

Non-melanoma does not mean benign. Previous images are preserved in previous_04/.
New image 04 model score: 0.00200363; threshold: 0.17569470.

## Added 2026-10-19
| File | Dataset label | Diagnosis code |
|---|---|---|
| 06_ISIC_0024352.jpg | Non-melanoma | nv |

Image 06 was deliberately chosen as a **false alarm**: dataset label non-melanoma, model score 0.94159
(Stage 4A saved prediction; threshold 0.17569). It shows that the model makes mistakes.
