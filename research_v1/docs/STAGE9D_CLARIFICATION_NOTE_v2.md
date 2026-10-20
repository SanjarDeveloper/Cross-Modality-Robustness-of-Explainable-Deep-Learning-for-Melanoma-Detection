# Stage 9D Clarification Note v2 — 2026-10-04

**This note supersedes v1.** `STAGE9D_CLARIFICATION_NOTE_v1.md` is kept byte-for-byte; its SHA-256 is recorded in `runs/stage9e/clarification_note_v1_sha256.txt`. v1 reported 301 vs 158 images/s as the standard throughput, which was wrong: 158 is the heavy recipe's second repeat. The frozen `STAGE9D_REPORT.md`, Addendum K, the spec and the `aug_heavy_v1` recipe are unchanged.

1. **Provenance**
   - The legacy transform order and explicit parameters and probabilities were preserved.
   - The implicit defaults were resolved for the installed Albumentations 2.0.8.
   - API keyword compatibility (for example `quality_range`) does not establish the exact historical environment or its implicit defaults.
2. **Withdrawn claim.** The Stage 9D statement that the GaussNoise default is "about 10–30 times stronger" is withdrawn. It lacked an exact historical version, source, quantity and calculation. The frozen recipe uses std 0.2–0.44 of the maximum, and the visual review showed strong noise when that transform was selected.
3. **Timing, verified against `runs/aug_heavy_review_9d/20261004T113020Z_fac166c0_7c72e964/review_and_timing.json`:**

   | Recipe | Images/s, repeats 1 and 2 | Seconds, repeats 1 and 2 |
   |---|---|---|
   | `aug_std_v1` | 212.9, 301.3 | 2.405, 1.699 |
   | `aug_heavy_v1` | 154.3, 158.2 | 3.318, 3.236 |

   - **Mean-time ratio** heavy/std = **1.597**. The first standard repeat includes warm-up.
   - **Second-repeat (warmed) throughput ratio** = 301.3 / 158.2 ≈ **1.91**.
   - Both figures cover augmentation and DataLoader only. **Neither is a total training slowdown.** The measured training cost is in `STAGE9E_REPORT.md`.
