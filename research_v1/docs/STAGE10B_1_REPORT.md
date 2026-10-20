# Stage 10B.1 Report — Bounded IG Readiness Extension (m = 2048 candidate, m = 4096 reference)

Date: 2026-10-08. Environment: `research_v1/.venv_model`.

**Run:** `runs/xai_sim_ext_10b1/20261008T163049Z_5ce62e92_abd19dad/` (48 MB).

**Stage records:** `runs/stage10b1/`.

**Status of this stage:** a **retrospective extension** motivated by the failed Stage 10B pilot.

**Not done:** the full simulated arm, E4.6, training, and any change to previous results, criteria or eligibility rules.

## Headline

| Gate | Result |
|---|---|
| **Gate 1** (m = 2048; criteria (a) completeness failure rate ≤ 5% and (b) p95 e_rel ≤ 0.20, pooled **and** every condition) | **READY.** Pooled: 2/220 failures (0.9%), p95 e_rel 0.020. Every condition: ≤ 1/20 failures and p95 ≤ 0.031 |
| **Gate 2** (m = 4096 reference; criterion (c) median \|ΔAUC\| < 0.005 on jointly eligible references; frozen Stage 10B rule, pooled **and** every condition) | **READY.** Pooled: 218/220 eligible, median \|ΔAUC\| ins 0.00037 / del 0.00013. Every condition: ≥ 19/20 eligible, maxima 0.00065 / 0.00028 |
| Decision | **m = 2048 meets the unchanged readiness rule for the 11 tested conditions.** m = 4096 is only a reference and is **not** claimed to be converged. No escalation beyond 4096. **The full arm was not launched** |

## 1. Versioned clarification (`docs/STAGE10B_CLARIFICATION_NOTE_v1.md`)

The Stage 10B report is unchanged; this note corrects it on three points:
1. **m = 1024 could never qualify** under the frozen rule. The saved table already showed 2/20 m = 1024 failures in both nuisance 0.5 and nuisance 1.0. Proposal A in Stage 10B was therefore wrong. The corrected next step is this stage: m = 2048 vs m = 4096.
2. **Units:** the 220 references are **20 images × 11 conditions**, not 220 independent cases.
3. **Matched-population localisation**, recomputed from the saved 10B records only (`runs/stage10b1/matched_localisation_10b.json`):
   - At s = 0, on 19 matched references (10 mel / 9 non-mel), centre IoU is **0.72** against IG512 **0.28**; the centre prior is higher in 19/19.
   - Under scale 1.0 (20 matched references), centre is 0.02 against IG 0.27.
   - Comparisons that include Grad-CAM reduce to 136 references from 18 images (42 non-mel), selected by model evidence.

## 2. Frozen extension (`spec.json`, hashed before any computation)

**Identical to the Stage 10B pilot:**
- the 20 images (no replacement) and the 11 conditions;
- image order and chunks (one per condition);
- the checkpoint (`303d21e7…`), preprocessing and precision (`xai_precision_v1`);
- the blurred baseline: all 220 SHA-256 values were re-checked against the 10B records;
- the IG rule (Riemann midpoint) and the internal batch of 16.

**Bound by hash:** the Stage 10B spec and all 11 chunk manifests, and the 10B source hashes (all verified unchanged).

**Criteria (unchanged, from `xai_pairs`):** fail rate ≤ 0.05; p95 e_rel ≤ 0.20; median \|ΔAUC\| < 0.005.

**Staging:** gate 1 runs m = 2048 only. Gate 2, with m = 4096 and the curves for both maps, was allowed only if gate 1 passed. A gate failure would exit with code 3, and m = 4096 would not be run.

**Reused, not recomputed:** Grad-CAM, all controls, localisation, and the IG m = 512 / 1024 results (Stage 10B).

**Tests:** `tests/test_xai_sim_ext.py`, 4/4:
- criteria (a)/(b) at the 1/20 boundary;
- missing references count as failures;
- **a pooled pass with any per-condition fail (or UNRESOLVED) stays NOT_READY**;
- a realistic pooled-pass / condition-fail case.

`test_xai_sim` 11/11 also passes. The full suite was not rerun, because no shared module changed.

## 3. Results

**Gate 1 (`gate1_table.csv`, `gate1_results.json`)**

| Group | Fails m = 2048 (mel / non-mel) | p95 e_rel | e_rel undefined |
|---|---|---|---|
| Pooled (220) | 2 (0 / 2) | 0.020 | 4 |
| none 0 | 0 | 0.012 | 1 |
| blur 0.5 / 1.0 | 1 / 0 | 0.008 / 0.005 | 1 / 0 |
| illumination 0.5 / 1.0 | 1 / 0 | 0.028 / 0.014 | 0 / 0 |
| compression 0.5 / 1.0 | 0 / 0 | 0.011 / 0.012 | 1 / 0 |
| scale 0.5 / 1.0 | 0 / 0 | 0.007 / 0.004 | 0 / 1 |
| nuisance 0.5 / 1.0 | 0 / 0 | 0.031 / 0.021 | 0 / 0 |

**The two m = 2048 failures**

| Reference | e_rel | e_abs |
|---|---|---|
| ISIC_0025753 \| blur 0.5 | 0.051 | 0.024 |
| ISIC_0030924 \| illumination 0.5 | 0.096 | 0.035 |

Both are non-melanoma, and both remain invalid maps; they are never repaired.

**Gate 2 (`gate2_table.csv`, `gate2_results.json`)**

Median \|AUC(2048) − AUC(4096)\| on jointly eligible references:

| Group | Eligible both | Median \|ΔAUC\| ins / del |
|---|---|---|
| Pooled | 218/220 | 0.00037 / 0.00013 |
| Per condition | 19–20 of 20 each (19 in blur 0.5 and illumination 0.5) | ≤ 0.00065 / ≤ 0.00028 |

All criteria pass, pooled and per condition. m = 4096 had 1/220 completeness failures (descriptive).

**Comparison with Stage 10B (pooled completeness failures)**

| m | Failures |
|---|---|
| 512 | 16/220 (7.3%) |
| 1024 | 8/220 (3.6%) |
| 2048 | 2/220 (0.9%) |
| 4096 | 1/220 |

**Caveats**
- With 20 references per condition, condition-level readiness is coarse: one more failure in any condition would have failed gate 1.
- Readiness is established only for these 20 images and 11 conditions.
- Completeness and curve agreement do not establish pixel-level map convergence.
- The full arm should still report its own completeness failure rate per condition and class.

## 4. Measured runtime (this stage only)

| Item | Measured |
|---|---|
| Gate 1 (IG m = 2048) | 2,110 s IG (35.2 min); **9.59 s per reference**; about 192 s per 20-image chunk; peak CUDA 1,598 / 1,980 MiB |
| Gate 2 (IG m = 4096 + curves for both maps) | 4,421 s IG (73.7 min) + 72 s curves. Chunks took about 384 s, except the **last (nuisance 1.0) at 575 s**. The cause is not identified; the GPU peaked at 80 °C, and no other job was found. Reported as measured |
| GPU temperature | 44–80 °C (222 samples) |
| Disk | 48 MB |
| Exit codes | gate 1 = 0, gate 2 = 0. Receipts verify |

## 5. Full-arm projection at m = 2048 (ESTIMATE, not measured; not launched)

**Workload:** 3,740 references (340 images × 11 conditions). Per-reference costs:
- IG m = 2048: 9.59 s (measured here);
- everything else: 1.97 s (measured in 10B: Grad-CAM, IG-map curves, 5 random, centre, Sobel, mismatch, localisation, perturbation).

**Total:** about **11.4–11.6 s per reference**, which gives about **11.9–12.0 h** of GPU time. The lower value assumes one IG-map curve set instead of the pilot's three.

**Disk:** about **3.2 GB**, using the 10B per-reference size (0.87 MB, which included the signed attribution).

**Assumptions:** the same GPU, no competing load and similar thermal behaviour. The last-chunk slowdown seen here suggests allowing a margin.

## 6. Integrity (`runs/stage10b1/integrity_comparison.json`)

- 4148 → 4150 files: **0 changed, 0 removed**.
- Added: `configs/xai_sim_ext_10b1_v1.json` and `docs/STAGE10B_CLARIFICATION_NOTE_v1.md`.
- The Stage 10B run, the report and the cache are unchanged.

## 7. Files

**Code, test and config:** `src/rv1/xai_sim_ext.py`, `tests/test_xai_sim_ext.py`, `configs/xai_sim_ext_10b1_v1.json`.

**Run dir:**
- `spec.json`;
- `ig2048_00`–`10/` and `ig4096_00`–`10/` (primary maps, records with completeness, eligibility and curves, manifests);
- `gate1_*` and `gate2_*` (JSON and CSV);
- the receipts.

**`runs/stage10b1/`:**
- `run_gates.bat` and `status.txt`;
- the phase logs;
- `gpu_samples.csv`;
- `matched_localisation_10b.py` / `.json`;
- the integrity files.

## 8. What remains

1. **Your decision to launch the full simulated arm at m = 2048.** It needs about 12 h, and the frozen 10B full-arm contract applies, with m = 2048 replacing the earlier candidate. The full arm must again report:
   - completeness failures per condition and class;
   - Grad-CAM all-zero rates;
   - matched-population localisation against the centre prior.
2. E4.6.
3. The deferred controls for the secondary IG variants.

**RQ4 is not complete. Stage 10B.1 stops here.**
