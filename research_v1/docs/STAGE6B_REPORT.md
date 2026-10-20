# Stage 6B Report — RQ1 Seed Spread (E1.1/E1.2) for ResNet-50 Seeds 0, 1, 2

Date: 2026-10-03. All work in this stage used `research_v1/.venv_model`.

**Run:** `runs/seed_spread_6b/20261003T171246Z_16d5722f_d2b87d3d/`. All four receipts verify (spec, threshold, infer, analyze).

**Stage records:** `runs/stage6b/`.

## Status

| Item | Result |
|---|---|
| Contract frozen before inference | `spec.json` (hashed). It records the checkpoint identities, inputs and manifest hashes, prep, precision, batch, order, threshold rule, tolerances, bootstrap seeds and the seed-0 artifact references |
| Selected checkpoints | Seed 1 **epoch 7** (`b70ff0e7…c781`); seed 2 **epoch 8** (`fecd74a2…5404`). Enforced against `best.json`, the file hash, the sidecar, the checkpoint epoch and `train_seed`. Both hashes also appear in the Stage 6A model cards |
| HAM VAL reproduction | Both seeds: identical IDs and labels, max \|Δp\| = 0.0 against the Stage 6A saved probabilities (tolerance 1e-6) |
| **Thresholds** (separate completed phase, before Derm7pt) | Seed 1: **0.11794479936361313** (152/160; specificity 835/1369). Seed 2: **0.0938335657119751** (152/160; specificity 795/1369). Seed 0: 0.17569…, unchanged. Each threshold is bound to its model hash (`bind_check`) |
| Derm7pt inference (seeds 1 and 2) | 2022/2022 rows each, exact coverage, finite p in [0, 1], model-specific pred re-verified, complete pairs. 9 same-file pairs with \|Δp\| = 0, within the 1e-5 tolerance fixed in the spec |
| Cohort membership | Identical across seeds 0, 1 and 2: same records, pairs, labels and primary flags |
| Seed-0 reuse | Stage 4B1 `results.json` and the Stage 4A predictions were verified by hash. A recomputation from the Stage 4A rows agrees **exactly** with the published b, c and CI, which are unchanged |
| Undefined replicates | 0 |
| Runtime (measured) | Threshold phase: 8.8 s (seed 1) and 5.8 s (seed 2). Derm7pt inference: 8.6 s and 7.5 s. Analysis: 1.4 s. **52 s** wall-clock for all phases |
| Tests | New `test_seed_spread.py` (5 tests). Relevant suites pass |
| Protected inputs | 0 changed, 0 removed |

**Why the full suite was not run:** no shared module changed.

## 1. Seed-spread table (PRIMARY: 1002 pairs, 249 melanoma pairs)

| Seed | Role | Epoch | Threshold | Sensitivity, dermoscopy | Sensitivity, clinical | Δ pp [95% CI] | b / c | Both detected / both missed | Exact McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| 0 | **Primary RQ1 test** | 8 | 0.1757 | 214/249 = 85.9% | 106/249 = 42.6% | **+43.4 [36.5, 50.2]** | 117 / 9 | 97 / 26 | 4.19 × 10⁻²⁵ |
| 1 | Descriptive spread | 7 | 0.1179 | 217/249 = 87.1% | 189/249 = 75.9% | +11.2 [5.6, 16.5] | 40 / 12 | 177 / 20 | 1.28 × 10⁻⁴ |
| 2 | Descriptive spread | 8 | 0.0938 | 231/249 = 92.8% | 171/249 = 68.7% | +24.1 [18.5, 30.1] | 64 / 4 | 167 / 14 | 5.87 × 10⁻¹⁵ |

**ALL cohort** (1011 pairs, 252 melanoma; sensitivity analysis):

| Seed | Δ pp [95% CI] | b / c |
|---|---|---|
| 0 | +42.9 [35.7, 50.0] | 117 / 9 |
| 1 | +11.1 [5.6, 17.1] | 40 / 12 |
| 2 | +23.8 [18.3, 29.4] | 64 / 4 |

**Range across the three runs**
- PRIMARY Δ: **+11.2 to +43.4 pp**, with all three in the same direction.
- ALL: +11.1 to +42.9 pp.
- **This range is not a confidence interval.**

**Bootstrap:** the same seeds as Stage 4B1 (20261001 for PRIMARY, +2 for ALL) are used for all models, so each model is resampled with identical pair draws. There is no other analytic difference from seed 0.

## 2. Interpretation

**Direction persisted; size did not.**
- All three runs showed higher melanoma sensitivity on dermoscopic than on clinical images.
- The size of the gap varied almost fourfold, mainly through clinical-image sensitivity (42.6% to 75.9%). Dermoscopic sensitivity ranged only from 85.9% to 92.8%.
- The primary model (seed 0) had the **largest** gap. Its estimate of 43.4 pp should therefore be presented alongside this spread, as one realisation of a seed-dependent quantity.

**What the intervals cover**
- The per-seed CIs reflect case resampling only, conditional on each fixed model and threshold.
- They do not capture training variability, which these three runs show is substantial.

**Not done**
- Seeds 1 and 2 are descriptive. Seed 0 remains the **sole** primary RQ1 test.
- No p-values are combined, no seed is chosen as a winner, no ensemble is formed, and seeds are not counted as cases.
- Three runs do **not** establish general robustness or a reliable estimate of the between-seed variance.

**Standing limits:** Derm7pt has prior exposure, there are no patient identifiers, the comparison is not causal, and the clinical images are not described as smartphone images.

## 3. Tests (`tests/test_seed_spread.py`, 5 tests)

| Area | What is checked |
|---|---|
| Threshold binding | A threshold bound to one model is rejected when applied to another model |
| Selected-checkpoint identity | Seed 1 at epoch 8 (latest, not selected) and swapped-seed identities are rejected |
| Summary rows | `per_seed` b, c, p and Δ equal the underlying `rq1_stats` results, and the replicate shape is 2000 × 3 |
| Threshold identity | A row with the wrong threshold identity is rejected |
| Seed-0 artifacts | The seed-0 threshold and checkpoint hashes are unchanged |

**Run-level checks:** the analysis enforces identical cohort membership across all three seeds, and the seed-0 recomputation must agree exactly with the published Stage 4B1 result.

## 4. Files

**New code, tests and config:** `src/rv1/seed_spread.py`, `tests/test_seed_spread.py`, `configs/seed_spread_6b_v1.json`.

**Run outputs:**
- `spec.json`;
- `threshold_s{1,2}.json` with sidecars;
- `ham_val_pred_s{1,2}.csv`;
- `derm7pt_pred_s{1,2}.csv`;
- `infer_verification.json`;
- `results.json`;
- `seed_spread_table.csv`;
- `bootstrap_replicates_s1_s2.npz`;
- `thesis_results_seed_spread.md`, which has its own sidecar;
- four receipts.

**Integrity:** 840 files before and after, with 0 changed, 0 removed and 0 added. The cache `.npy` files are unchanged. No `__pycache__` was created.

## 5. Remaining (not started)

- Full XAI (RQ4).
- RQ2 and RQ3.
- Other architectures.
- Any further seed analyses beyond E1.1 and E1.2.

**Stage 6B is complete.**
