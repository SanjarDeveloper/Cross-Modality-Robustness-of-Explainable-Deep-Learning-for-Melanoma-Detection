# 04 — Four-Week Backlog and Acceptance (rev2)

Supersedes `04_FOUR_WEEK_BACKLOG_AND_ACCEPTANCE.md` (preserved).

**How to read the numbers.**
- All hours are **estimates of active researcher work**. None of them has been measured.
- **Compute** is listed separately and is unknown until the pilot measures it.
- The plan does **not** assume unattended GPU availability. Whether overnight runs are possible is a researcher constraint, checked in the pilot (E1).

**Budget.** Total active work ≈ **146 h**, within the 120–160 h envelope. Stage 1 and 1.1 are counted in this total.

---

## Week 0/1 — Already spent: Stage 1 + 1.1 planning — **8 h** (researcher reading and review time; estimate, not measured)

## Week 1 — Environment, manifests, foundations, introductory writing (≈ 36 h)

| ID | Task | Deps | Actions | Artifacts | Acceptance | Est. h | Stop / failure |
|---|---|---|---|---|---|---|---|
| T-A1 | Environment inventory (read-only) | — | See `NEXT_TASK_SPEC_STAGE2A.md` | `research_v1/env/` | Spec criteria | 1.5 | Spec |
| T-A2 | Manifests v1 + minimal validation | T-A1 | See spec | `research_v1/manifests/` | Spec criteria | 6 | Any count mismatch → stop and report |
| T-B1 | config / seeds / provenance + tests | T-A1 | Implement 03_rev2 §8 and §13 | `src/rv1/*`, tests | Cross-process `stable_seed` test passes | 5 | — |
| T-B2 | Data layer (preprocessing versions; augmentation-seed test) | T-A2, T-B1 | 03_rev2 §8 | `data.py`, tests | Augmentation-seed and cache-order tests pass | 5 | If `set_random_seed` is not deterministic → switch to the fallback design (+2 h) |
| T-B3 | Statistics library + tests | T-B1 | 03_rev2 §14 | `stats/`, tests | All tests pass | 6 | — |
| W-1 | **Introduction + Related Work drafts** | — | Write from verified sources only | thesis draft | Every citation marked verified or UNRESOLVED | 8 | — |
| T-R | Reference verification (pivotal refs) | — | Read primary sources; record URL and location | `REFERENCES_VERIFIED_v1.md` | No invented references | 4.5 | Unresolvable → remove the claim |

## Week 2 — Reproduction, XAI, training, pilot, methods (≈ 38 h)

| ID | Task | Deps | Acceptance | Est. h | Compute |
|---|---|---|---|---|---|
| T-C1 | Inference writer + **legacy reproduction (LR)** | T-B2 | Rows complete; difference report written per 02_rev2 §1.3 (this is **not** a pass/fail gate) | 4 | < 1 h (UNV) |
| T-C2 | **V1 RQ1 inference**, seed 0 (legacy checkpoint re-analysed under v1 rules) | T-C1, T-B3 | Per-image rows; same-file pairs give identical predictions | 2 | < 1 h (UNV) |
| T-D1 | Perturbation module (`scale_v1`, stable RNG) | T-B2 | 03_rev2 §14 perturbation and mask tests pass | 4 | — |
| T-D2 | **XAI module** (Grad-CAM, IG, controls, curves) | T-B2 | Controlled Grad-CAM, IG linear, coverage and invalid-flag tests pass | 9 | — |
| T-D3 | Training loop with resume + resume test | T-B2 | 03_rev2 §12 tolerances met | 5 | < 0.5 h |
| T-E1 | **Pilot** (after T-D2 and T-D3): training time and VRAM per architecture; XAI time per pair; IG convergence; confirm whether overnight runs are possible | T-D2, T-D3 | `PILOT_REPORT_v1.md` gives a measured compute projection and the chosen reduction level | 6 | 2–4 h (UNV) |
| W-2 | Data and Methods chapters | 02_rev2, 03_rev2 | Draft complete | 8 | — |

## Week 3 — Main runs, analysis, RQ5, demo, results (≈ 38 h)

| ID | Task | Deps | Acceptance | Est. h | Compute |
|---|---|---|---|---|---|
| T-F1 | Primary seeds 1–2; RQ1 + RQ2 runs | T-E1 | Receipts present; row counts = manifest × conditions | 3 | from pilot |
| T-F2 | RQ4 runs (matched-pair eligibility computed in analysis) | T-E1, T-C2 | Invalid-map and IG failure rates reported | 3 | from pilot |
| T-F3 | RQ3 matrix, at the pilot-selected size | T-E1 | Missing cells shown as missing | 3 | from pilot |
| T-F4 | RQ5 partition use, adaptation, evaluation | T-A2, T-D3 | Role disjointness asserted; primary and secondary contrasts reported separately | 5 | < 1 h (UNV) |
| T-G | Analysis and report layer | T-B3, T-F* | Every table number is traceable to an analysis JSON listing its run IDs | 10 | — |
| T-H | Demo (viewer; Streamlit installation handled at this point, or a static HTML fallback) | T-G | Consistency test passes | 6 | — |
| W-3 | Results drafting from analysis outputs | T-G | Placeholders in 05_rev2 filled in | 8 | — |

## Week 4 — Thesis completion and verification (≈ 26 h, plus a validity-repair reserve)

| ID | Task | Est. h |
|---|---|---|
| T-V | Full verification pass: re-run analysis from receipts; compare with the thesis tables | 5 |
| T-X | **Critical validity-repair reserve** | 5 |
| W-4 | Discussion, Limitations, Conclusion, Abstract | 12 |
| T-Z | Document corrections (01_rev2 §7); AI declaration finalised by the author; packaging and licence check | 4 |

Validity-repair reserve (T-X): if a known validity defect is found in Week 4, it is repaired and the affected analysis is re-run. **A result known to be invalid is never reported to protect the schedule.** If the repair cannot be finished, the affected result is withdrawn and the withdrawal is reported.

**Total:** 8 + 36 + 38 + 38 + 26 = **146 h** (estimate).

---

## §R Reductions if the pilot shows the budget is exceeded (apply in order)

1. Drop the RQ4 seed 1–2 subset.
2. Run RQ3 heavy augmentation for ResNet-50 only.
3. Train one seed each for EfficientNet-B3 and DenseNet-121.
4. Reduce the RQ2 grid to 6 levels, and the simulated XAI arm to severities {0, 1}.
5. Reduce the XAI manifest to 150 + 150 pairs.
6. Limit RQ5 to descriptive tables only; adaptation moves to future work.

**Never reduce:** control maps, per-image persistence, the cluster bootstrap, E1.1, cohort definitions, the matched-pair eligibility rule, mask geometry handling, or receipts.

## Global stop conditions

| Condition | Response |
|---|---|
| Overlap between manifest roles, or wrong cohort counts | Stop all downstream work |
| Resume or seed tests fail | Main training does not start |
| XAI correctness tests fail | No RQ4 runs |
| GPU unavailable for more than 48 h | Apply reduction level 3 |
| A new idea for an endpoint | Label it exploratory; it is never promoted to primary |
