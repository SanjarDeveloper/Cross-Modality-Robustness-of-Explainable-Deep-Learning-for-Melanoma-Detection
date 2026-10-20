# 04 — Four-Week Backlog and Acceptance Criteria

All hour figures are **estimates of active researcher work**, not measurements. Unattended compute is listed separately. Its values are also unknown until the pilot (T-08) measures them.

Budget: base plan ≈ **138 active hours** over 4 weeks (30–40 h per week). Week 4 is protected for the thesis and final verification.

Each task is small and independently reviewable. A task is "done" only when its artifacts exist **and** its acceptance checks have passed.

---

## Week 1 — Foundations and legacy reproduction (≈ 36 h)

### T-01 Environment inventory (read-only) — 1.5 h
- **Objective:** establish exactly which libraries are available.
- **Inputs:** current Python environment; `audit/requirements_observed.txt`.
- **Actions:** check torch/CUDA, timm, albumentations, pandas, matplotlib, pyarrow and captum with `find_spec` and `__version__`. Write the results to `research_v1/env/env_T01.json`.
- **Artifacts:** `env_T01.json`.
- **Accept:** every library needed by `03_…` §16 is listed as present or absent.
- **Stop / failure handling:** if CUDA is unavailable, stop and report it. Do not reinstall anything without approval.

### T-02 Reference verification — 4 h (in parallel with coding)
- **Objective:** verify the pivotal references against primary sources.
- **Inputs:** references [1]–[24] of the Research Plan.
- **Actions:** open the primary source of each pivotal reference ([4], [5], [6], [9], [12], [13], [14], [15], [17], [21]). Record the exact URL, the claim each one supports, and a status.
- **Artifacts:** `docs/REFERENCES_VERIFIED_v1.md`.
- **Accept:** every reference cited in the thesis is either marked verified with a URL or marked unresolved. Nothing is invented.
- **Stop / failure handling:** if a reference cannot be found, mark it UNRESOLVED and remove the claim it supports.

### T-03 Scaffold, configuration, seeds and provenance — 5 h
- **Objective:** build the foundation that all later runs depend on.
- **Actions:** implement `config.py`, `seeds.py` and `provenance.py`, with unit tests (stable_seed cross-process test; receipt hashing).
- **Artifacts:** `src/rv1/{config,seeds,provenance}.py`, `tests/test_seeds.py`.
- **Accept:** tests pass. A subprocess test shows `stable_seed` gives the same value in two processes with different `PYTHONHASHSEED` values.

### T-04 Cohort manifests v1 — 5 h
- **Objective:** freeze the cohorts used by every RQ.
- **Actions:** build the manifests listed in `03_…` §5, including the Derm7pt primary cohort (1002), the XAI manifests and the PAD RQ5 patient partition.
- **Artifacts:** `manifests/*.csv`, `.sha256` files, `manifest_card.json`.
- **Accept:** counts equal 6938/1529/1548; 1002/249; 1011/252; PAD 2298/52/36. Zero cluster overlap across splits. The PAD partition has melanoma patients on both sides.
- **Stop / failure handling:** any count mismatch → stop and investigate before continuing.

### T-05 Data and perturbation layer — 6 h
- **Objective:** make image loading and perturbations explicit and reproducible.
- **Actions:** implement `data.py` (the `prep_v1_cached256` path, with a cache-order check) and `perturb.py` (explicit rng; `scale_v1` with mask geometry).
- **Artifacts:** source files plus tests (bit-identity at severity 0; cross-process identity; mask centroid and area; legacy `verify()` monotonicity).
- **Accept:** all tests pass. A visual QC grid is saved and **looked at** by the researcher.

### T-06 Metrics and statistics library — 6 h
- **Objective:** provide the tested statistical functions for all analyses.
- **Actions:** implement sens/spec/AUROC, DeLong, melanoma-only exact McNemar, cluster bootstrap and Holm, with tests from `03_…` §14.
- **Accept:** all tests pass.

### T-07 Inference writer and legacy reproduction — 5 h active + compute (<1 h, UNV)
- **Objective:** confirm that the v1 pipeline reproduces the legacy numbers before it produces new ones.
- **Actions:** run `infer.py` with the legacy checkpoint on the HAM test set and all Derm7pt and PAD images, persisting per-image rows.
- **Accept:** reproduces legacy sensitivity and AUROC within ±0.005 (see `03_…` §14). This yields the **first valid melanoma-only E1.1 result for seed 0**. It is labelled "legacy model, v1 analysis".
- **Stop / failure handling:** if the values diverge, diagnose the preprocessing pathway (F-10). Spend at most 3 h. If still unresolved, report both values and continue with v1 as the reference.

### T-08a Training loop with full resume — 3.5 h
- **Objective:** make training reproducible and interruptible.
- **Actions:** implement `train.py` according to `03_…` §12, then run the kill/resume test on a small subset.
- **Accept:** the resume test passes within tolerance.

---

## Week 2 — Pilot, then main compute starts (≈ 36 h)

### T-08 Pilot and budget gate — 6 h active + ≈ 2–4 h compute (UNV)
- **Objective:** measure real costs before committing to the full matrix.
- **Actions:** for each architecture, train 2 epochs × 1 seed. Measure seconds per epoch, peak VRAM, and the largest batch that fits in 8 GB. Measure XAI seconds per image for Grad-CAM, IG and the controls on 20 pairs, and the IG completeness error at 16/32/64 steps.
- **Artifacts:** `runs/pilot_*`, `docs/PILOT_REPORT_v1.md` containing the measured times and a **projected compute total**.
- **Accept:** projected unattended compute fits the available calendar time (about 10 nights, or roughly 80 h of unattended GPU time, UNV). IG completeness median below 5% at the chosen step count.
- **Stop / failure handling:** if the budget is exceeded, apply the reductions in §R below, in the order given. **Never drop** the controls, the per-image persistence, the cluster bootstrap or the melanoma-only endpoint.

### T-09 Train the primary model seeds — 2 h active + compute
- **Objective:** produce the primary model and its seed spread.
- **Actions:** ResNet-50 standard, seeds 0, 1, 2.
- **Accept:** all three models have receipts. Validation AUROC is recorded.

### T-10 XAI module — 8 h
- **Objective:** implement the explanation pipeline specified in the protocol.
- **Actions:** Grad-CAM port, IG, the 4 controls and curves v1, with tests.
- **Accept:** coverage, constant-map, IG-linear and endpoint tests pass.

### T-11 RQ1 and RQ2 runs — 3 h active + compute
- **Objective:** produce the per-image data for RQ1 and RQ2.
- **Actions:** inference for all primary seeds on Derm7pt, PAD and HAM test, plus the RQ2 sweep (9 levels × 5 factors).
- **Accept:** receipts exist. Row counts equal manifest size × conditions.

### T-12 RQ4 runs — 3 h active + compute
- **Objective:** produce the per-image data for RQ4.
- **Actions:** the XAI manifest on seed 0 (Grad-CAM, IG and controls on both modalities), the simulated arm, and the seed 1–2 subset.
- **Accept:** receipts exist. Constant-map count and IG completeness are reported.

### T-13 RQ3 matrix training starts (unattended) — 2 h active
- **Objective:** start the remaining models of the matrix.
- **Actions:** EfficientNet-B3 and DenseNet-121 × {standard, heavy} × seeds, plus ResNet-50 heavy (15 remaining models), run overnight.
- **Accept:** each model has a receipt, or a logged failure followed by a resume.

### Methods writing — 12 h (in parallel)
- Draft thesis chapters 3 (Data) and 4 (Methods) from `02_…` and `03_…`, as described in `05_…`.

---

## Week 3 — Analysis, RQ5, demo (≈ 36 h)

### T-14 Analysis layer and statistics — 10 h
- **Objective:** turn per-image rows into the pre-specified results.
- **Actions:** `analysis/rq1..rq5.py` and `cmrs.py`. Generate tables and figures from the analysis JSON only.
- **Accept:** every number in the planned tables comes from `analysis_*.json`, and each JSON records the run IDs it consumed.

### T-15 RQ3 evaluation — 3 h active + compute
- **Objective:** evaluate the completed RQ3 models.
- **Actions:** RQ1 and RQ2 inference for each completed matrix model. Per-seed tables.
- **Accept:** consistency with the matrix actually completed. Missing cells are shown as missing, never imputed.

### T-16 RQ5 adaptation — 5 h active + compute (<1 h, UNV)
- **Objective:** run the limited smartphone adaptation experiment.
- **Actions:** fine-tune the primary model on PAD adapt-train, select on adapt-val, evaluate on PAD eval patients, Derm7pt clinical and HAM test. Run the descriptive DDI and PAD tone tables.
- **Accept:** no patient appears in more than one role (asserted). All results are labelled descriptive.

### T-17 CMEX demo — 8 h
- **Objective:** build the demonstration application from verified outputs.
- **Actions:** requires approval to install Streamlit (D-13); otherwise build a static HTML fallback. The demo is a viewer of stored results plus a single-image illustration.
- **Accept:** the consistency test from `03_…` §15 passes. No per-image CMRS is displayed.

### T-18 Results drafting — 10 h
- Results chapter drafted from the analysis outputs only. The placeholders from `05_…` are filled in.

---

## Week 4 — Protected: thesis completion and verification (≈ 30 h)

### T-19 Full verification pass — 6 h
- **Objective:** confirm that everything in the thesis is traceable and consistent.
- **Actions:** re-run the analysis from receipts in a fresh process. Diff the output against the thesis tables. Check every claim against `05_…` §4.
- **Accept:** zero unexplained differences.

### T-20 Thesis writing — Discussion, Limitations, Conclusion, Abstract — 16 h
### T-21 Correction of external documents — 3 h
- Applies the `01_…` §7 wording, the AI-use declaration (written by the researcher) and the ethics statement (from the institutional answer, D-14).
### T-22 Packaging — 5 h
- Pinned requirements, README_rv1, a reproduction command list, and a licence check (Derm7pt must not be redistributed).
- **No new experiments in Week 4.** A failed analysis is reported as a limitation, not rescued by new runs.

Total active hours ≈ 1.5+4+5+5+6+6+5+3.5 + 6+2+8+3+3+2+12 + 10+3+5+8+10 + 6+16+3+5 = **138 h (estimate)**.

---

## §R Reductions if the pilot exceeds budget (apply in this order)

1. RQ4 seeds 1–2 subset → dropped. Seed 0 only; stability is described as a limitation.
2. RQ3 heavy augmentation → ResNet-50 only (3 archs × standard × 3 seeds + ResNet-50 heavy × 3 = 12 models).
3. RQ3 seeds for EfficientNet-B3 and DenseNet-121 → 1 seed each (reported as a single fit, with no spread).
4. RQ2 severity grid 9 → 6 levels; simulated XAI arm → severities {0, 1.0} only.
5. XAI manifest 249+249 → 150+150 pairs (the power reduction is stated).
6. RQ5 → DDI/PAD descriptive tables only; adaptation → future work.

**Never reduce:** control maps, per-image persistence, the cluster bootstrap, melanoma-only E1.1, the primary cohort definition, mask geometry handling, or provenance receipts.

## Global stop conditions

- Legacy reproduction fails by more than 0.02 after 3 h of diagnosis → pause and bring the question to the researcher.
- Any cohort overlap or leakage between roles → stop all downstream work.
- The GPU is unavailable for more than 48 h → switch to reduction level 3 immediately.
- Any result that would require a new endpoint → record it as exploratory. Do not make it primary.
