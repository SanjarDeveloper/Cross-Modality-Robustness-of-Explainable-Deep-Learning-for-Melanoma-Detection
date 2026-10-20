# Stage 13B Report — Bounded Legacy Reproduction (protocol §1.3)

Date: 2026-10-10. Diagnostic appendix task, separate from all v1 scientific results. No training, tuning, XAI or new studies.

**Run:** `research_v1/runs/lr_13b/20261010T181416Z_lr13b/`.

**Stage records:** `research_v1/runs/stage13b/`.

**Appendix:** `research_v1/thesis_materials/APPENDIX_LEGACY_REPRODUCTION.md`.

**Versioned update:** `research_v1/thesis_materials/UPDATE_13B_COMPLETION_AND_LIMITATIONS.md`.

## 1. Read-only preflight

**Legacy artefacts found**

| Artefact | Detail |
|---|---|
| Checkpoint | `checkpoints/resnet50_standard_s0.pt` (`5400f5aa…`). It stores the arch, size 224, the **full-precision threshold 0.12517227232456207**, `val_auroc` and its test metrics |
| Historical aggregates | `results/real_resnet50_standard_s0.json` (Derm7pt ALL dermoscopy / clinical, PAD all, HAM test); `results/derm7pt_paired.json` (paired tests); `results/baseline_…json`; `results/report_…json` |
| Legacy code | `code/run_eval.py`, `data.py`, `eval/metrics.py`, `labels.py` |
| Environment evidence | The legacy logs show the global Python 3.10 site-packages. Historical library versions were not recorded |

**Amendment applied:** the v1 primary model is the newly trained seed-0 model (Addendum A1). The legacy checkpoint is used only here.

**Hashes:** all of the above are in `lr_spec.json`, which was frozen before inference. Known and unknown settings are recorded there and in the appendix.

## 2. Execution

| Attempt | Pathway | Result |
|---|---|---|
| **A, primary (frozen)** | Cached 256-px arrays, fp16, batch 64 | 5/30 checks within rule. Large count differences, e.g. HAM sensitivity 157 vs 164, McNemar b / c 158 / 169 vs 139 / 183 |
| **Diagnostic step 1** (identifiers / cache / order) | — | `validate_cache` passed. **The cache files were written about 3 h after every historical result and the checkpoint**, so the historical results could not have used them. Hypothesis H1: raw JPEG decoding |
| **B, `raw_fp16_b64`** | Legacy `LesionDataset` raw-JPEG pathway; only the input source changed | **30/30 exact**: all integer counts; AUROC and legacy ECE differences 0.0; identical p-values |

**Outcome:** the discrepancy is **explained**. The precision, batch and version steps were not needed. The investigation ended about 30 min into the 3 h time box. No setting search was performed. Every attempt is preserved, with its predictions and results, in the run dir and the diagnostic log.

**Runtime:** 42.6 s (A) and 50.3 s (B) of inference.

## 3. Checks

**Predictions (both attempts)**
- 5,868 rows: Derm7pt 1,011 + 1,011, PAD 2,298, HAM 1,548;
- finite probabilities in [0, 1];
- unique row indices per cohort;
- identical keys and labels across attempts.

**Legacy code:** hashes were verified before each attempt. The legacy `__pycache__` was untouched, still dated 13 Sep.

**Integrity:** 4,881 → 4,881 protected files, with the legacy `code/`, `results/`, `checkpoints/` and `logs/` in scope: **0 changed, 0 removed, 0 added**. The cache is unchanged.

**Environment:** the global interpreter was used read-only, because it is the one evidenced by the legacy logs and the legacy metrics need scikit-learn. No package was installed or modified.

## 4. Findings for the thesis

1. **LR is completed as a diagnostic appendix.** The legacy aggregates can be regenerated exactly from the legacy checkpoint and code, using direct JPEG decoding.
2. **Correction to the protocol's §1.3 assumption.** The historical pathway was **not** the cache. This settles the open STAGE1_1 question for these results.
3. **Preprocessing sensitivity (descriptive).** For the legacy model, the cache and raw pathways change per-image probabilities substantially (median |Δp| 0.028–0.067; up to 0.874). This is one reason, alongside the different models, thresholds and endpoints, why v0 and v1 numbers are not directly comparable. v1 is unaffected because it is internally consistent.
4. **LR neither validates nor invalidates v1** by numerical agreement alone.

## 5. Updates (versioned; earlier files unchanged)

`UPDATE_13B_COMPLETION_AND_LIMITATIONS.md` makes three changes:
- moves LR from **open** to **completed (diagnostic appendix)**;
- adds deviation 9 (historical pathway and interpreter);
- adds the preprocessing-sensitivity limitation.

**After this stage:** no open research items remain. RQ5 Secondary B is not estimable by rule, CMRS is deferred by the protocol, and the optional extensions remain optional.

**Stage 13B is complete. The thesis is not complete.**
