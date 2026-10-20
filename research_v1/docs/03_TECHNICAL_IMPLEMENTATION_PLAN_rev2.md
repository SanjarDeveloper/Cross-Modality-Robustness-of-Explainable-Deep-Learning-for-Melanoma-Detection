# 03 — Technical Implementation Plan (rev2)

Status: plan only. Nothing is implemented yet.
This version supersedes `03_TECHNICAL_IMPLEMENTATION_PLAN.md`; the original is preserved.
Terminology and estimands follow `02_…_rev2.md`.

## 1. Principles

1. Legacy directories are read-only. research_v1 writes only under `research_v1/`.
2. The pipeline produces per-image outputs first. Aggregates come only from the separate analysis layer.
3. Every run records its versions, seeds, manifest hashes and code hash.
4. Outputs are immutable. A result counts as complete only if it has a receipt.
5. Dependencies are kept to a minimum. Nothing is installed until the relevant task needs it and the researcher approves.
6. Legacy reproduction (LR) and protocol-v1 analyses (V1) use separate configurations and separate output folders, and are never mixed. v1 is never tuned to match legacy numbers.

## 2. Directory layout (proposed)

```
research_v1/
  docs/                    planning documents
  env/                     read-only environment inventory
  manifests/               frozen cohort manifests + .sha256 + cards
  src/rv1/  config.py seeds.py provenance.py manifests.py data.py perturb.py
            models.py train.py infer.py xai/{explainers,curves}.py stats/ analysis/ report/
  tests/
  configs/  legacy_repro.json  v1_*.json
  runs/lr_<id>/   runs/v1_<id>/
  demo/
```

## 3. Components and dependency order (corrected)

| # | Component | Depends on | Notes |
|---|---|---|---|
| A1 | Environment inventory (read-only) | — | **Next task** |
| A2 | Manifests v1 + minimal validation | A1 | **Next task** |
| B1 | `config`, `seeds`, `provenance` | A1 | |
| B2 | `data` (preprocessing versions, joint image-mask transforms) | A2, B1 | |
| B3 | `stats` (pure functions + tests) | B1 | |
| C1 | `infer` + LR run | B2 | LR config only |
| C2 | V1 inference on the RQ1 cohorts | B2, C1 | |
| D1 | `perturb` (explicit RNG, `scale_v1`) | B1, B2 | |
| D2 | `xai` (Grad-CAM, IG, controls, curves) + correctness tests | B2 | **Must be finished before the pilot** |
| D3 | `train` with full resume + resume test | B2 | |
| E1 | **Pilot**: training throughput per architecture, XAI runtime, IG convergence | D2, D3 | Measures the implemented XAI, not a proxy |
| F | Main runs (RQ1–RQ5) | E1 gate | |
| G | `analysis`, `report` | B3, F | |
| H | Demo | G | |

## 4. Reuse vs rewrite

Same as rev1 §4, with these changes:

- `metrics.threshold_at_sensitivity` is replaced by `stats.threshold_highest_at_sens` (02_rev2 §1.1). The legacy function is used only in LR.
- The CMRS code is not ported (CMRS is deferred).

## 5. Manifest schema (v1)

The fields are the same as rev1 §5, with these additions:

- `stratum_rq5` (S1/S2/S3, PAD only)
- `rq5_role` (adapt_train / adapt_val / eval)
- `stratum_rule_version`
- `legacy_row_index` (row position in the legacy loader order, used by LR)

The full schema is listed in `NEXT_TASK_SPEC_STAGE2A.md`.

## 6. Per-image output schemas

**predictions** (one row per image × condition × model):
`run_id, run_kind(lr|v1), model_id, arch, strategy, train_seed, split_seed, ckpt_sha256, prep_version, infer_precision(fp16|fp32), image_uid, pair_id, cluster_id, dataset, modality, label_mel, factor, severity, perturb_seed, logit, prob, threshold, threshold_rule, pred_mel`

**xai_images** (one row per image × condition × map_type):
`… identifiers …, map_type (gradcam_pos | ig_pos | ig_abs | ig_pos_smooth | ctrl_rand_k | ctrl_centre | ctrl_sobel | ctrl_mismatch), target=mel_logit, baseline_id, steps, p_full, p_base, d, ins_auc_v1, del_auc_v1, ins_mean_v0, del_mean_v0, map_valid, invalid_reason, ig_m, ig_e_abs, ig_e_rel, iou, pointing_hit, mask_transformed, curve_ref`

Normalised scores and control aggregates are **derived in the analysis layer**. They are not stored as primary fields, so their definitions can be versioned.

## 7. Versions

`prep_version`, `aug_version`, `perturb_version`, `metric_version`, `xai_version`, `threshold_rule` (`legacy_first_roc_tpr95` | `v1_highest_t_img_sens95`).
`cmrs_version` is not used while CMRS is deferred.

## 8. Seed policy

| Seed | Rule |
|---|---|
| `split_seed` = 0 | HAM lesion split; PAD RQ5 partition |
| `train_seed` | `seed_everything`: Python, NumPy, Torch, CUDA. DataLoader `generator` seeded from `stable_seed(train_seed, "loader")` |
| **Augmentation (stateless design)** | For each sample: `aug_seed = stable_seed(train_seed, epoch, image_uid)`, then `compose.set_random_seed(aug_seed)` immediately before that sample is transformed (the method exists in Albumentations 2.0.8; its behaviour must be verified in B2 by a test that the same seed gives an identical output). The augmentation therefore depends only on (seed, epoch, image) and has no hidden state to restore. If the test fails, the fallback is to implement the transforms with an explicit `np.random.Generator` argument |
| `perturb_seed` | `stable_seed(split_seed, image_uid, factor)`, computed with SHA-256. Built-in `hash()` is never used |
| `control_seed` | `stable_seed("ctrl", k, image_uid)` |
| Sample / bootstrap seeds | Fixed constants stored in the config |
| Environment | The runner sets `PYTHONHASHSEED=0` for child processes |
| Determinism | `cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`. Warnings are logged. Bitwise GPU reproducibility is not claimed |

## 9. Insertion/deletion v1

Same as rev1 §9: blurred baseline, stable order by (−s, pixel_index), boundaries `e_j = round(j·N/S)`, curves of length S+1, trapezoidal area plus the v0 mean, endpoint assertions.

Change from rev1: invalid maps are **flagged, not dropped**. Eligibility is decided in the analysis layer, based on matched pairs (02_rev2 §6.4).

## 10. Attribution maps

- **Grad-CAM (primary):** ReLU(Σ_c w_c A_c), upsampled bilinearly to 224, computed in fp32. The map stored before normalisation is used for validity checks. Ranking is invariant to per-image min–max scaling.
- **IG positive (primary):** attr = IG(x, b, m), summed over RGB (signed), then ReLU. No smoothing.
- **IG absolute (secondary):** Σ_RGB |attr|.
- **IG positive smoothed (secondary):** Gaussian σ = 2 px applied to the IG positive map.
- **Completeness:** `e_abs`, and `e_rel` when |Δlogit| ≥ 0.1. A per-image failure is recorded as defined in 02_rev2 §6.6.
- **Convergence:** measured in the pilot at m ∈ {16, 32, 64, 128}.
- **Implementation:** Captum if T-A1 finds it installed; otherwise a small in-house implementation. The same tests apply in either case.

## 11. Image–mask geometry

Same as rev1 §11: photometric factors leave the mask unchanged; `scale_v1` transforms the mask with nearest-neighbour interpolation and pads with 0; a factor that cannot transform the mask declares `mask_supported=False`.

## 12. Checkpoint and resume

**Contents of each checkpoint:**
- model, optimizer, scheduler and GradScaler states;
- `epoch_completed`, `global_step`;
- best-so-far metric and best epoch;
- the **actual RNG states**: Python `random.getstate()`, `np.random.get_state()`, `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, and the DataLoader generator `get_state()`. Augmentation is stateless (§8), so no Albumentations state is needed;
- config, config hash, code hash, manifest hash, environment hash.

Checkpoints are written atomically: write to a temporary file, then rename. Resume is allowed only when all hashes match, and it restarts at the beginning of the next epoch.

**Resume test (predefined):**
- Both runs use the same configuration: total epoch budget E = 2 and cosine `T_max` = E × steps_per_epoch, on the same 256-image two-class subset.
- Run U trains for 2 epochs without interruption.
- Run R trains epoch 1, stops, restores from its checkpoint, and trains epoch 2.
- **CPU tolerance:** maximum absolute difference in parameters ≤ 1e-6; validation probabilities differ by at most 1e-6.
- **GPU tolerance:** maximum absolute difference in validation probabilities ≤ 1e-3; validation AUROC difference ≤ 1e-3; identical scheduler learning rate and `global_step`.

The observed differences are recorded. A failure blocks the main training runs.

## 13. Provenance and immutability

As in rev1 §13. In addition:
- `run_kind` (lr or v1) is recorded in `run.json`.
- The analysis layer refuses to mix run kinds in one table unless the table is explicitly a comparison between LR and V1.

## 14. Correctness tests (predefined)

| Check | Test and tolerance |
|---|---|
| Threshold rule | Synthetic example with ties at the k-th score: t equals that score; realised sensitivity ≥ 0.95; n₊ < 20 → "not estimable" |
| McNemar (melanoma only) | Known b and c; non-melanoma rows do not change the result |
| Cluster bootstrap | Clusters keep their identity. (a) Every replicate contains exactly K cluster draws, and all rows of a drawn cluster appear together. (b) Duplicating every row *within* its cluster, keeping the same `cluster_id`, leaves the distribution of per-cluster mean statistics unchanged under the same seed (maximum difference in CI bounds ≤ 1e-12). (c) Data generated with intra-cluster correlation ρ = 0.5: the cluster-bootstrap CI is wider than the naive row-bootstrap CI in ≥ 95% of 200 simulations |
| Undefined replicates | Forcing single-class resamples gives the correct count |
| DeLong | AUC equals the Mann–Whitney AUC to within 1e-12. Variance within 15% of a 5000-replicate bootstrap on a synthetic set |
| Holm | Textbook example |
| Trapezoid / coverage | Linear curve → exact value. For N ∈ {224², 300², 49} with S = 32, each pixel is replaced exactly once |
| **Grad-CAM (controlled)** | Toy network: one convolution whose single channel responds to a bright square, followed by global average pooling and a linear layer with positive weight. On an input with the square at a known position, the CAM peak lies inside the square and IoU(CAM > 0.5·max, square) ≥ 0.5. With the linear weight set negative, the positive CAM is all zero, so the map is flagged invalid |
| IG | Linear model: attribution = w·(x − b) exactly (error ≤ 1e-5). Completeness holds for a small nonlinear model at m = 128 (e_rel ≤ 0.01) |
| Random controls | Monte Carlo property, not a fixed invariant. Assumption: a model whose output depends on pixel values only through their mean (a mean-pool toy model), so random orderings are exchangeable. Under this assumption, E[ins(f)] = g(f·x̄ + (1−f)·b̄) and E[del(f)] = g((1−f)·x̄ + f·b̄), so the expected insertion and deletion areas are equal. Over 500 random maps, mean `ins_auc_v1` and mean `del_auc_v1` agree within 3 Monte Carlo standard errors. Not applied to the real CNN |
| Invalid-map flagging | Constant, NaN and IG-failure maps are flagged with the correct reason |
| Perturbation | Severity 0 is bit-identical. Two processes with different PYTHONHASHSEED values give byte-identical outputs. Monotone image statistics (legacy `verify`) hold |
| Mask geometry | `scale_v1`: mask centroid within 1 px of the expected position; area ratio within 5% of 1/z² |
| Augmentation seed | The same (seed, epoch, image) gives byte-identical augmented output in two processes |
| Resume | §12 |
| LR diagnostic | 02_rev2 §1.3 procedure. This is a report, not a pass/fail gate for V1 |

## 15. Demo / report consistency

- The demo displays stored V1 analysis values and per-image rows.
- The optional single-image illustration uses the `rv1.data` and `rv1.perturb` code paths and is labelled "illustration".
- No CMRS is displayed while it is deferred.
- Negatives are labelled "non-melanoma".
- Consistency test: 5 stored predictions are recomputed through the demo path, with maximum difference ≤ 1e-4 (same precision).

## 16. Dependencies

- Installed [FACT]: torch 2.11.0+cu128, timm 1.0.28, albumentations 2.0.8, scikit-learn 1.7.2, scipy 1.15.3. Pandas, matplotlib, pyarrow and Captum are to be checked in T-A1.
- Not used: MLflow, Quantus, pytorch-grad-cam, Docker. This deviation from the original plan is disclosed.
- Streamlit: a later implementation detail for the demo, handled at that stage. It is not a blocker now.
