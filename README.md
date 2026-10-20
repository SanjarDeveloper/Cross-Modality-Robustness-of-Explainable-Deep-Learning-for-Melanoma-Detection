# Cross-Modality Robustness of Explainable Deep Learning for Melanoma Detection

Can dermoscopy-trained CNNs generalise to smartphone images — and do their
Grad-CAM explanations stay honest when they do not?

ITPU M.Sc. International Research Project · Sanjar Bakhtiyorov · advisor Leszek Siwik

---

## Status

**2026-09-20 safety update:** the earlier results below are legacy evidence, not
validated resume markers. Do not delete or overwrite them. The next
3 architectures x 2 strategies x 3 seeds, second explainer, XAI controls, CMRS
and CMEX package remains a proposal, not an approved experimental protocol.
No full training campaign is authorised by this update.

Week-1 and Week-2 deliverables are complete and the **Week-2 decision gate has
passed**: degradation is measurable on both real and simulated modality shift.
First results for RQ1, RQ2 and RQ4 (the core question) are in `results/`.

| Deliverable | State |
|---|---|
| Label mapping + leakage-safe splits | done, validated |
| Frozen dermoscopic baseline (ResNet-50) | done — AUROC **0.917** |
| Modality-Shift Simulator + verification | done, 5/5 factors verified |
| RQ1/RQ2 real + simulated degradation | done (1 architecture) |
| RQ4 faithfulness-under-shift + tests | done (1 architecture) |
| CMEX demo app | written, runnable |
| RQ3 architectures / RQ5 fairness | not started |

## Headline results (ResNet-50, trained on HAM10000 only)

**Baseline ceiling** — held-out dermoscopy: AUROC 0.917, sensitivity 0.965 at the
frozen operating point (threshold 0.125, set once at 95% sensitivity on validation).

**RQ1 — real modality shift** (Derm7pt, 1011 lesions photographed twice, 252 melanomas):

| | dermoscopy | clinical (phone-like) |
|---|---|---|
| melanoma sensitivity | 0.948 | **0.802** |
| AUROC | 0.817 | **0.688** |

Drop = **0.147 (14.7 percentage points)**, 95% CI [0.095, 0.196].
DeLong p = 3.8×10⁻⁹; McNemar p = 0.016 tests correctness on **all paired
cases**, not melanoma-only sensitivity.
This closely reproduces the 95.3% → 75.3% reported in the literature.

**RQ2 — which acquisition factor hurts most** (worst-case sensitivity loss):

| rank | factor | worst drop | 10-percentage-point-loss threshold |
|---|---|---|---|
| 1 | blur | −0.641 | severity **0.22** |
| 2 | scale | −0.241 | 0.45 |
| 3 | illumination | −0.224 | 0.72 |
| 4 | nuisance | −0.153 | 0.84 |
| 5 | compression | −0.018 | never |

Focus blur is by far the most damaging, and it breaks the model earliest.
JPEG compression is essentially harmless to sensitivity. Illumination is the
exception worth noting: modest sensitivity loss but the **largest AUROC loss
(−0.269)** — it degrades ranking more than thresholded detection.

**RQ4 — observed insertion-score loss on the enriched paired subset**

Legacy insertion score (reported as AUC; higher is the chosen convention),
400 paired cases selected as 200 melanoma + 200 non-melanoma:

| | dermoscopy | clinical | change |
|---|---|---|---|
| all | 0.330 | 0.198 | −40% |
| **melanoma** | 0.450 | 0.226 | **−50%** |
| non-melanoma | 0.211 | 0.169 | −20% |

Retention under shift: sensitivity 0.86, AUROC 0.77, faithfulness (all) 0.60,
**faithfulness (melanoma) 0.50**.

Statistical support (`results/rq4_tests_*.json`, n=400, 200 melanomas):
- faithfulness drops under shift — Wilcoxon p = 1.3×10⁻²², d = −0.56
- the drop is **5.4× larger for melanoma than non-melanoma** — Mann-Whitney
  p = 3.2×10⁻¹⁵, d = −0.84
- explanation drift is associated with misclassification — ROC-AUC 0.66
  (in-sample; an association, not yet a validated predictor)

The insertion-score loss is larger in the melanoma subgroup. This is not yet
proof of general explanation unfaithfulness: Grad-CAM targets the melanoma
logit for both classes, scores depend on model confidence, no random-map
controls exist, and deletion scores move in the opposite quality direction.

**CMRS** for the real dermoscopy→clinical shift: **72.8/100 (amber)** — from 2 of
4 axes: sensitivity and insertion AUC on the XAI subset. Localisation is absent
because Derm7pt has no masks; calibration is absent because the XAI rows do not
record ECE. This is an exploratory composite, not a validated clinical readiness
score; its colour cutoffs are not clinical decision thresholds.

Binary labels mean **melanoma vs non-melanoma**, not malignant vs benign.
The negative class includes other cancers. Legacy JSON keys such as `ben` and
`melanoma_vs_benign_drop` are retained for compatibility only.

**Endpoint decisions (2026-09-20):** the primary degradation threshold remains
an absolute **10 percentage-point** sensitivity loss (95% to 85%), not a
relative 10% reduction (95% to 85.5%). The user delegated this choice and
approved excluding malignant-vs-benign DDI results from primary fairness
conclusions: a melanoma model is not a general malignancy detector. Those
legacy metrics may be retained only as separately labelled off-target
exploratory results. Melanoma-by-tone results remain descriptive and
underpowered; neither endpoint establishes a causal skin-tone effect.

## Safe campaign execution and resume

```powershell
python .\code\campaign.py --smoke --archs resnet50 --strats standard --run-dir .\runs\smoke\phase1-final-20260920
python .\code\campaign.py --smoke --archs resnet50 --strats standard --run-dir .\runs\smoke\phase1-final-20260920 --status
python -m unittest discover -s .\code -p test_campaign.py -v
```

The smoke profile uses one epoch, a two-class training subset of 300, evaluation
subsets of 200, 40 XAI pairs and 8 deletion/insertion steps. These are software
rehearsals, **not scientific results**. Prefix-based subsets were unsafe because
HAM metadata is diagnosis-ordered and the first 300 training rows contain no
melanoma; smoke now selects deterministic two-class subsets.

Phase-1 execution: the initial six-model smoke passed **44/44** stages under
`runs\smoke\phase1-20260920`. After the approved fairness separation and final
validation/plotting changes, a fresh ResNet-50 rehearsal passed **9/9** stages
under `runs\smoke\phase1-final-20260920`; its read-only resume check also passed
9/9. Eleven regression tests cover unknown-label rejection, failure blocking, missing/corrupt outputs,
configuration/dependency invalidation, cache mismatch and subset alignment.
The initial run is retained with its original source fingerprint; it is not
silently treated as a run of the revised source.

Every stage writes to a fresh `steps\<stage>\<attempt>\` directory containing
its own outputs and `process.log`. `summary-*.json` lists final output locations.
Interrupted, failed and stale attempts are preserved. `--force` makes new
attempts, never overwrites old ones. Smoke and experiment modes cannot share a
run configuration. Existing `results\`, `checkpoints\` and `logs\` are untouched
and never automatically adopted as complete runs.

A stage resumes only after its completion receipt, output SHA-256 checksums,
artifact schema, expected rows/conditions, model tag and configuration match.
Source, installed package versions, dataset and cache contents are fingerprinted;
upstream receipt changes invalidate dependent stages. A zero process exit alone
is insufficient. Failures return a nonzero campaign exit and block dependent
stages while independent model branches may continue. Resume restarts incomplete
stages, **not an interrupted training epoch**. `--status` is read-only and returns
nonzero when the requested campaign is incomplete.

`analyse.py` and `figures.py` are explicit dependency stages. Paired results are
model-tagged, figures are stored under `results\figures\<tag>\`, and missing
inputs are errors, not successful skips. Untagged legacy `derm7pt_paired.json`
is not used automatically because its model provenance is ambiguous. Dataset
and cache paths remain separate from the output root (`MELANOMA_RUN_ROOT`).
Cache metadata must match requested record order, labels and lesion IDs; a
seed-0 HAM cache is rejected for a different split rather than silently reused.

Bare `campaign.py` does not start training. Non-smoke work additionally requires
`--execute` and a distinct run directory **after scientific protocol approval**.
The current runner supports the historical seed-0 matrix only; this does not
implement or approve the proposed three-seed study. Legacy fairness is exercised
as an explicitly off-target smoke diagnostic; its non-smoke inclusion is opt-in.

## Quick start

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install timm albumentations opencv-python-headless scikit-learn scipy pandas matplotlib streamlit
```

```powershell
python .\code\validate_labels.py
python .\code\campaign.py --smoke --archs resnet50 --strats standard --run-dir .\runs\smoke\phase1-final-20260920 --status
```

Use the isolated campaign rather than direct legacy training/evaluation commands
that default to the old output directories. CMEX is not yet integrated with
immutable run manifests and was not launched during phase 1.

## Design decisions that protect the result

**Train on HAM10000, never on Derm7pt.** This keeps all 1011 paired cases
(252 melanomas) out of training. The previous power claims (0.36/0.78/0.98)
are **not validated**: the legacy correlation adjustment fails to preserve
the stated sensitivity difference, and its melanoma-only population differs
from the implemented all-case McNemar test. `validate_labels.py` retains the
historical calculation with an explicit warning, not an adequacy decision.
A replacement power analysis requires an approved endpoint and joint model.

**Split HAM10000 by `lesion_id`, not `image_id`.** 1956 lesions have more than one
image; an image-level split leaks the same lesion into train and test and inflates
the baseline, which would make every degradation number wrong.

**The operating threshold is frozen.** Chosen once on the dermoscopic validation
split at 95% sensitivity, then reused unchanged for every condition. Re-tuning it
per condition would hide exactly the degradation being measured.

**Deletion/insertion uses a blurred baseline, not black.** A black patch is far
outside the data distribution and produces large probability drops that look like
faithfulness but are just distribution shift.

**Grad-CAM is hand-written for the single-logit head.** With one output, a library's
argmax-over-classes explains the wrong quantity. It also must not run under
autocast — fp16 gradients underflow to zero and yield silently blank maps.

**CMRS uses equal axis weights.** Fitting weights on the same 252 melanomas that are
then scored is circular and would overfit; data-derived weights belong in a
sensitivity analysis, not the headline metric.

**The simulator verifies itself.** Each factor must move a monotone image statistic
in a known direction. This caught four real bugs that would each have produced a
false "no degradation" finding — see `code/msim/simulator.py`.

## Limitations (current, honest)

- **Nine nominal Derm7pt pairs reference the same file**, including three
  melanomas (case numbers 800, 824, 827). There are 1002 different-path pairs
  with 249 melanomas; this is not a byte-level duplicate audit. No exclusions
  were applied. The retained legacy population remains 1011/252 pending an
  explicit protocol decision.
- **McNemar is not a sensitivity-loss test here.** All-case correct predictions
  rise from 477 to 521: melanoma true positives fall 239 to 202, while
  non-melanoma true negatives rise 238 to 319. Its p=0.016 reflects the former
  overall-correctness endpoint, not evidence for melanoma sensitivity decline.
  The sensitivity-loss CI (9.50 to 19.57 pp) includes 10 pp, so it does not
  establish a loss exceeding 10 pp at 95% confidence.
- **HAM splits are lesion-safe, not proven patient-safe.** Seed-0 train/val/test
  contain 6938/1529/1548 images, with zero lesion-ID or path overlap. The 170
  melanoma test images represent 91 lesions; image-level bootstrap ignores
  this clustering. PAD's 52 melanoma images represent 36 patients. Clustered
  uncertainty and cross-dataset duplicate checks remain protocol work.
- **Clinical is not necessarily smartphone.** Derm7pt pairing controls lesion
  identity, not all acquisition/camera/site differences. PAD supplies the
  explicitly smartphone dataset.
- **RQ4 sampling and metric limits.** Real-shift XAI uses 400 enriched pairs
  (200/200), not natural prevalence or the full cohort. Simulated XAI uses
  170 melanoma + 230 non-melanoma HAM images. The legacy AUC is an arithmetic
  mean of step scores, not trapezoidal integration. Overall deletion falls
  0.12722 to 0.10298 (better under its convention) while insertion falls
  (worse). Endpoint-normalisation, controls, target choice and multiplicity
  need approval before broader claims.
- **Legacy reproducibility is incomplete.** Main per-image predictions and
  explanation scores were not saved. Aggregate consistency can be checked,
  but melanoma-only McNemar discordances cannot be recovered from them.
  The untagged paired file has no checkpoint/source hash, so it is not silently
  migrated into the new pipeline.
- **One architecture, one seed.** RQ3 (EfficientNet-B3, DenseNet-121, training
  strategies) is not yet run, so no architecture claim is supported.
- **Localisation is only measurable on simulated shift.** Derm7pt and PAD-UFES-20
  have no lesion masks, and XAI rows lack ECE, so real-shift CMRS uses 2 of 4 axes and
  `cmrs()` reports `complete: false`. Annotating ~200 Derm7pt clinical images
  (SAM-assisted) would close this.
- **PAD-UFES-20 has only 52 melanomas** (2.3%) — supporting evidence, not primary.
- **DDI has 21 melanomas, exactly 7 per skin-tone group.** RQ5 fairness can only be
  descriptive; it cannot support an inferential claim.
- **Drift→error is in-sample.** Needs a held-out split before being called a predictor.
- **One explainer.** Grad-CAM only; cross-method consistency (Score-CAM,
  Integrated Gradients) is not yet tested, so faithfulness findings could still
  be method-specific.

## Layout

```
code/
  labels.py           explicit diagnosis -> binary mapping (raises on unknown)
  validate_labels.py  Week-0 gate: label coverage, file existence, power analysis
  data.py             records, leakage-safe splits, torch datasets
  msim/simulator.py   Modality-Shift Simulator + self-verification
  eval/metrics.py     sensitivity/AUROC/ECE, DeLong, McNemar, bootstrap CIs
  eval/xai.py         Grad-CAM, deletion/insertion, IoU / pointing game
  train.py            frozen dermoscopic baseline
  run_eval.py         suites: real | sweep | xai
  rq4_test.py         paired tests + drift->error model
  analyse.py          thresholds, factor ranking, faithfulness, CMRS
  figures.py          publication figures
  cmex_app.py         CMEX Streamlit demo
DATASETS/  derm7pt, ham10000, pad_ufes_20, ddi..., fitzpatrick17k
results/   per-suite JSON, report_*.json, figures/, simulator_check/
```

## Datasets (all local, counts verified)

| dataset | n | melanoma | role |
|---|---|---|---|
| HAM10000 | 10015 | 1113 | training + masks (all 10015 present) |
| Derm7pt | 1011 pairs | 252 | **primary** cross-modality test |
| PAD-UFES-20 | 2298 | 52 | real smartphone, supporting |
| DDI | 656 | 21 | skin tone, exploratory |
| Fitzpatrick17k | 16577 | — | CSV only, images not downloaded |

Derm7pt is CC BY-NC-ND and must not be redistributed — keep it out of any
public release.

## Phase-1 evidence records

The session artifact directory contains `audit_inventory_20260920T110132529170Z.json`
(434 files inventoried with sizes and SHA-256), and
`scientific_audit_20260920T110322860863Z.json` (dataset/split/cache checks,
checkpoint metadata and recomputation from saved aggregates, not fresh inference).
The inventory was captured after the smoke runs; its unchanged-file verification
applies to the audit interval, not to source files intentionally edited afterward.
Original source/document backups were saved before editing. Legacy scientific
results, checkpoints and logs were not overwritten.

Audit follow-through: DDI's unknown-diagnosis fallback was replaced with an
explicit allowlist; all 656 observed records retain their labels (21 melanoma).
The post-fix rehearsal and resume both passed 9/9, and all 11 regression tests
passed. `phase1-final-verification.json` records the final state and rehash
confirmation for all 18 legacy result/checkpoint/log files. The audit's historical
unknown-label finding is therefore fixed; its scientific caveats remain open.
