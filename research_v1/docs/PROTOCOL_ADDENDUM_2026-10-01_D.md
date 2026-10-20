# Protocol Addendum D — 2026-10-01: Bounded Real-Model XAI Pilot (Stage 3B)

This addendum is written and frozen **before** any inference on Derm7pt. Earlier documents are preserved.

It authorises one bounded XAI pilot with the selected v1 model:
- `runs/main_train/resnet50_s0_20260927T155924Z/ckpt_epoch8.pt`;
- SHA-256 `303d21e7…03dd`, from `best.json`.

It does **not** authorise:
- full XAI;
- RQ1 or RQ4 inference;
- thresholds, hypothesis tests or confidence intervals;
- retraining, seeds 1–2 or other architectures;
- m = 1024 or a different integration rule.

## D1. Conflict with the earlier plan (identified before inference)

| Earlier plan | This pilot | Why |
|---|---|---|
| 02_rev2 §6.6: convergence pilot with m ∈ {16, 32, 64, 128} on **40 pairs** | **24 pairs** (12 melanoma + 12 non-melanoma, 48 image references); IG candidates **m ∈ {128, 256, 512}** | User instruction for Stage 3B. Stage 2D2.1 already showed m = 128 failing criterion (c) on a diagnostic, so values below 128 are not re-examined |

The criteria (a)–(c) themselves are unchanged. The 24-pair pilot is smaller than the 40 pairs planned, and this is a **documented deviation**: the m decision rests on 48 references.

## D2. Disclosure

Derm7pt images are inspected here to choose numerical XAI settings. Derm7pt is therefore **no longer a pristine, untouched confirmatory set** for the XAI settings.

To limit this:
- no model weights, thresholds, controls or hypotheses are tuned here;
- prediction performance is not computed;
- the pilot's attribution scores are not findings.

## D3. Frozen sample

**Source:** the frozen pair manifest `manifests/v1/xai_derm7pt_v1.csv` (498 pairs).

**Selection:**
- Within each label, sort by `(stable_seed("stage3b_xai_pilot_v1", pair_id), pair_id)` and take the first 12.
- Both modalities of every pair are used.

**Membership check:** each record must be in the Derm7pt primary cohort (`in_derm7pt_primary = 1`, both modalities, consistent label and `pair_id`).

**No exclusion** is made because of predictions, map appearance or eligibility failures.

The pair list and the resolved image-reference keys are persisted before inference: `configs/xai_pilot_stage3b_v1_pairs.csv` and `configs/xai_pilot_stage3b_v1_refs.csv`.

## D4. Fixed computation

| Item | Setting |
|---|---|
| Preprocessing | `prep_v1_cached256` |
| Precision | `xai_precision_v1` |
| Grad-CAM | `layer4` |
| Baseline | Blurred, k = 57, σ = 9.5 |
| IG | Riemann midpoint rule, `internal_batch` 16, m ∈ {128, 256, 512} on all 48 references |
| Curves | `curves_v1`, S = 32, provisional `eval_batch` 11 |
| Controls | `controls_v1` (frozen spec) |

## D5. IG decision rule (criteria from 02_rev2 §6.6, unchanged; denominators fixed now)

For candidate m compared against 2m (candidates 128 and 256; **512 is never a candidate**, because 1024 is not run):

**(a) Completeness**
- Failure rate at m is ≤ 5%.
- The denominator is **all 48 references**.
- A non-finite completeness result counts as a failure.

**(b) Relative error**
- The 95th percentile of e_rel at m is ≤ 0.20, using the numpy linear percentile.
- It is computed over references where e_rel is defined.
- The number of undefined values is reported.
- If no e_rel is defined, (b) is UNRESOLVED.

**(c) AUC stability**
- The median |AUC(m) − AUC(2m)| is < 0.005, for insertion **and** deletion separately.
- It is computed over references whose IG map is eligible for the primary analysis (finite, non-constant and completeness pass), with curve status `ok` and endpoint status `pass`, **at both m and 2m**.
- The count eligible at both m values is reported next to every median, and it is never presented as 48.
- If no reference is eligible at both m values, (c) is UNRESOLVED.

**Decision**
- The candidate passes only if (a), (b), (c-insertion) and (c-deletion) all pass on the **pooled** 48 references.
- Results by modality and by label are descriptive only.
- The selected m is the **smallest** candidate that passes.
- If neither candidate passes, the m decision is **UNRESOLVED**, and m = 256 controls may be computed only as a labelled diagnostic.

An all-zero positive-evidence (ReLU) IG map is **not** a completeness failure, but it is ineligible for the primary map analysis.

## D6. Batch-equivalence check

**Subset**, fixed now: the first 2 melanoma and first 2 non-melanoma pairs in the frozen pilot order (the rank key above), giving 8 references.

**Maps compared:** Grad-CAM and IG at m = 128, 256 and 512, evaluated at `eval_batch` 11 versus 1. Attribution batching is unchanged.

**Pass condition:** for every compared eligible map:
- both endpoint statuses pass (tolerance 1e-4);
- |ΔAUC| ≤ 1e-3 for insertion and deletion.

Ineligible maps are reported and not compared.

**If the check fails:** all pilot curves are evaluated at `eval_batch` 1. Tolerances are not relaxed.

## D7. Controls and paired eligibility

**Controls**
- The four control types are computed for Grad-CAM and for the selected IG m (or the m = 256 diagnostic).
- Mismatch assignments use groups of (label, modality, explainer, condition = none) with case = `pair_id`. They are persisted before any control curve is evaluated.
- Invalid donors keep their planned assignment and are marked unavailable.

**Explainer pair eligibility:** the explainer map is primary-eligible with curve `ok` and endpoint `pass` in **both** modalities.

**Control contrast eligibility:** the explainer pair is eligible, and that control's score is available in **both** modalities.

**No substitution:**
- a single modality is never dropped;
- no case is resampled;
- the 5 random controls form one within-image mean, not 5 observations.

## D8. Bounds

- The pilot is bounded to 30 minutes of wall-clock time.
- On a timeout, or on an unresolved numerical or resource failure, the run stops and partial diagnostics are preserved.
- No hypothesis tests, CIs, Holm correction or research conclusions.
