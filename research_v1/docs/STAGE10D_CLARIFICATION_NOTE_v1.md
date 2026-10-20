# Stage 10D Clarification Note v1 — 2026-10-10

This note clarifies Stage 10D. The following are all unchanged:
- `docs/STAGE10D_REPORT.md`;
- `runs/stage10d/thesis_results_rq4_simulated_arm.md` (and its `.sha256`);
- the run outputs.

## 1. Which estimates are Hodges–Lehmann

**Hodges–Lehmann (HL)** means the one-sample pseudomedian: the median of Walsh averages i ≤ j (`rq4_stats.hodges_lehmann`). These values are HL estimates:
- all faithfulness values in Table S1 and in the report's "Main findings" (Δ_ins and Δ_del levels);
- all paired changes in Table S2 (ΔΔ);
- the IG − Grad-CAM differences.

Each comes with a 95% lesion-cluster bootstrap percentile interval for that HL.

**Medians and means are also reported** in the CSV files (`median`, `paired_change_median`). They are not HL.

**Localisation values are not HL.** IoU is the median area-matched IoU; pointing is a proportion (mean of hits); explainer − control differences are medians (IoU) or means (pointing).

## 2. Near-zero advantage is not equivalence to random

For IG at blur 1.0, Δ_ins has HL −0.000 [−0.002, 0.001]. The report and thesis describe this as "indistinguishable from random". The correct reading is narrower:

- the point estimate is close to zero;
- the interval is narrow and includes zero.

This does **not** establish equivalence to random maps. No equivalence margin was prespecified, and no equivalence analysis was performed. Preferred wording: "the insertion advantage over random was near zero (HL −0.000, 95% CI −0.002 to 0.001)".

## 3. Two kinds of eligibility counts

The Stage 10D tables use two different populations, and their counts are not interchangeable:

| Population | Definition | Where used | Example: Grad-CAM, blur 1.0 |
|---|---|---|---|
| **Condition-only eligible** | Valid map and random control **at that condition** | Table S1, `faithfulness_by_condition.csv`, `populations.csv` | n = 131 |
| **Baseline-joint eligible** | Valid map and random control **both at severity 0 and at the condition** | Table S2, `change_from_s0.csv`; equals the Stage 10C `jointly_eligible_with_random_control` | n = 128 |

For IG at blur 1.0 the counts are 340 (condition-only) and 335 (baseline-joint).

**Consequence:** the change from severity 0 must be read from Table S2. Subtracting Table S1 levels mixes populations.
