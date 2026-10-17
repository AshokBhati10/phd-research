# Person 4 — Frozen Integration Protocol

Status: **FROZEN** for the P1+P2+P3 integration and final evaluation.
P4 is read-only with respect to P1/P2/P3; every output traces to a registered artifact.

## 1. Sources (registered in `code/common_p4.py`, mirrored in `metadata/sources.json`)

| Person | Techniques | Results file | Prediction files |
|---|---|---|---|
| P1 | No Balancing (baseline), Random Oversampling, SMOTE, ADASYN | `P1_Basic_Oversampling/11_Audit_2015_2024_Rerun/07_combined_results_audit/combined_results_2015_2024.csv` | `03–06_*_audit/test_predictions_2015_2024.parquet` |
| P2 | Borderline-SMOTE, KMeans-SMOTE, SVM-SMOTE†, SMOTE-N† (+ baseline *replication*) | `P2_Advanced_Oversampling/results/results_person2.csv` + `results_baseline_no_balancing.csv` | `results/test_predictions_*.parquet` (5) |
| P3 | Tomek Links, ENN, SMOTE-Tomek, SMOTE-ENN | `P3_Advanced_Resampling/results/results_person3.csv` | `results/test_predictions_*.parquet` (4) |

## 2. Canonical rules

- **Canonical table:** 12 techniques (baseline + 11 resampling strategies) × 3 models = 36 rows.
- **Baseline policy:** P1's corrected No-Balancing is the single baseline row. P2's identical
  baseline run is replication evidence (verified: LR/RF bit-identical, XGB within cross-machine
  tolerance) and is never double-counted.
- **Naming/canonical order:** No Balancing; A. Basic oversampling (P1): ROS, SMOTE, ADASYN;
  B. Advanced oversampling (P2): Borderline-SMOTE, KMeans-SMOTE, SVM-SMOTE†, SMOTE-N†;
  C. Cleaning & hybrid (P3): Tomek Links, ENN, SMOTE-Tomek, SMOTE-ENN.
- **Metric precision:** displayed to 4 decimals in the report, 10 decimals in CSVs (source-native).
- **Timing excluded** from all comparisons (machine-dependent).
- **Dagger (†)** flags any technique trained on the 50k subsample; conclusions never rest on dagger rows.

## 3. Fairness register (must accompany every comparison)

Identical across persons: dataset/scope, frozen split + fingerprint `70e180c6…`, features,
classifiers, metrics, threshold 0.5, seed 42, train-only resampling.
Deviations: ① 50k subsample (†); ② KMeans-SMOTE non-default params; ③ SMOTE-N categorical
generation; ④ SVM-SMOTE `out_step` extrapolation; ⑤ XGBoost cross-machine nondeterminism
(~2e-3 noise floor for XGB-only); ⑥ P3 staged SMOTE-ENN (net accounting); ⑦ float32 vs
float64 proba storage; ⑧ timings excluded.

## 4. Verification gates

- **Gate 1 (`01_verify_inputs.py`):** re-derive test fingerprint from P1's splits; assert all
  13 prediction files share `y_true`; recompute all metrics/CM counts from parquets and match
  every source CSV row (tolerance 1e-6); verify P2 baseline replication.
- **Gate 2 (`07_validate_p4.py`):** combined table integrity, 198-test statistical matrix with
  sane Holm adjustment, six figures (png+pdf), workbook sheets, report embeds generated tables,
  and git-clean P1/P2/P3 folders.

## 5. Statistical methods

- Paired **McNemar** per classifier over all C(12,2)=66 technique pairs (198 tests), on
  identical test rows; **Holm-Bonferroni** within each classifier; effect sizes
  (accuracy/recall differences) reported beside every p-value.
- **Paired bootstrap** (500 resamples, seed 42) 95% CIs for champion-vs-rival ROC/PR-AUC gaps.

## 6. Outputs

`results/` (combined_results.csv, family_summary.csv, winners_tradeoffs.csv, mean_ranks.csv,
resampling_effects.csv, statistical_tests_mcnemar.csv, statistical_tests_summary.csv,
auc_bootstrap_cis.csv, input_validation.json, validation_report.json, report_tables/),
`figures/` (6 × png+pdf), `workbook/Person_4_Integration_Workbook.xlsx` (11 sheets),
`report/Person_4_Integration_Report.md`, `metadata/` (sources.json, environment.json).

## 7. Reproducibility

Same environment as P1/P3 (pinned in `P2_Advanced_Oversampling/code/requirements.txt`;
versions in `metadata/environment.json`). Any rerun must pass both gates before the
report/workbook are considered valid.
