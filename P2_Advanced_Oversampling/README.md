# Person 2 — Advanced Oversampling for Class Imbalance

**Scope.** Advanced oversampling on the corrected BRFSS 2015–2024 diabetes dataset:
**Borderline-SMOTE**, **KMeans-SMOTE**, **SVM-SMOTE**, **SMOTE-N (SMOTEN)** ×
**Logistic Regression**, **Random Forest**, **XGBoost**.

Aligned to Person 1's final corrected scope and Person 3's frozen protocol: identical
2,297,168-row analytical dataset, identical 80/20 stratified split (1,837,734 train /
459,434 test, `random_state=42`), identical 15 int64 predictors, identical model
hyperparameters, identical metrics, and a **test set whose SHA-256 fingerprint matches
Person 3's manifest** (`70e180c6…`). Resampling is train-only.

## Contents

| Path | What |
|---|---|
| `PROTOCOL_P2.md` | Frozen methodology, parameter provenance, disclosed deviations |
| `code/` | Reproducible pipeline (prepare → experiments → McNemar → workbook → figures → validation) |
| `results/results_person2.csv` | 12 rows (4 techniques × 3 models), P1/P3 header and 10-decimal format |
| `results/results_baseline_no_balancing.csv` | Self-contained No-Balancing baseline (3 rows) |
| `results/test_predictions_{slug}.parquet` | `y_true` int64, `{Model}__pred` int8, `{Model}__proba` float32 |
| `results/class_distributions.json` | Train class counts before/after per technique |
| `results/audit.json` | Resampling audit incl. synthetic-row validity checks |
| `results/mcnemar_results.csv` | Paired McNemar tests (P2 internal + vs P1 corrected reference) |
| `results/validation_report.json` | Pre-P4 validation gate results |
| `results/run.log`, `results/checkpoint.json` | Execution log and resume checkpoint |
| `metadata/manifest.json`, `metadata/environment.json` | Split fingerprint, versions, hardware |
| `workbook/Person_2_results_workbook.xlsx` | P1/P3-style workbook |
| `figures/` | Metric comparisons, class distributions, confusion matrices, PR/ROC curves |
| `pilot/` | Engineering validation only (never merged into `results/`) |

## Protocol alignment (summary)

| Aspect | P1 (final) | P2 | P3 |
|---|---|---|---|
| Scope | 2015–2024 (2014 excluded) | same | same |
| Train / test | 1,837,734 / 459,434 | same | same |
| Test fingerprint | `70e180c6…` | verified equal | `70e180c6…` |
| Features | 15 int64 codes | same | same |
| Models | LR / RF / XGB (fixed params) | same | same |
| Metrics | 6 + CM + time | same | same |
| Resampling | train-only, seed 42 | train-only, seed 42 | train-only, seed 42 |

## Disclosed deviations (affect P4 interpretation)

1. **SVM-SMOTE and SMOTE-N use a shared stratified 50,000-row train subsample**
   (O(n²) SVM fit / pairwise minority distance matrix). Marked † in tables/figures.
2. **KMeans-SMOTE uses `kmeans_estimator=100`, `cluster_balance_threshold=0.2`**; library
   defaults raise `RuntimeError` on this all-categorical feature space.
3. SMOTE-N is categorical-native (majority vote), not interpolation — inherent to the method.
4. **SVM-SMOTE extrapolates** (`out_step=0.5`): 2.49% of its synthetic values are one code below
   the observed minimum (all code 0). Borderline-SMOTE / KMeans-SMOTE / SMOTE-N produce only
   observed codes.
5. **XGBoost is not bit-reproducible across machines/thread counts.** Reproducing Person 1's
   corrected baseline gives **bit-identical LR and RF** but XGBoost differs by ≈1.9e-3 in metrics
   (1,238 label flips, max proba diff 0.145) — the P4 cross-machine noise floor for XGBoost.
6. Timing is machine-dependent; do not compare across persons.

All other settings reproduce Person 1's implementation exactly; the P2 baseline
cross-check in `06_validate.py` verifies this numerically against Person 1's corrected
No-Balancing predictions.

## Reproduce

```powershell
# from the repository root
python -m venv P2_Advanced_Oversampling/.venv
P2_Advanced_Oversampling/.venv/Scripts/python.exe -m pip install -r P2_Advanced_Oversampling/code/requirements.txt

$py = "P2_Advanced_Oversampling/.venv/Scripts/python.exe"
& $py P2_Advanced_Oversampling/code/01_prepare_p2.py
& $py P2_Advanced_Oversampling/code/02_run_p2_experiments.py --pilot --determinism   # optional
& $py P2_Advanced_Oversampling/code/02_run_p2_experiments.py
& $py P2_Advanced_Oversampling/code/03_mcnemar.py
& $py P2_Advanced_Oversampling/code/04_workbook.py
& $py P2_Advanced_Oversampling/code/05_visualizations.py
& $py P2_Advanced_Oversampling/code/06_validate.py
```

The full run is checkpointed: completed techniques are skipped on restart; `--force` redoes them.

## Key results

Full corrected test set (n = 459,434). Best F1 per classifier:

| Classifier | Best technique | F1 | Recall |
|---|---|---|---|
| LogisticRegression | Borderline-SMOTE (full train) | 0.4399 | 0.706 |
| RandomForest | SVM-SMOTE † (50k subsample) | 0.4405 | 0.574 |
| XGBoost | SVM-SMOTE † (50k subsample) | 0.4389 | 0.586 |

All four advanced oversamplers roughly **double minority-class F1** versus the
No-Balancing baseline (0.153–0.217 → 0.392–0.441) by trading precision/accuracy for recall.
Borderline-SMOTE is the strongest full-train technique; KMeans-SMOTE is the most conservative
(highest accuracy ≈ 0.80, lowest recall); SMOTE-N has the weakest ROC-AUC/PR-AUC; baseline
ROC-AUC is higher than every oversampled run (oversampling improves the threshold-0.5
operating point, not probability ranking). McNemar: 87/90 pairs significant at
Bonferroni α ≈ 0.00056; the only 3 non-significant pairs are P2 baseline vs P1 baseline —
direct evidence the harness reproduces Person 1's protocol (LR/RF bit-identical).
See `report/Person_2_Final_Report.md` for the full table and caveats († = 50k subsample).

**Validation:** 117/117 checks pass (`results/validation_report.json`).
