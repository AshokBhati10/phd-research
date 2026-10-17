# Person 2 — Advanced Oversampling for Class Imbalance

Class-imbalance study on BRFSS diabetes data (target `DIABETE4`, ~17% positive, ~2.5M rows).
Person 2's part: **advanced oversampling** — Borderline-SMOTE, KMeans-SMOTE, SVM-SMOTE, SMOTE-N (SMOTEN).

## Protocol
- Runs on the provided **frozen stratified 80/20 splits** (train 1,998,562 rows / test 499,641 rows), same 15 integer-coded categorical predictors, `YEAR` excluded.
- Same classifiers and metrics as the team protocol: LogisticRegression, RandomForest, XGBoost — Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion counts, fit/predict time.
- All runs use `random_state=42`.

## Honest adaptations (disclosed, not hidden)
- **SVM-SMOTE** and **SMOTEN** are O(n²): both run on one shared stratified 50,000-row train subsample (`results/subsample_*_50k.parquet`); the frozen test set is unchanged.
- **KMeansSMOTE** defaults fail on this data (`RuntimeError: No clusters found with sufficient samples of class 1`): uses `kmeans_estimator=100`, `cluster_balance_threshold=0.2`.
- Full k-fold CV is disclosed as computationally infeasible at this scale; paired **McNemar tests** (30 technique-pairs × 3 classifiers, Bonferroni-corrected) are in `results/mcnemar_results.csv`.

## Contents
| Path | What |
|---|---|
| `notebook/P2_Advanced_Oversampling.ipynb` | Source notebook (all cells) |
| `notebook/P2_Advanced_Oversampling_EXECUTED.ipynb` | Executed notebook with outputs |
| `P2_Advanced_Oversampling_Results.xlsx` | Results workbook (11 sheets) |
| `results/` | Per-run CSVs, combined results, McNemar results, prediction files (`predictions_*.parquet` with `y_true`/`pred`/`proba` — ROC-AUC & PR-AUC are recomputable) |
| `visualizations/` | Confusion matrices + F1/Recall/ROC-AUC comparison charts |

## Key result
All four techniques roughly double the No-Balancing baseline F1 (~0.20 → ~0.44). SVM-SMOTE is marginally best (F1 0.4416, XGBoost); KMeans-SMOTE is the most conservative (higher accuracy, lower recall).

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install "scikit-learn==1.9.1" "imbalanced-learn==0.14.2" "xgboost==3.4.1" \
  "pandas==3.0.6" "numpy==2.5.4" "pyarrow==26.0.0" openpyxl matplotlib seaborn
# place the frozen splits next to the notebook as train/test parquet, then run all cells
```
