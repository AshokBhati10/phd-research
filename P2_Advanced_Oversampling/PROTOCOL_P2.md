# Person 2 — Frozen Methodology (advanced oversampling, BRFSS 2015–2024)

Status: **FROZEN**. Aligned to Person 1's final corrected scope and Person 3's frozen protocol.
Reference implementation: Person 1 workspace (`09_Code/03_experiments.py`) and corrected rerun
(`11_Audit_2015_2024_Rerun`).

## 1. Data scope (fixed)

- Source splits: `P1_Basic_Oversampling/11_Audit_2015_2024_Rerun/02_preprocessing_audit/`
  (`X_train`, `X_test`, `y_train`, `y_test`). These are the corrected 2015–2024 arrays
  (YEAR 2014 excluded).
- Analytical dataset: **2,297,168 rows × 15 predictors**. Target `DIABETE4`: `"Yes" → 1`, `"No" → 0`.
- Predictors (15, exact Person 1 list): EXERANY2, _MICHD, ASTHMA3, ADDEPEV3, CHCKDNY2, MARITAL,
  _EDUCAG, DRNKANY6, _AGEG5YR, _BMI5CAT, _INCOMG1, _RFSMOK3, GENHLTH, HAVARTH4, SEXVAR.
- Representation: int64 integer-coded categoricals, no one-hot, no target encoding, no scaling
  before resampling. `YEAR` is metadata only, never a predictor.

## 2. Train/test protocol (fixed)

- Person 1's frozen split: `train_test_split(test_size=0.20, random_state=42, stratify=y,
  shuffle=True)`.
- Train **1,837,734** (No 1,524,272 / Yes 313,462); test **459,434** (No 381,068 / Yes 78,366).
- Test fingerprint (SHA-256 of `X_test` values then `y_test` values, Person 3 algorithm):
  `70e180c659c74efbb6ac6368f5b061c9a94cf7efc10a707f52abcf29280b0d11`.
  `01_prepare_p2.py` aborts on mismatch; `02_run_p2_experiments.py` re-verifies at start and
  after the run.
- Resampling/scaler fitting: **training data only**, never the test set.

## 3. Person 2 techniques (exact)

imbalanced-learn 0.14.2. `sampling_strategy="auto"` (minority → majority size),
`random_state=42` wherever supported.

| Technique | Constructor | Train source |
|---|---|---|
| Borderline-SMOTE | `BorderlineSMOTE(sampling_strategy="auto", random_state=42)` — defaults `kind="borderline-1"`, `k_neighbors=5`, `m_neighbors=10` | full 1,837,734-row train split |
| KMeans-SMOTE | `KMeansSMOTE(sampling_strategy="auto", kmeans_estimator=100, cluster_balance_threshold=0.2, random_state=42)` | full train split |
| SVM-SMOTE | `SVMSMOTE(sampling_strategy="auto", random_state=42)` — library defaults (`k_neighbors=5`, `m_neighbors=10`, `out_step=0.5`, RBF SVM) | shared stratified **50,000-row** train subsample |
| SMOTE-N | `SMOTEN(sampling_strategy="auto", random_state=42)` — library defaults (`k_neighbors=5`) | same shared 50,000-row subsample |

### Deviations from library defaults (documented, approved)

1. **KMeans-SMOTE params.** Defaults (`kmeans_estimator=None` → 8 clusters,
   `cluster_balance_threshold="auto"` → 0.5) raise
   `RuntimeError: No clusters found with sufficient samples of class 1` on this all-categorical
   feature space (verified in the original P2 run). `kmeans_estimator=100` (MiniBatchKMeans,
   100 clusters) and `cluster_balance_threshold=0.2` (≈ the global minority rate 0.171) are the
   frozen P2 configuration.
2. **SVM-SMOTE / SMOTE-N subsample.** Both are O(n²) in samples (SVM fit / pairwise minority
   distance matrix); the full split is computationally infeasible. They use one shared
   stratified 50k subsample (`train_test_split(train_size=50000, stratify=y,
   random_state=42)`; No 41,472 / Yes 8,528), regenerated from the corrected train split and
   persisted as `results/subsample_*_50k.parquet`. The test set is unchanged.
   **P4 implication:** these two rows are not strictly comparable to full-train techniques;
   they are flagged with † in every table/figure.

## 4. Models (exact, per technique)

| Model | Parameters | Scaling |
|---|---|---|
| LogisticRegression | `max_iter=1000, solver="lbfgs", random_state=42` | `StandardScaler` fitted on the **resampled training data** (Person 1's actual code), applied to train and test |
| RandomForest | `n_estimators=100, max_depth=12, n_jobs=-1, random_state=42` | none (raw integer codes) |
| XGBoost | `n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=42` | none (raw integer codes) |

Decision threshold 0.5; `predict` for labels, `predict_proba[:, 1]` for ROC-AUC / PR-AUC.

## 5. Metrics (per technique × model)

Accuracy, Precision, Recall/Sensitivity, F1, ROC-AUC, PR-AUC (average precision),
TN, FP, FN, TP (confusion matrix at 0.5), `fit_predict_sec` (fit + predict + predict_proba).
CSV: 10 decimals for metrics, 1 decimal for timing. Header identical to Person 1/3:
`Technique,Model,Accuracy,Precision,Recall,F1,ROC_AUC,PR_AUC,TN,FP,FN,TP,fit_predict_sec`.

## 6. Baseline

A self-contained **No Balancing** run (3 models, raw train split, same code path) is included
for the P2 workbook and McNemar anchoring. It is written separately
(`results/results_baseline_no_balancing.csv`) and cross-checked against Person 1's corrected
baseline in `06_validate.py`; it must reproduce P1's metrics to < 1e-6.

## 7. Execution, checkpointing, integrity

- Sequential: one technique at a time, one model at a time; resampled arrays released before the
  next technique.
- Checkpoint/resume: a technique is complete iff all 3 rows exist in `results_person2.csv` and
  `test_predictions_{slug}.parquet` exists; `--force` redoes it. Rows are appended only after a
  successful model run; parquet files are written to temp then atomically renamed.
- Resampling is deterministic (`random_state=42` everywhere supported). The pilot runs each
  resampler twice and requires byte-identical `(X_res, y_res)` fingerprints.
- `run.log` records every step with RSS; `audit.json` records class bookkeeping and
  synthetic-row validity; `class_distributions.json` records before/after counts.

## 8. Synthetic-row validity audit

For every technique that appends rows, P2 verifies: all values finite; synthetic rows are within
the observed per-feature [min, max]; strict fractional-value scan (must be 0); unobserved
category-code count (reported as observed, never claimed without verification). Because
imbalanced-learn casts interpolated values with `astype(X.dtype)`, fractional codes never
persist; unobserved integer codes can occur and are counted.

Observed results on the full corrected run:

| Technique | Synthetic rows | Fractional values | Out-of-range values | Unobserved codes |
|---|---|---|---|---|
| Borderline-SMOTE | 1,210,810 | 0 | 0 | 0 |
| KMeans-SMOTE | 1,210,828 | 0 | 0 | 0 |
| SVM-SMOTE | 32,944 | 0 | 12,314 (2.49% of values) | 12,314 |
| SMOTE-N | 32,944 | 0 | 0 | 0 |

SVM-SMOTE's out-of-range values are all the integer code `0` (or one step below the observed
minimum, e.g. `_AGEG5YR` min 5 → 4): its default `out_step=0.5` extrapolates along minority
support-vector directions, i.e. outside the convex hull of the training minority. This is an
inherent, documented property of the technique and is reported in `results/audit.json`.
Borderline-SMOTE / KMeans-SMOTE / SMOTE-N generate only observed codes within range.

## 9. Outputs

- `results/results_person2.csv` — 12 rows (4 techniques × 3 models), P1/P3 header/format.
- `results/results_{borderline_smote,kmeans_smote,svm_smote,smoten}.csv` — per-technique, 3 rows each.
- `results/results_baseline_no_balancing.csv` — 3 baseline rows.
- `results/P2_combined_results.csv` — baseline + 12 technique rows (workbook source).
- `results/class_distributions.json`, `results/audit.json`, `results/mcnemar_results.csv`,
  `results/run.log`, `results/checkpoint.json`, `results/validation_report.json`.
- `results/test_predictions_{slug}.parquet` — `y_true` (int64), `{Model}__pred` (int8),
  `{Model}__proba` (float32); identical schema to Person 1/3.
- `results/subsample_X_50k.parquet`, `results/subsample_y_50k.parquet` — shared subsample.
- `workbook/Person_2_results_workbook.xlsx` — P1/P3-style sheets.
- `figures/` — metric comparisons, class distributions, confusion matrices, PR/ROC curves.
- `metadata/manifest.json`, `metadata/environment.json`.
- `pilot/` — engineering validation only; never merged into `results/`.

## 10. Reproducibility

`code/requirements.txt` pins the Person 1/3 stack: Python 3.12, numpy 2.5.3, pandas 3.0.6,
scikit-learn 1.9.1, imbalanced-learn 0.14.2, xgboost 3.4.1, pyarrow 25.0.1, openpyxl 3.1.5.
Actual environment recorded in `metadata/environment.json` and the workbook `Protocol_config`
sheet.

## 11. Differences that affect the P1–P2–P3 comparison (disclose in P4)

1. SVM-SMOTE and SMOTE-N use a 50k train subsample (P1/P3 techniques use the full train split).
2. KMeans-SMOTE uses non-default cluster parameters required for this feature space.
3. SMOTE-N is categorical-native (majority vote) rather than interpolating — inherent to the
   technique.
4. Minor environment drift vs Person 3: numpy 2.5.3 (identical), pyarrow 25.0.1 (identical);
   Person 2's original run used numpy 2.5.4 / pyarrow 26.0.0, so this rerun is *closer* to P3.
5. Timing values are machine-dependent and must not be compared across persons.
6. Paired tests are McNemar on one frozen test set only; no k-fold CV (computational limit at
   this scale, same limitation as P3).
7. **SVM-SMOTE extrapolation.** With default `out_step=0.5`, SVM-SMOTE generates 12,314
   out-of-range integer values (2.49% of its synthetic values, all code 0 / one below the
   observed minimum). The other three techniques generate only observed codes.
8. **XGBoost cross-machine nondeterminism.** Reproducing Person 1's corrected No-Balancing
   baseline on identical data gives **bit-identical LogisticRegression and RandomForest**
   (0 label flips) but a slightly different **XGBoost** model (1,238 label flips, max metric
   diff ≈ 1.9e-3, max probability diff ≈ 0.145). XGBoost `hist` training with `n_jobs=-1` is not
   bit-reproducible across thread counts/machines, so this ≈2e-3 level of metric drift is the
   P4 cross-machine noise floor for XGBoost comparisons (LR/RF are exact).
9. Stored probabilities are float32 (identical to P1/P3), so metrics recomputed from the saved
   parquets agree with the CSV to ≈3e-8 rather than machine precision. The CSV remains the
   authoritative result.
