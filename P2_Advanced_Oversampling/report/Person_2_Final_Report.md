# Person 2 — Final Report
## Advanced Oversampling for Class Imbalance (BRFSS 2015–2024)

**Scope.** Four advanced oversampling techniques — **Borderline-SMOTE, KMeans-SMOTE,
SVM-SMOTE, SMOTE-N (SMOTEN)** — each with **Logistic Regression, Random Forest, XGBoost**,
on the corrected BRFSS 2015–2024 diabetes dataset. Results are directly mergeable with
Person 1 and Person 3 for the P4 integrated comparison, subject to the deviations in §6.

---

## 1. Data and protocol alignment

| Aspect | Person 1 (final) | Person 2 | Person 3 |
|---|---|---|---|
| Scope | 2015–2024 (YEAR 2014 excluded) | same | same |
| Analytical rows | 2,297,168 | same | same |
| Train / test | 1,837,734 / 459,434 | same | same |
| Train counts | No 1,524,272 / Yes 313,462 | same | same |
| Test counts | No 381,068 / Yes 78,366 | same | same |
| Test fingerprint (SHA-256) | `70e180c6…` | **verified equal** | `70e180c6…` |
| Features | 15 int64 codes | same | same |
| Split | 80/20 stratified, `random_state=42` | same | same |
| Models | LR / RF / XGB (fixed params) | same | same |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, CM, time | same | same |
| Resampling | train-only, seed 42 | train-only, seed 42 | train-only, seed 42 |

Resampling is applied to the training data only; the frozen test set is never touched.
The test fingerprint is checked before the run and re-verified on disk afterwards.

## 2. Techniques and parameters (frozen)

| Technique | Configuration | Train source |
|---|---|---|
| Borderline-SMOTE | `BorderlineSMOTE(sampling_strategy="auto", random_state=42)`; defaults `kind="borderline-1"`, `k_neighbors=5`, `m_neighbors=10` | full 1,837,734 rows |
| KMeans-SMOTE | `KMeansSMOTE(sampling_strategy="auto", kmeans_estimator=100, cluster_balance_threshold=0.2, random_state=42)` | full 1,837,734 rows |
| SVM-SMOTE † | `SVMSMOTE(sampling_strategy="auto", random_state=42)`; defaults incl. `out_step=0.5` | shared 50,000-row subsample |
| SMOTE-N † | `SMOTEN(sampling_strategy="auto", random_state=42)`; `k_neighbors=5` | shared 50,000-row subsample |

† O(n²) cost/memory; both use one shared stratified 50k subsample (No 41,472 / Yes 8,528,
seed 42) regenerated from the corrected train split.

Models — LogisticRegression `max_iter=1000, solver="lbfgs", random_state=42` with
`StandardScaler` fitted on the resampled train; RandomForest `n_estimators=100, max_depth=12,
n_jobs=-1, random_state=42`; XGBoost `n_estimators=300, max_depth=6, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="logloss", n_jobs=-1,
random_state=42`. Decision threshold 0.5.

## 3. Resampling audit

| Technique | Train before (No/Yes) | Train after (No/Yes) | Added | Final train | Resample |
|---|---|---|---|---|---|
| No Balancing | 1,524,272 / 313,462 | 1,524,272 / 313,462 | 0 | 1,837,734 | – |
| Borderline-SMOTE | 1,524,272 / 313,462 | 1,524,272 / 1,524,272 | 1,210,810 | 3,048,544 | 609s |
| KMeans-SMOTE | 1,524,272 / 313,462 | 1,524,272 / 1,524,290 | 1,210,828 | 3,048,562 | 112s |
| SVM-SMOTE † | 41,472 / 8,528 | 41,472 / 41,472 | 32,944 | 82,944 | 117s |
| SMOTE-N † | 41,472 / 8,528 | 41,472 / 41,472 | 32,944 | 82,944 | 44s |

Synthetic-row validity: Borderline-SMOTE, KMeans-SMOTE and SMOTE-N generated **only observed
category codes, all within the observed per-feature range, with zero fractional values**.
SVM-SMOTE (default `out_step=0.5`) extrapolated 12,314 values (2.49% of its synthetic values),
all code `0` / one step below the observed minimum — an inherent property of the technique,
reported in `results/audit.json`.

## 4. Results (frozen test set, n = 459,434)

| Technique | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| No Balancing | LogisticRegression | 0.83298 | 0.54160 | 0.13540 | 0.21664 | 0.77127 | 0.40101 |
| No Balancing | RandomForest | 0.83472 | 0.60756 | 0.08761 | 0.15314 | 0.77752 | 0.41370 |
| No Balancing | XGBoost | 0.83531 | 0.58134 | 0.12322 | 0.20334 | 0.78028 | 0.41755 |
| Borderline-SMOTE | LogisticRegression | 0.69327 | 0.31945 | 0.70617 | **0.43990** | 0.76858 | 0.39620 |
| Borderline-SMOTE | RandomForest | 0.66401 | 0.30552 | **0.76176** | 0.43613 | 0.77397 | 0.40230 |
| Borderline-SMOTE | XGBoost | 0.66550 | 0.30634 | 0.76014 | 0.43670 | 0.77340 | 0.39910 |
| KMeans-SMOTE | LogisticRegression | 0.77570 | 0.37531 | 0.47404 | 0.41894 | 0.75686 | 0.37730 |
| KMeans-SMOTE | RandomForest | 0.79857 | 0.40393 | 0.38029 | 0.39175 | 0.76349 | 0.37521 |
| KMeans-SMOTE | XGBoost | 0.79946 | 0.40800 | 0.38958 | 0.39858 | 0.77012 | 0.38094 |
| SVM-SMOTE † | LogisticRegression | 0.73958 | 0.34733 | 0.59923 | 0.43976 | 0.76306 | 0.39144 |
| SVM-SMOTE † | RandomForest | 0.75122 | 0.35733 | 0.57414 | **0.44050** | 0.76762 | 0.39795 |
| SVM-SMOTE † | XGBoost | 0.74455 | 0.35093 | 0.58573 | 0.43890 | 0.76460 | 0.39654 |
| SMOTE-N † | LogisticRegression | 0.70542 | 0.31623 | 0.62553 | 0.42009 | 0.73743 | 0.37162 |
| SMOTE-N † | RandomForest | 0.72263 | 0.32314 | 0.57197 | 0.41297 | 0.73828 | 0.35972 |
| SMOTE-N † | XGBoost | 0.71456 | 0.31190 | 0.55833 | 0.40022 | 0.72649 | 0.34963 |

### Headline findings

1. **All four advanced oversamplers roughly double minority-class F1** versus the
   No-Balancing baseline (0.153–0.217 → 0.392–0.441), by trading precision and accuracy
   for recall (0.088–0.135 → 0.380–0.762).
2. **Borderline-SMOTE is the strongest full-train technique** (F1 ≈ 0.436–0.440 across all
   three classifiers; highest recall 0.762 with Random Forest). It provides balanced,
   classifier-robust gains.
3. **SVM-SMOTE (50k subsample) is marginally the best on F1** (RF 0.4405, LR 0.4398,
   XGB 0.4389) and is highly consistent across classifiers — but this rests on the
   † subsample deviation and must be read with that caveat.
4. **KMeans-SMOTE is the most conservative**: highest accuracy among the oversamplers
   (0.775–0.799) and the highest precision at its recall level, but the lowest recall
   (0.380–0.474) and F1 (0.392–0.419).
5. **SMOTE-N is the weakest on ranking metrics**: lowest ROC-AUC (0.726–0.738) and PR-AUC
   (0.350–0.372), though its F1 remains close to KMeans-SMOTE's.
6. **Baseline ROC-AUC is higher than every oversampled run** (0.771–0.780 vs 0.726–0.774):
   oversampling improves threshold-0.5 operating-point metrics (recall/F1) but does not
   improve, and can degrade, probability ranking. This is a key P4 discussion point.

## 5. Statistical testing (paired McNemar, frozen test set)

`results/mcnemar_results.csv`: 90 paired tests (per classifier: 10 within-Person-2 pairs +
16 Person-2 vs Person-1 corrected reference pairs), Bonferroni α = 0.000556.

- **87/90 pairs are significant.** With n = 459,434, even small differences reach significance;
  effect sizes (§4) must be read alongside p-values.
- The **only 3 non-significant pairs are P2 baseline vs P1 baseline** (LR p = 1.0, RF p = 1.0 —
  identical predictions; XGB p = 0.125). This is direct evidence that the P2 harness reproduces
  Person 1's protocol.

## 6. Validation gate and deviations for P4

`results/validation_report.json` — **117/117 checks pass** (fingerprint, 12 rows, format,
prediction schema/dtypes, recomputed metrics, confusion totals, class distributions, synthetic
validity, determinism).

Documented deviations that constrain the P1–P2–P3 comparison:

1. **† SVM-SMOTE and SMOTE-N use a 50k train subsample** (O(n²)); P1/P3 techniques use the
   full 1,837,734-row train split. Not strictly like-for-like.
2. **KMeans-SMOTE uses non-default cluster parameters** (`kmeans_estimator=100`,
   `cluster_balance_threshold=0.2`); library defaults fail on this all-categorical space.
3. **SMOTE-N is categorical-native** (majority vote), unlike interpolating SMOTE variants.
4. **SVM-SMOTE extrapolates** (`out_step=0.5`): 2.49% of its synthetic values fall one code
   below the observed minimum.
5. **XGBoost cross-machine nondeterminism.** Reproducing P1's corrected No-Balancing baseline
   yields bit-identical LR and RF but a slightly different XGBoost (1,238 label flips;
   ≈1.9e-3 max metric difference) — the P4 noise floor for XGBoost comparisons.
6. Timing is machine-dependent and must not be compared across persons.
7. Metrics recomputed from saved float32 probabilities agree with the CSV to ≈3e-8; the CSV is
   authoritative (identical float32 storage to P1/P3).

## 7. Artifacts

- `results/results_person2.csv` (12 rows), `results/results_baseline_no_balancing.csv` (3 rows),
  `results/P2_combined_results.csv` (15 rows).
- `results/test_predictions_{borderline_smote,kmeans_smote,svm_smote,smoten,no_balancing}.parquet`
  — `y_true` (int64), `{Model}__pred` (int8), `{Model}__proba` (float32).
- `results/class_distributions.json`, `results/audit.json`, `results/mcnemar_results.csv`,
  `results/validation_report.json`, `results/run.log`, `results/checkpoint.json`.
- `workbook/Person_2_results_workbook.xlsx` (10 sheets), `figures/` (10 figures × PNG/PDF),
  `metadata/manifest.json`, `metadata/environment.json`, `pilot/` (engineering validation).

## 8. Recommendation for P4

Merge P2's 12 technique rows with P1 (No Balancing, ROS, SMOTE, ADASYN) and P3
(ENN, Tomek, SMOTE-Tomek, SMOTE-ENN) using the shared `(Technique, Model)` index and the
identical frozen test set. Present P2's SVM-SMOTE/SMOTE-N rows with the † subsample flag, note
the KMeans-SMOTE parameter choice, and interpret XGBoost differences below ≈2e-3 as
cross-machine noise. Because the baseline is statistically indistinguishable from Person 1's,
P2 rows can be compared directly against P1/P3 rows for the same technique family.
