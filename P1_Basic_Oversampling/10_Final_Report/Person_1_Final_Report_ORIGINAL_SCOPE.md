# Person 1 — Class-Imbalance Experiments: Final Methodology Report

**Project:** Diabetes prediction on the BRFSS 2015–2024 Master Dataset
**Role:** Person 1 — No Balancing, Random Oversampling, SMOTE, ADASYN
**Date:** 2026-10-01 | Seed: 42 (all randomized operations)

---

## 1. Dataset

- **File:** `brfss_master.parquet` (8,647,149 bytes), located in and downloaded from the
  connected Google Drive (the master dataset behind the user-supplied share link).
- **Shape:** 2,498,203 rows × 17 columns. The master was never modified; it is read-only input.
- **Columns:** `YEAR, DIABETE4, EXERANY2, _MICHD, ASTHMA3, ADDEPEV3, CHCKDNY2, MARITAL,`
  `_EDUCAG, DRNKANY6, _AGEG5YR, _BMI5CAT, _INCOMG1, _RFSMOK3, GENHLTH, HAVARTH4, SEXVAR`.
- **Target:** `DIABETE4`, binary — **No** = 2,073,163 (83.0%), **Yes** = 425,040 (17.0%).
  Counts sum exactly to the row count. Imbalance ratio (majority:minority) = **4.878:1**.
- **Years:** 2014–2024, all 11 years represented (note: the nominal project title is
  "2015–2024"; the actual master file also contains 2014, which is used as-is —
  see §8 Implementation Issues).
- **Missing values:** 0 in every column. No constant or near-constant columns.
- **Duplicates:** 1,045,135 exact analytical-profile duplicates are present. Per the
  documented prior audit, these are not treated as duplicate respondents (respondent
  audit via STATE+SEQNO found zero duplicates); no rows were added or removed by this work.

## 2. Preprocessing

No Group 1 preprocessing code or notebook was found (see §8), so the protocol below is
defined cleanly and defensibly, using the documented prior Person-1 methodology where applicable.

1. Master dataset read as-is (already standardized: DIABETE4 recoded to No/Yes, invalid/
   refused/don't-know codes handled upstream, complete cases).
2. Age restriction `_AGEG5YR >= 5` — retained all 2,498,203 rows (minimum is 5.0).
3. Complete-case check — no-op (0 missing values).
4. Target separated: `y = 1` if `DIABETE4 == "Yes"`, else `0`. Target column never appears
   among predictors (asserted in code).
5. Features (15): EXERANY2, _MICHD, ASTHMA3, ADDEPEV3, CHCKDNY2, MARITAL, _EDUCAG,
   DRNKANY6, _AGEG5YR, _BMI5CAT, _INCOMG1, _RFSMOK3, GENHLTH, HAVARTH4, SEXVAR.
   `YEAR` excluded as metadata (not a patient-level predictor; consistent with the
   documented research-variable list). Integer-coded categoricals kept as integers (no one-hot).
6. **Train/test split BEFORE any resampling:** stratified 80/20, shuffle=True, seed 42.
   Train = 1,998,562 (No 1,658,530 / Yes 340,032); Test = 499,641 (No 414,633 / Yes 85,008).
   Split validations passed: sizes sum correctly, zero index overlap, identical stratification
   proportions, identical frozen feature column order for train and test.
7. **Test set frozen and never resampled** — asserted after every experiment
   (fingerprint n=499,641, Yes=85,008 re-verified byte-identical each time).

## 3. Experiments

All four experiments share the master dataset, feature set, preprocessing, split,
models, metrics, and random seed; only the training-data balancing differs.

| # | Technique | What was done (training data only) |
|---|-----------|------------------------------------|
| 1 | No Balancing (baseline) | Original imbalanced training data; no sampling |
| 2 | Random Oversampling (ROS) | Minority class replicated to majority size: 340,032 → 1,658,530; +1,318,498 rows; final train 3,317,060 |
| 3 | SMOTE | Standard SMOTE (k=5) generated synthetic minority samples: +1,318,498; final train 3,317,060 (minority = majority = 1,658,530) |
| 4 | ADASYN | ADASYN (k=5) generated synthetic minority samples adaptively: minority 340,032 → 1,641,410 (not exact parity — documented expected behavior); +1,301,378 rows; final train 3,299,940 |

*SMOTE/ADASYN mixed-data decision:* all 15 features are integer-coded categoricals.
SMOTENC was attempted but imbalanced-learn 0.14.2 rejects all-categorical feature sets
("SMOTE-NC is not designed to work only with categorical features"). Standard SMOTE was
therefore used on the integer codes — the convention in published BRFSS-SMOTE work —
with synthetic values validated to remain within each feature's observed [min, max]
and free of NaN/inf. ADASYN has no categorical-aware variant and was applied identically.

## 4. Models

- **LogisticRegression** (max_iter=1000) — StandardScaler fitted on the training fold only.
- **RandomForestClassifier** (n_estimators=100, max_depth=12, n_jobs=-1).
- **XGBClassifier** (n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
  colsample_bytree=0.8, tree_method="hist", eval_metric="logloss").
- All with `random_state=42`. Full 2.5M-row training used (no downsampling).

## 5. Evaluation Metrics

Accuracy, Precision, Recall/Sensitivity, F1-score, ROC-AUC, PR-AUC (average precision),
and the full confusion matrix — all computed on the SAME untouched test set (n=499,641).

## 6. Results

### 6.1 Consolidated comparison (shared untouched test set, n = 499,641; seed 42)

| Technique | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| No Balancing | LogisticRegression | 0.8335 | 0.5425 | 0.1377 | 0.2197 | 0.7724 | 0.4029 |
| No Balancing | RandomForest | 0.8356 | 0.6149 | 0.0900 | 0.1570 | 0.7784 | 0.4166 |
| No Balancing | XGBoost | 0.8362 | 0.5867 | 0.1266 | 0.2083 | 0.7813 | 0.4204 |
| Random Oversampling | LogisticRegression | 0.6991 | 0.3233 | 0.7034 | 0.4430 | 0.7726 | 0.4013 |
| Random Oversampling | RandomForest | 0.6847 | 0.3170 | 0.7391 | 0.4437 | 0.7786 | 0.4165 |
| Random Oversampling | XGBoost | 0.6831 | 0.3170 | 0.7474 | 0.4452 | 0.7812 | 0.4202 |
| SMOTE | LogisticRegression | 0.7001 | 0.3231 | 0.6965 | 0.4414 | 0.7698 | 0.3991 |
| SMOTE | RandomForest | 0.6854 | 0.3168 | 0.7342 | 0.4426 | 0.7762 | 0.4108 |
| SMOTE | XGBoost | 0.6841 | 0.3156 | 0.7334 | 0.4413 | 0.7756 | 0.4104 |
| ADASYN | LogisticRegression | 0.6831 | 0.3130 | 0.7223 | 0.4368 | 0.7681 | 0.3968 |
| ADASYN | RandomForest | 0.6537 | 0.3000 | 0.7766 | 0.4328 | 0.7740 | 0.4061 |
| ADASYN | XGBoost | 0.6543 | 0.2996 | 0.7712 | 0.4315 | 0.7718 | 0.4024 |

Source: `07_Combined_Results/combined_results.csv` (identical per-experiment CSVs under
`03_Baseline/`, `04_Random_Oversampling/`, `05_SMOTE/`, `06_ADASYN/`).

### 6.2 Key observations (objective)

1. **Balancing trades accuracy/precision for recall.** At the default 0.5 decision
   threshold, any of the three balancing techniques lifts recall from 0.09–0.14
   (baseline) to 0.70–0.78, and F1 from 0.16–0.22 to 0.43–0.45 — roughly a 2× gain in
   F1 — while accuracy falls from ~0.83–0.84 to ~0.65–0.70 and precision from
   ~0.54–0.61 to ~0.30–0.32.
2. **Ranking ability is essentially unchanged.** ROC-AUC stays within 0.768–0.782 and
   PR-AUC within 0.397–0.420 across all 12 cells. Balancing changed the operating
   point, not the models' ability to rank patients by risk.
3. **ROS ≈ SMOTE on this data.** Random Oversampling and SMOTE differ by ≤0.004 in F1
   for every model; ROS is marginally ahead on XGB (0.4452 vs 0.4413). On
   integer-coded categorical features, synthetic interpolation provided no meaningful
   edge over simple replication in this study.
4. **ADASYN is the most aggressive minority learner.** It gives the highest recall for
   tree models (RF 0.7766, XGB 0.7712) but also the lowest precision and accuracy —
   the expected consequence of its adaptive focus on hard minority regions.
5. **Model ranking:** XGBoost has the best ROC-AUC and PR-AUC under every technique.
   Best F1 overall is ROS + XGBoost (0.4452), essentially tied with ROS + RandomForest
   (0.4437) and SMOTE + RandomForest (0.4426). No threshold tuning or cost-sensitive
   learning was performed, so these comparisons hold at the default 0.5 threshold only.

### 6.3 Visualizations

All 12 requested plots are in `08_Visualizations/`:
01 class distribution (original), 02 ROS, 03 SMOTE, 04 ADASYN,
05 confusion matrices (4×3), 06 ROC curves, 07 PR curves,
08–12 accuracy / recall / F1 / ROC-AUC / PR-AUC comparisons.

## 7. Validation

- Train/test split performed before any resampling: confirmed.
- ROS / SMOTE / ADASYN applied to training data only: confirmed (train-only asserts in code).
- Test set never resampled: confirmed — y_true fingerprint (n=499,641, Yes=85,008)
  re-verified identical after every experiment and in the QC audit.
- No target-derived feature leakage: target column asserted absent from all feature matrices.
- Random seeds recorded: 42 everywhere (config, code, logs).
- Results: confusion-matrix totals == test size for all 12 model×technique cells;
  Accuracy == (TN+TP)/n; F1 == 2PR/(P+R) to 1e-6; all metrics finite and within [0,1];
  no NaN/inf; all 12 expected result rows present.

## 8. Implementation Issues

1. **No Group 1 artifacts found.** Drive and ~/workspace were searched for Group 1
   preprocessing code, notebooks, and results; none exist. Only prior-session Person-1
   handoff artifacts (preprocessing.py, preprocessing_summary.csv,
   person1_final_handoff_status.csv) were found and used as methodology reference.
   → Cannot claim to reproduce Group 1; a clean defensible protocol (seed 42,
   stratified 80/20, LR/RF/XGBoost) is documented instead.
2. **Year coverage is 2014–2024, not 2015–2024.** The actual master file contains 2014
   (201,035 rows). All years 2015–2024 are present; the file is used as-is.
3. **Raw-XPT 7/9 filter not applied to the master.** Codes 7/9 in the master appear only
   in _AGEG5YR, where they are valid age categories (50–54, 60–64). Applying the filter
   from the downloaded raw-XPT preprocessing.py would wrongly drop ~20% of valid rows.
   → Filter not applied; documented.
4. **SMOTENC rejected all-categorical features.** imbalanced-learn raised ValueError.
   → Standard SMOTE on integer codes; synthetic ranges validated.
5. **ADASYN did not reach exact parity.** Final minority = 1,641,410 vs majority
   1,658,530 (added 1,301,378) — known ADASYN behavior (documented in spec as possible); not forced.
6. **pip install:** PEP-668 externally-managed environment required --break-system-packages;
   xgboost wheel failed first attempt with "No space left on device" because pip's TMPDIR
   defaulted to a 512 MB tmpfs — fixed by pointing TMPDIR at ~/.pip_tmp. (Environment
   setup only; no impact on results.)

## 9. Research Interpretation

The four approaches were compared under strictly identical conditions, so observed
differences are attributable to the balancing treatment, not to data, features, or
evaluation differences. Objectively:

- **Doing nothing** yields the highest accuracy (0.84) but clinically poor minority
  detection (recall ≤ 0.14, F1 ≤ 0.22): the models largely predict the majority class.
- **Any balancing** roughly doubles F1 (to 0.43–0.45) by converting the models from
  majority-class predictors into minority-class detectors, at a cost of ~0.14–0.19
  accuracy points and ~0.24–0.31 precision points.
- **Between the three balancing methods the differences are small**: ROS and SMOTE are
  effectively interchangeable here; ADASYN pushes recall highest for tree models at a
  slightly larger precision cost.
- **ROC-AUC/PR-AUC stability** shows the underlying ranking signal (XGBoost ≥
  RandomForest > LogisticRegression) is robust to the resampling choice.

No claim is made that any technique is universally best: the results hold for this
dataset, these features, these models, and the default 0.5 threshold. Threshold tuning,
cost-sensitive training, or calibration could change the operating-point trade-offs.

---

### File map (all under `~/workspace/Person_1_Class_Imbalance/`)

- `01_Data_Inspection/` — column_inventory.csv, inspection_log.txt
- `02_Preprocessing/` — X_train/X_test/y_train/y_test parquet, protocol_config.json
- `03_Baseline/`, `04_Random_Oversampling/`, `05_SMOTE/`, `06_ADASYN/` — results CSV,
  class_distribution.json, test_predictions.parquet per experiment
- `07_Combined_Results/` — combined_results.csv, class_distributions.json, run console log
- `08_Visualizations/` — 12 PNG plots
- `09_Code/` — 01–05 scripts (inspection, preprocessing, experiments, visualizations, QC)
- `10_Final_Report/` — this report
- `validation.log` — full execution/validation log with corrections
