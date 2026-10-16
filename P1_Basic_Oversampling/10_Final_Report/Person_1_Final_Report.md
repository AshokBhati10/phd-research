# Person 1 — Class-Imbalance Experiments: Final Methodology Report

**Project:** Diabetes prediction on the BRFSS 2015–2024 Master Dataset
**Role:** Person 1 — No Balancing, Random Oversampling, SMOTE, ADASYN

> **Scope correction (v2):** the master dataset contains 201,035 YEAR=2014 rows (8.05%) that were included in the original analysis. The primary results below are from the corrected 2015–2024 rerun (identical protocol, seed 42, new frozen test set n=459,434). The scope discrepancy changed no conclusion (max |Δ| across all 72 metric cells = 0.0074; see §10).
**Date:** 2026-10-01 | Seed: 42 (all randomized operations)
**Report version:** v2 (2026-10-01) — corrected 2015–2024 scope (year-scope audit). The original 2014–2024 analysis is preserved verbatim in `Person_1_Final_Report_ORIGINAL_SCOPE.md` and as Appendix A of this report.

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
   *v2: corrected 2015–2024 scope — Train = 1,837,734 (No 1,524,272 / Yes 313,462); Test = 459,434 (No 381,068 / Yes 78,366); new frozen test set, never resampled — see §10.*
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

## 6. Results — corrected 2015–2024 scope (primary)

### 6.1 Consolidated comparison (shared untouched test set, n = 459,434; seed 42)

| Technique | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| No Balancing | LogisticRegression | 0.8330 | 0.5416 | 0.1354 | 0.2166 | 0.7713 | 0.4010 |
| No Balancing | RandomForest | 0.8347 | 0.6076 | 0.0876 | 0.1531 | 0.7775 | 0.4137 |
| No Balancing | XGBoost | 0.8354 | 0.5832 | 0.1232 | 0.2034 | 0.7802 | 0.4176 |
| Random Oversampling | LogisticRegression | 0.6990 | 0.3239 | 0.7033 | 0.4435 | 0.7715 | 0.3995 |
| Random Oversampling | RandomForest | 0.6847 | 0.3176 | 0.7385 | 0.4441 | 0.7777 | 0.4137 |
| Random Oversampling | XGBoost | 0.6829 | 0.3173 | 0.7463 | 0.4453 | 0.7802 | 0.4176 |
| SMOTE | LogisticRegression | 0.6995 | 0.3230 | 0.6951 | 0.4410 | 0.7685 | 0.3974 |
| SMOTE | RandomForest | 0.6843 | 0.3162 | 0.7315 | 0.4415 | 0.7751 | 0.4084 |
| SMOTE | XGBoost | 0.6840 | 0.3160 | 0.7321 | 0.4415 | 0.7742 | 0.4079 |
| ADASYN | LogisticRegression | 0.6812 | 0.3122 | 0.7224 | 0.4359 | 0.7669 | 0.3952 |
| ADASYN | RandomForest | 0.6541 | 0.3001 | 0.7716 | 0.4321 | 0.7727 | 0.4032 |
| ADASYN | XGBoost | 0.6528 | 0.2988 | 0.7689 | 0.4303 | 0.7703 | 0.3998 |

Source: `11_Audit_2015_2024_Rerun/07_combined_results_audit/combined_results_2015_2024.csv`
(per-experiment CSVs under `11_Audit_2015_2024_Rerun/03_baseline_audit/` … `06_adasyn_audit/`).

**Scope sensitivity:** the corrected rerun reproduces the original pattern almost exactly —
max absolute difference vs the original 2014–2024 table (Appendix A) is 0.0074
across all 72 metric cells (largest shift: No Balancing / RandomForest).
Best F1 remains ROS + XGBoost (0.4453); highest recall remains ADASYN + RandomForest
(0.7716).

### 6.2 Key observations (objective — hold for both scopes)

1. **Balancing trades accuracy/precision for recall.** At the default 0.5 decision
   threshold, any of the three balancing techniques lifts recall from 0.09–0.14
   (baseline) to 0.70–0.77, and F1 from 0.15–0.22 to 0.43–0.45 — roughly a 2× gain in
   F1 — while accuracy falls from ~0.83–0.84 to ~0.65–0.70 and precision from
   ~0.54–0.61 to ~0.30–0.32.
2. **Ranking ability is essentially unchanged.** ROC-AUC stays within 0.767–0.780 and
   PR-AUC within 0.395–0.418 across all 12 cells. Balancing changed the operating
   point, not the models' ability to rank patients by risk.
3. **ROS ≈ SMOTE on this data.** Random Oversampling and SMOTE differ by ≤0.004 in F1
   for every model; ROS is marginally ahead on XGB (0.4453 vs 0.4415). On
   integer-coded categorical features, synthetic interpolation provided no meaningful
   edge over simple replication in this study.
4. **ADASYN is the most aggressive minority learner.** It gives the highest recall for
   tree models (RF 0.7716, XGB 0.7689) but also the lowest precision and accuracy —
   the expected consequence of its adaptive focus on hard minority regions.
5. **Model ranking:** XGBoost has the best ROC-AUC and PR-AUC under every technique.
   Best F1 overall is ROS + XGBoost (0.4453), essentially tied with ROS + RandomForest
   and SMOTE + RandomForest. No threshold tuning or cost-sensitive learning was performed,
   so these comparisons hold at the default 0.5 threshold only.

### 6.3 Visualizations

The 12 requested plots in `08_Visualizations/` were generated from the original
2014–2024 scope. They are retained as-is because the corrected-scope metrics differ by
at most 0.0074 — no visual conclusion changes. (Regenerating all plots on the
corrected scope is left as optional follow-up; the consolidated CSVs are authoritative.)

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
- **Audit rerun (2015–2024):** identical QC battery on the corrected scope — all 12 confusion-matrix totals == 459,434; F1 == 2PR/(P+R) to 1e-9; metrics finite and within [0,1]; new test-set fingerprint (n=459,434, Yes=78,366) byte-identical after every experiment; SMOTE/ADASYN reproduction assertions passed (class counts exactly equal the rerun's saved class_distribution JSONs; X_res[:n_orig] == X_train exactly).

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
7. **Year-scope correction (audit).** The v1 analysis used the master as-is (2014–2024);
   201,035 YEAR=2014 rows (8.05%) entered both training and test sets. The full protocol
   was rerun on the corrected 2015–2024 working set (2,297,168 rows) with a new frozen
   test set (n=459,434); all conclusions unchanged (max |Δ| = 0.0074). See §10.
8. **SMOTE/ADASYN synthetic-value audit.** A strict per-feature audit (see §11) found 0
   invalid categorical codes in ~39M synthetic cells across both methods and both scopes.
   Mechanism note: imbalanced-learn 0.14.2 casts interpolated floats back to the input
   dtype via `X_new.astype(X.dtype)` (truncation toward zero), not rounding — disclosed,
   impact negligible here (synthetic code distributions ≈ minority-train distributions).
9. **Rerun script syntax error.** First launch of the audit rerun failed at parse time
   (stray parenthesis in a test-set assertion); fixed, syntax re-verified, rerun launched
   clean. No partial outputs were written by the failed launch.

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

*v2: the corrected 2015–2024 rerun confirms the interpretation is scope-robust — excluding 2014 moved no metric by more than 0.0074.*

---


---

## Appendix A — Original 2014–2024 results (sensitivity reference)

Preserved verbatim from the v1 report (shared untouched test set n = 499,641; seed 42).
Superseded as the primary table by §6.1; kept so the year-scope correction is auditable.

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


## 10. Year-scope audit

**Finding:** the master file (`brfss_master.parquet`) contains all years 2014–2024.
YEAR=2014 contributes **201,035 rows (8.0472%)**.
The v1 experiments used the master as-is, so all four experiments trained on and were
evaluated against 2014 data: **160,553** 2014 rows in the original
training set and **40,482** in the original test set.

**Proof:** (i) the saved frozen split was re-joined to the master on row index — 2014 rows
are present in both train and test; (ii) the exact v1 preprocessing + split was recomputed
(seed 42, stratified 80/20) and produced byte-identical indices and sizes
(train=1,998,562 / test=499,641). Reproduction assertion: PASS.

**Correction:** a 2015–2024 working set was built by excluding YEAR==2014
(201,035 rows dropped → 2,297,168 rows) and the entire
protocol was rerun identically: same 15 features, same integer encoding, stratified 80/20
split seed 42 → new frozen test set n=459,434 (No 381,068 /
Yes 78,366), same 3 models + hyperparameters, resampling on train only,
same QC battery (all PASS). Results: `11_Audit_2015_2024_Rerun/`.

**Impact:** negligible. Max absolute metric change vs v1 = 0.0074; the technique
ranking, the ROS≈SMOTE finding, and the ADASYN-recall finding are all unchanged. The
v1 table is therefore a valid sensitivity reference (Appendix A), and the corrected
2015–2024 table (§6.1) is the primary result for the thesis.

## 11. Synthetic-feature validity audit

**Question:** standard SMOTE/ADASYN interpolate continuous values, but all 15 BRFSS
predictors are integer-coded categoricals (e.g. EXERANY2 ∈ {1,2}, MARITAL ∈ {1..6},
_AGEG5YR ∈ {5..14}). Do the synthetic samples contain fractional or otherwise invalid codes?

**Method (strict):** SMOTE and ADASYN were regenerated deterministically (seed 42) on both
scopes. Reproduction was verified first: resampled class counts exactly equal the saved
`class_distribution.json` values, and `X_res[:n_orig] == X_train` exactly, so rows
`X_res[n_orig:]` are precisely the synthetic samples. For each of the 15 features, the
valid set = unique values observed in the original training data; a synthetic value is
**invalid** if it is non-integer OR not in the valid set (being inside [min, max] is not
sufficient). Per-feature tables: `12_Audit_Feature_Validity/*/​*_per_feature_validity.csv`;
synthetic samples persisted as parquet.

**Result:**

| Scope | Method | Synthetic rows | Rows with ≥1 invalid feature value | Affected features |
|---|---|---|---|---|
| Original (2014–2024) | SMOTE | 1,318,498 | 0 (0.0%) | — none — |
| Original (2014–2024) | ADASYN | 1,301,378 | 0 (0.0%) | — none — |
| Corrected (2015–2024) | SMOTE | 1,210,810 | 0 (0.0%) | — none — |
| Corrected (2015–2024) | ADASYN | 1,201,468 | 0 (0.0%) | — none — |

**Zero invalid codes** across ~39M synthetic feature cells (both methods, both scopes):
no fractional values, no integer-but-unobserved codes, no out-of-range values. **Affected
features: none.**

**Mechanism note (why, verified in imbalanced-learn 0.14.2 source):** both
`BaseSMOTE._generate_samples` and `ADASYN._fit_resample` end with
`X_new = X_new.astype(X.dtype)` — the interpolated floats are **silently truncated
toward zero** to the input dtype (int64 here), not rounded. Additionally, on this data
~93% of SMOTE synthetic rows exactly duplicate an existing training profile (their
k-nearest neighbours are exact duplicates, so no interpolation occurs at all), which is
why SMOTE behaves nearly identically to ROS here. The silent truncation is disclosed —
not hidden — but its practical impact on these features is negligible: synthetic code
distributions closely match the minority-class training distributions
(e.g. EXERANY2 0.612/0.388 → 0.645/0.355; _AGEG5YR per-code shares within 0.005).

**Why no SMOTE-N:** SMOTE-N is the categorical-valid alternative, but it is assigned to
another team member and is unnecessary for *this* feature representation — all codes are
consecutive integers and the strict audit passes with 0% invalid. SMOTE-N remains the
right recommendation for non-consecutive or non-ordinal categorical codes, where the
truncation mechanism above could silently map interpolations onto wrong codes.

## 12. Final recommendation

**The Person 1 SMOTE/ADASYN results are methodologically acceptable for the thesis,**
with the following disclosed caveats (all documented above, none hidden):

1. **Reported metrics are honest regardless of synthetic-sample mechanics** — every
   metric is computed on real, untouched test observations; the test set was never
   resampled in any experiment (fingerprints re-verified after every run, both scopes).
2. **ROS is categorical-valid by construction** (exact row replication); SMOTE/ADASYN
   pass the strict per-feature validity audit with 0% invalid codes on both scopes.
3. **The silent int-truncation in imbalanced-learn's SMOTE/ADASYN must be disclosed**
   (done in §11): interpolated values are truncated toward zero rather than rounded.
   Impact here is negligible, but the thesis should state the mechanism rather than
   imply textbook continuous interpolation on categorical codes.
4. **Scope:** cite the corrected 2015–2024 results (§6.1) as primary; the original
   2014–2024 analysis is a valid sensitivity check (Appendix A) since excluding 2014
   changed no metric by more than 0.0074.
5. **No re-implementation is recommended**: do not round/clip synthetic values (would
   alter the published-method comparison), do not switch to SMOTE-N (assigned to another
   group; unnecessary here), and do not claim these results generalize beyond this
   dataset, feature set, models, and the 0.5 decision threshold.

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
- `11_Audit_2015_2024_Rerun/` — year-scope audit: year_counts.csv, corrected splits +
  protocol_config_audit.json, per-experiment results/class-distributions/test-predictions,
  combined_results_2015_2024.csv
- `12_Audit_Feature_Validity/` — per-feature validity CSVs, validity_summaries.json,
  persisted synthetic samples (parquet) for SMOTE/ADASYN × original/corrected scope
- `09_Code/audit/` — 01_year_audit.py, 02_rerun_2015_2024.py, 03_feature_validity.py,
  04_build_report_v2.py (this report's builder)
- `10_Final_Report/Person_1_Final_Report_ORIGINAL_SCOPE.md` — v1 report, verbatim
