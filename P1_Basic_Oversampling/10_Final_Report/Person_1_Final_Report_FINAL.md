# Person 1 — Class-Imbalance Experiments: Final Methodology Report

**Project:** Diabetes prediction on the BRFSS 2015–2024 Master Dataset
**Role:** Person 1 — No Balancing, Random Oversampling, SMOTE, ADASYN
**Date:** 2026-10-01 | Seed: 42 (all randomized operations)
**Report version:** FINAL (2026-10-01) — primary scope **BRFSS 2015–2024**.

> **Scope note:** the master file contains 201,035 YEAR=2014 rows (8.05%) that were included
> in the original analysis. The **primary results in this report are from the corrected
> 2015–2024 rerun** (identical protocol, seed 42, new frozen test set n=459,434). The original
> 2014–2024 analysis is preserved **only as a sensitivity analysis** in Appendix A and in
> `Person_1_Final_Report_ORIGINAL_SCOPE.md` / `Person_1_Final_Report_v2_AUDITED.md`.
> Excluding 2014 changed no metric by more than 0.0074 (see §11).

---

## 1. Dataset

### 1.1 Master source dataset (read-only input)

- **File:** `brfss_master.parquet` (8,647,149 bytes), located in and downloaded from the
  connected Google Drive (the master dataset behind the user-supplied share link).
- **Shape:** 2,498,203 rows × 17 columns. The master was never modified; it is read-only input.
- **Columns:** `YEAR, DIABETE4, EXERANY2, _MICHD, ASTHMA3, ADDEPEV3, CHCKDNY2, MARITAL,`
  `_EDUCAG, DRNKANY6, _AGEG5YR, _BMI5CAT, _INCOMG1, _RFSMOK3, GENHLTH, HAVARTH4, SEXVAR`.
- **Years in master:** 2014–2024 (all 11 years).
- **Duplicates:** 1,045,135 exact analytical-profile duplicates are present in the master.
  Per the documented prior audit, these are not treated as duplicate respondents (respondent
  audit via STATE+SEQNO found zero duplicates); no rows were added or removed by this work.

### 1.2 Primary 2015–2024 analytical dataset (used for all primary results)

The primary analysis uses the master **excluding YEAR=2014** (201,035 rows, 8.05%):

- **Shape:** **2,297,168 rows × 17 columns** (15 predictors + `YEAR` + `DIABETE4`).
- **Year coverage:** all ten years 2015–2024 are represented:

| Year | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|---|
| Rows | 176,224 | 287,067 | 254,491 | 251,517 | 234,260 | 217,393 | 218,971 | 214,921 | 217,594 | 224,730 |
- **Target:** `DIABETE4`, binary — **No = 1,905,340 (82.9430%)**, **Yes = 391,828 (17.0570%)**.
  Counts sum exactly to the row count. Imbalance ratio (majority:minority) = **4.8627:1**.
- **Missing values:** 0 in every column. No constant or near-constant columns.
- **Class counts in the corrected split:** train = 1,837,734 (No 1,524,272 / Yes 313,462);
  test = 459,434 (No 381,068 / Yes 78,366). Split proportions are identical to the full-set
  proportions (stratification verified).

## 2. Preprocessing

No Group 1 preprocessing code or notebook was found in Drive or the workspace (see §8),
so the protocol below is defined cleanly and defensibly.

1. **Master dataset loaded** from `brfss_master.parquet` (already standardized: DIABETE4
   recoded to No/Yes, invalid/refused/don't-know codes handled upstream, complete cases).
2. **`YEAR` inspected** — 2014–2024 present; YEAR=2014 contributes 201,035 rows (8.05%).
3. **YEAR=2014 excluded** for the primary analysis → corrected 2015–2024 analytical dataset
   (2,297,168 rows).
4. **Corrected analytical dataset created** (all years 2015–2024, see §1.2).
5. **Complete-case check** — no-op (0 missing values in the corrected set).
6. **Age restriction** `_AGEG5YR >= 5` — retained all 2,297,168 rows (minimum code is 5).
7. **Target separated:** `y = 1` if `DIABETE4 == "Yes"`, else `0`. The target column never
   appears among predictors (asserted in code; no target leakage).
8. **Feature selection (15):** EXERANY2, _MICHD, ASTHMA3, ADDEPEV3, CHCKDNY2, MARITAL,
   _EDUCAG, DRNKANY6, _AGEG5YR, _BMI5CAT, _INCOMG1, _RFSMOK3, GENHLTH, HAVARTH4, SEXVAR.
   Integer-coded categoricals kept as integers (no one-hot).
9. **`YEAR` excluded from predictors** — kept as metadata only (not a patient-level predictor).
10. **Train/test split BEFORE any resampling:** stratified 80/20, shuffle=True, seed 42.
    Train = 1,837,734 (No 1,524,272 / Yes 313,462); Test = 459,434 (No 381,068 / Yes 78,366).
    Split validations passed: sizes sum correctly, zero index overlap, identical stratification
    proportions, identical frozen feature column order for train and test.
11. **Test set frozen and never resampled** — fingerprint (n=459,434, Yes=78,366) asserted
    byte-identical after every experiment; **resampling is applied to training data only.**

Because the split (step 10) strictly precedes any resampling (step 11), it is impossible for
synthetic or replicated samples to enter the test set.

## 3. Experiments

All four experiments share the corrected 2015–2024 analytical dataset, feature set,
preprocessing, split, models, metrics, and random seed; only the training-data balancing differs.

| # | Technique | What was done (training data only) |
|---|-----------|------------------------------------|
| 1 | No Balancing (baseline) | Original imbalanced training data; no sampling (train 1,837,734: No 1,524,272 / Yes 313,462) |
| 2 | Random Oversampling (ROS) | Minority class replicated to exact majority size: 313,462 → 1,524,272; +1,210,810 rows; final train **3,048,544** |
| 3 | SMOTE | Standard SMOTE (k=5) generated synthetic minority samples: +1,210,810; final train **3,048,544** (minority = majority = 1,524,272) |
| 4 | ADASYN | ADASYN (k=5) generated synthetic minority samples adaptively: minority 313,462 → 1,514,930 (not exact parity — documented expected behavior); +1,201,468 rows; final train **3,039,202** |

*SMOTE/ADASYN mixed-data decision:* all 15 features are integer-coded categoricals.
SMOTENC was attempted but imbalanced-learn 0.14.2 rejects all-categorical feature sets
("SMOTE-NC is not designed to work only with categorical features"). Standard SMOTE was
therefore used on the integer codes — the convention in published BRFSS-SMOTE work — and a
strict per-feature audit (see §12) confirmed **0 invalid synthetic codes** across all features,
both methods, and both scopes (no fractional, integer-but-unobserved, or out-of-range values).
ADASYN has no categorical-aware variant and was applied identically.

## 4. Models

- **LogisticRegression** (max_iter=1000) — StandardScaler fitted on the training fold only.
- **RandomForestClassifier** (n_estimators=100, max_depth=12, n_jobs=-1).
- **XGBClassifier** (n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
  colsample_bytree=0.8, tree_method="hist", eval_metric="logloss").
- All with `random_state=42`.
- Training used **1,837,734 original training observations before resampling** (no
  downsampling). ROS, SMOTE, and ADASYN increase the training-set size per the validated
  resampling outputs in §3 (final sizes 3,048,544 / 3,048,544 / 3,039,202).

Model definitions and hyperparameters are exactly as validated; they were not changed.

## 5. Evaluation Metrics

Accuracy, Precision, Recall/Sensitivity, F1-score, ROC-AUC, PR-AUC (average precision),
and the full confusion matrix — **all computed on the same untouched corrected 2015–2024
test set of n = 459,434** (No 381,068 / Yes 78,366).

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

Source (authoritative): `11_Audit_2015_2024_Rerun/07_combined_results_audit/combined_results_2015_2024.csv`
(per-experiment CSVs under `11_Audit_2015_2024_Rerun/03_baseline_audit/` … `06_adasyn_audit/`).
Cross-checked cell-by-cell against this table: 0 mismatches; every confusion matrix sums to 459,434.

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
4. **ADASYN is the most aggressive minority learner.** It gives the highest observed
   recall for tree models (RF 0.7716, XGB 0.7689) but also the lowest precision and
   accuracy — the expected consequence of its adaptive focus on hard minority regions.
5. **Model ranking:** XGBoost has the best ROC-AUC and PR-AUC under every technique.
   Highest observed F1 in this experiment is ROS + XGBoost (0.4453), essentially tied
   with ROS + RandomForest and SMOTE + RandomForest. No threshold tuning or
   cost-sensitive learning was performed, so these comparisons hold at the default 0.5
   threshold only.

### 6.3 Visualizations (regenerated from the corrected 2015–2024 results)

All 12 plots were regenerated from the corrected 2015–2024 results and corrected test
predictions in `08_Visualizations_FINAL_2015_2024/` (the earlier `08_Visualizations/`
plots, generated from the original scope, are retained as sensitivity reference only):

1. `01_class_dist_analytical.png` — analytical-set class distribution (n = 2,297,168)
2. `02_class_dist_random_oversampling.png` — training distribution after ROS
3. `03_class_dist_smote.png` — training distribution after SMOTE
4. `04_class_dist_adasyn.png` — training distribution after ADASYN
5. `05_confusion_matrices.png` — 4 techniques × 3 models (corrected test set, n = 459,434)
6. `06_roc_curves.png` — ROC curves per technique
7. `07_pr_curves.png` — Precision-Recall curves per technique
8. `08_accuracy_comparison.png` — Accuracy comparison
9. `09_recall_comparison.png` — Recall comparison
10. `10_f1_comparison.png` — F1-score comparison
11. `11_roc_auc_comparison.png` — ROC-AUC comparison
12. `12_pr_auc_comparison.png` — PR-AUC comparison

All axes are honest (bars from 0); the consolidated CSVs remain authoritative.

## 7. Validation

The MAIN validation below describes the corrected 2015–2024 experiment:

- Train/test split performed before any resampling: confirmed.
- ROS / SMOTE / ADASYN applied to training data only: confirmed (train-only asserts in code).
- Test set never resampled: confirmed — the same untouched corrected test set
  (n = 459,434; No 381,068 / Yes 78,366) was used for all four experiments, and the
  y_true fingerprint was re-verified byte-identical after every experiment.
- No target-derived feature leakage: target column asserted absent from all feature matrices.
- Random seeds recorded: 42 everywhere (config, code, logs).
- Results: confusion-matrix totals == 459,434 for all 12 model×technique cells;
  Accuracy == (TN+TP)/n; F1 == 2PR/(P+R) to 1e-9; all metrics finite and within [0,1];
  no NaN/inf; all 12 expected result rows present.
- SMOTE/ADASYN synthetic validity audit passed: 0 invalid synthetic categorical codes
  under the strict audit (see §12).
- Original-scope validation numbers (test n = 499,641) apply only to the sensitivity
  analysis in Appendix A.

## 8. Implementation Issues

### A. Original-scope issue
The first analysis used the master dataset as-is, so 201,035 YEAR=2014 rows (8.05%)
entered both the training set (160,553 rows) and the test set (40,482 rows) — outside the
intended 2015–2024 research scope.

### B. Correction
2014 was excluded and the complete primary analysis was rerun on the 2015–2024 working set
(2,297,168 rows) with the identical protocol (seed 42, stratified 80/20, same models):
new train 1,837,734 / new frozen test 459,434. All conclusions were unchanged
(max |Δ| = 0.0074 across all 72 metric cells). See §11.

### C. Synthetic feature audit
SMOTE and ADASYN synthetics were audited against each feature's actually observed
categorical code set (strict rule: fractional or unobserved codes = invalid; range-only
validity rejected). Result: **0 invalid codes** in ~39M synthetic cells across both methods
and both scopes; affected features: none. Full discussion in §12.

### D. Other technical issues (reproducibility-relevant only)
1. **No Group 1 artifacts found.** Drive and ~/workspace were searched for Group 1
   preprocessing code, notebooks, and results; none exist. Only prior-session Person-1
   handoff artifacts were found and used as methodology reference. → Cannot claim to
   reproduce Group 1; a clean defensible protocol (seed 42, stratified 80/20,
   LR/RF/XGBoost) is documented instead.
2. **Raw-XPT 7/9 filter not applied.** Codes 7/9 in the master appear only in _AGEG5YR,
   where they are valid age categories (50–54, 60–64). Applying the raw-XPT filter would
   wrongly drop ~20% of valid rows. → Filter not applied; documented.
3. **SMOTENC rejected all-categorical features** (imbalanced-learn ValueError) →
   standard SMOTE on integer codes; validity confirmed by strict audit (§12).
4. **ADASYN did not reach exact parity** (final minority 1,514,930 vs majority
   1,524,272; +1,201,468 rows) — known ADASYN behavior; not forced.
5. **Environment-only issues** (no impact on results): pip required --break-system-packages
   (PEP-668); xgboost wheel initially failed on a 512 MB tmpfs — fixed via TMPDIR under home;
   the audit rerun script's first launch failed at parse time on a stray parenthesis —
   fixed and relaunched clean (no partial outputs written).

## 9. Research Interpretation

The four approaches were compared under strictly identical conditions on the corrected
2015–2024 dataset, so observed differences are attributable to the balancing treatment, not
to data, features, or evaluation differences. Objectively:

- **Doing nothing** yields the highest accuracy (0.83–0.84) but clinically poor minority
  detection (recall ≤ 0.14, F1 ≤ 0.22): the models largely predict the majority class.
- **Any balancing** roughly doubles F1 (to 0.43–0.45) by converting the models from
  majority-class predictors into minority-class detectors, at a cost of ~0.14–0.19
  accuracy points and ~0.24–0.31 precision points.
- **Between the three balancing methods the differences are small**: ROS and SMOTE are
  effectively interchangeable here; ADASYN pushes recall highest for tree models at a
  slightly larger precision cost.
- **ROC-AUC/PR-AUC stability** shows the underlying ranking signal (XGBoost ≥
  RandomForest > LogisticRegression) is robust to the resampling choice.

These conclusions were also checked against the original 2014–2024 sensitivity analysis
(Appendix A): excluding 2014 moved no metric by more than 0.0074, and the technique
ranking, the ROS≈SMOTE finding, and the ADASYN-recall finding are identical in both scopes.

No claim is made that any technique is universally best: the results hold for this
dataset, these features, these models, and the default 0.5 threshold. Threshold tuning,
cost-sensitive training, or calibration could change the operating-point trade-offs.

## 10. Year-scope audit

**Finding:** the master file contains all years 2014–2024. YEAR=2014 contributes
201,035 rows (8.0472%). The original analysis used the master as-is, so all four
experiments trained on and were evaluated against 2014 data (160,553 rows in the original
training set, 40,482 in the original test set).

**Proof:** (i) the saved frozen split was re-joined to the master on row index — 2014 rows
present in both train and test; (ii) the exact original preprocessing + split was
recomputed (seed 42, stratified 80/20) and produced byte-identical indices and sizes
(train = 1,998,562 / test = 499,641). Reproduction assertion: PASS.

**Correction:** 2015–2024 working set (201,035 rows dropped → 2,297,168 rows); full protocol
rerun identically → new frozen test set n = 459,434 (No 381,068 / Yes 78,366); same QC
battery, all PASS. Results: `11_Audit_2015_2024_Rerun/`.

**Impact:** negligible — max |Δ| = 0.0074 vs the original table; all findings unchanged.

## 11. Sensitivity vs primary scope

The original 2014–2024 analysis (Appendix A) is retained **only as a sensitivity analysis**,
not as a result. The corrected 2015–2024 table (§6.1) is the primary result for the thesis.
Maximum absolute metric difference between scopes across all 72 cells: **0.0074**
(largest shift: No Balancing / RandomForest precision, 0.6149 → 0.6076).

## 12. Synthetic-feature validity audit and discussion

**Question:** standard SMOTE/ADASYN interpolate continuous values, but all 15 BRFSS
predictors are integer-coded categoricals (e.g. EXERANY2 ∈ {1,2}, MARITAL ∈ {1..6},
_AGEG5YR ∈ {5..14}). Do the synthetic samples contain fractional or otherwise invalid codes?

**Method (strict):** SMOTE and ADASYN were regenerated deterministically (seed 42) on both
scopes. Reproduction was verified first: resampled class counts exactly equal the saved
`class_distribution.json` values, and `X_res[:n_orig] == X_train` exactly, so rows
`X_res[n_orig:]` are precisely the synthetic samples. For each of the 15 features, the
valid set = unique values observed in the original training data; a synthetic value is
**invalid** if it is non-integer OR not in the valid set (being inside [min, max] is not
sufficient). Per-feature tables: `12_Audit_Feature_Validity/*/`*_per_feature_validity.csv`;
synthetic samples persisted as parquet.

**Result:**

| Scope | Method | Synthetic rows | Rows with ≥1 invalid feature value | Affected features |
|---|---|---|---|---|
| Original (2014–2024) | SMOTE | 1,318,498 | 0 (0.0%) | — none — |
| Original (2014–2024) | ADASYN | 1,301,378 | 0 (0.0%) | — none — |
| Corrected (2015–2024) | SMOTE | 1,210,810 | 0 (0.0%) | — none — |
| Corrected (2015–2024) | ADASYN | 1,201,468 | 0 (0.0%) | — none — |

**Zero invalid codes** across ~39M synthetic feature cells (both methods, both scopes):
no fractional values observed in the persisted synthetic outputs, no integer-but-unobserved
codes, no out-of-range codes. **Affected features: none.**

**Mechanism note (verified in imbalanced-learn 0.14.2 source):** both
`BaseSMOTE._generate_samples` and `ADASYN._fit_resample` end with
`X_new = X_new.astype(X.dtype)` — interpolated floats are **silently truncated toward zero**
to the input dtype (int64 here), not rounded. This is disclosed, not hidden. Additionally,
on this data ~93% of SMOTE synthetic rows exactly duplicate an existing training profile
(their k-nearest neighbours are exact duplicates, so no interpolation occurs at all), which
is why SMOTE behaves nearly identically to ROS here. Practical impact is negligible:
synthetic code distributions closely match the minority-class training distributions
(e.g. EXERANY2 0.612/0.388 → 0.645/0.355; _AGEG5YR per-code shares within 0.005).

**Caveat (not overstated):** this 0%-invalid finding is specific to these consecutive-integer
codes and this imbalanced-learn behavior; it is **not** proof that standard SMOTE is
universally appropriate for categorical data. SMOTE-N is the categorical-valid alternative
and remains assigned to the responsible team member; it is unnecessary for this feature
representation.

## 13. Final recommendation

**The Person 1 SMOTE/ADASYN results are methodologically acceptable for the thesis,**
with the following disclosed caveats (all documented above, none hidden):

1. **Reported metrics are honest regardless of synthetic-sample mechanics** — every
   metric is computed on real, untouched test observations; the test set was never
   resampled in any experiment (fingerprints re-verified after every run, both scopes).
2. **ROS is categorical-valid by construction** (exact row replication); SMOTE/ADASYN
   pass the strict per-feature validity audit with 0% invalid codes on both scopes.
3. **The silent int-truncation in imbalanced-learn's SMOTE/ADASYN must be disclosed**
   (done in §12): interpolated values are truncated toward zero rather than rounded.
   Impact here is negligible, but the thesis should state the mechanism rather than
   imply textbook continuous interpolation on categorical codes.
4. **Scope:** cite the corrected 2015–2024 results (§6.1) as primary; the original
   2014–2024 analysis is a valid sensitivity check (Appendix A).
5. **No re-implementation is recommended**: do not round/clip synthetic values (would
   alter the published-method comparison), do not switch to SMOTE-N (assigned to another
   group; unnecessary here), and do not claim these results generalize beyond this
   dataset, feature set, models, and the 0.5 decision threshold.

---

## Appendix A — Original 2014–2024 results (sensitivity reference ONLY)

Preserved from the original analysis (shared untouched test set n = 499,641; seed 42).
Superseded as the primary table by §6.1. Kept so the year-scope correction is auditable.
**These numbers describe the sensitivity analysis, not the primary experiment.**

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

---

## Thesis-ready methodology paragraph

Class imbalance was investigated on the Behavioral Risk Factor Surveillance System
(BRFSS) 2015–2024 analytical dataset (n = 2,297,168; 82.94% no diabetes, 17.06% diabetes;
imbalance ratio 4.86:1), derived from the BRFSS master source (n = 2,498,203) by excluding
2014 observations, which lie outside the intended research scope. The binary target was
diabetes status (DIABETE4), with fifteen integer-coded categorical predictors; YEAR was
excluded from the predictor set. After complete-case verification and target separation
with no target-derived feature leakage, the data were partitioned once via a stratified
80/20 train/test split (random seed 42), yielding a training set of 1,837,734 observations
and a frozen, never-resampled test set of 459,434 observations. Four class-imbalance
treatments were applied to the training data only — no balancing (baseline), random
oversampling, SMOTE, and ADASYN — and evaluated with three classifiers held constant
across all experiments: logistic regression (standardized features), random forest
(100 trees, maximum depth 12), and XGBoost (300 rounds, maximum depth 6, learning rate
0.05), all with random state 42. Performance was measured on the identical untouched
test set using accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion
matrices at the default 0.5 decision threshold. The protocol was validated for absence
of train/test leakage (split strictly preceding any resampling), metric consistency
(confusion-matrix totals equal to the test size; F1 identities), seed reproducibility, and
synthetic-sample categorical validity (a strict per-feature audit of all SMOTE/ADASYN
outputs against observed category codes found zero invalid values).

---

### File map (all under `~/workspace/Person_1_Class_Imbalance/`)

**PRIMARY FINAL OUTPUTS (2015–2024)**
- `10_Final_Report/Person_1_Final_Report_FINAL.md` — this report (primary, thesis-ready)
- `11_Audit_2015_2024_Rerun/` — authoritative corrected outputs: `01_year_audit/`
  (year counts), `02_preprocessing_audit/` (corrected X/y train/test splits + protocol config),
  `03_baseline_audit/` … `06_adasyn_audit/` (per-experiment results, class distributions,
  test predictions), `07_combined_results_audit/combined_results_2015_2024.csv`
- `08_Visualizations_FINAL_2015_2024/` — 12 plots regenerated from corrected results
- `09_Code/audit/` — `01_year_audit.py`, `02_rerun_2015_2024.py`, `03_feature_validity.py`,
  `04_build_report_v2.py`, `05_regenerate_plots_2015_2024.py`
- `validation.log` — full execution/validation/cleanup log

**AUDIT / SENSITIVITY OUTPUTS (original 2014–2024 scope; preserved, not primary)**
- `10_Final_Report/Person_1_Final_Report_ORIGINAL_SCOPE.md` — v1 report, verbatim (untouched)
- `10_Final_Report/Person_1_Final_Report_v2_AUDITED.md` — v2 audited report, byte-identical copy
- `10_Final_Report/Person_1_Final_Report.md` — v2 audited report (superseded by FINAL)
- `01_Data_Inspection/` … `09_Code/` (01–05), `07_Combined_Results/`,
  `08_Visualizations/`, `validation.log` (original entries) — original-scope artifacts
- `12_Audit_Feature_Validity/` — SMOTE/ADASYN synthetic-value validity audit (both scopes)
