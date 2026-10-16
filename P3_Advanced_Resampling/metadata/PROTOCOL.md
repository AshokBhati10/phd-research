# Person 3 — Frozen Methodology (BRFSS 2015–2024, class-imbalance techniques)

Status: FROZEN for pilot. No full-scale execution until pilot is reviewed and approved.
Reference implementation: Person 1 workspace (Drive), final report primary scope = corrected 2015–2024 rerun.

## 1. Data scope (fixed)

- Source: `~/workspace/user/files/brfss_master__1.parquet` (read-only; never modified).
- Exclude `YEAR == 2014` completely. Expected analytical dataset: **2,297,168 rows × 17 columns**.
- Target `DIABETE4`: `"Yes" → 1`, `"No" → 0`.
- Predictors (15, exact Person 1 list): EXERANY2, _MICHD, ASTHMA3, ADDEPEV3, CHCKDNY2, MARITAL,
  _EDUCAG, DRNKANY6, _AGEG5YR, _BMI5CAT, _INCOMG1, _RFSMOK3, GENHLTH, HAVARTH4, SEXVAR.
- `YEAR` is metadata only, never a predictor.
- Representation (Person 1, `02_preprocess_split.py`): features `astype(int)` (int64),
  integer-coded categoricals preserved, no one-hot, no target encoding, no scaling before resampling.
- Validation asserts before split: no nulls anywhere; `DIABETE4 ∈ {"No","Yes"}`; `_AGEG5YR >= 5`.

## 2. Train/test protocol (fixed)

- `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y, shuffle=True)`.
- Expected: train **1,837,734** (No 1,524,272 / Yes 313,462), test **459,434** (No 381,068 / Yes 78,366).
- The test set is frozen at prepare time. A SHA-256 fingerprint of `(X_test, y_test)` is stored in
  `data/manifest.json`; every run verifies it before and after experiments and aborts on mismatch.
- Resampling, balancing, augmentation, scaler fitting: **training data only, never the test set**.

## 3. Person 3 techniques (exact)

imbalanced-learn implementations. `random_state=42` wherever the estimator supports it;
otherwise library defaults. No invented hyperparameters.

| Technique | Constructor | Parameter provenance |
|---|---|---|
| ENN | `EditedNearestNeighbours(sampling_strategy="auto")` | `sampling_strategy` source-specified; `n_neighbors=3`, `kind_sel="all"` = library defaults (class exposes no `random_state`) |
| Tomek Links | `TomekLinks(sampling_strategy="auto")` | `sampling_strategy` source-specified; all else = library defaults (class exposes no `random_state`) |
| SMOTE-Tomek | `SMOTETomek(sampling_strategy="auto", random_state=42, smote=<P1 SMOTE>, tomek=TomekLinks(sampling_strategy="auto"))` | `random_state` source-specified; SMOTE component = Person 1's exact SMOTE (below); Tomek component = defaults |
| SMOTE-ENN | `SMOTEENN(sampling_strategy="auto", random_state=42, smote=<P1 SMOTE>, enn=EditedNearestNeighbours(sampling_strategy="auto"))` | `random_state` source-specified; SMOTE component = Person 1's exact SMOTE; ENN component = defaults |

Person 1's SMOTE (used as the SMOTE component above, and the documented reference):
`SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)`.
Explicit sub-samplers are passed (rather than relying on defaults) so the SMOTE component is
byte-identical to Person 1's; this is documented, not a custom invention. `n_jobs` left at library
default for all resamplers (execution-only parameter; Person 1 also left it unset).

Note on SMOTENC: rejected by Person 1 — imbalanced-learn 0.14.2 raises for all-categorical
feature sets. Standard SMOTE on integer codes is the documented convention. imbalanced-learn
silently truncates interpolated floats toward zero via `astype(X.dtype)` (verified in 0.14.2
source; disclosed in Person 1 report §12) — fractional codes never persist.

## 4. Models (exact, per technique)

| Model | Parameters | Scaling |
|---|---|---|
| LogisticRegression | `max_iter=1000, solver="lbfgs", random_state=42` | `StandardScaler` **fitted on the resampled training data** (Person 1's actual code: `StandardScaler().fit(Xr)` after resampling), applied to train and test |
| RandomForest | `n_estimators=100, max_depth=12, n_jobs=-1, random_state=42` | none (raw integer codes) |
| XGBoost | `n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=42` | none (raw integer codes) |

Decision threshold 0.5 for class predictions; `predict_proba` for ROC-AUC / PR-AUC.

## 5. Metrics (per technique × model)

Accuracy, Precision, Recall/Sensitivity, F1, ROC-AUC, PR-AUC (average precision),
TN, FP, FN, TP (from confusion matrix at threshold 0.5), `fit_predict_sec` (wall time of
fit + predict + predict_proba on the frozen test set). CSV: metrics 10 decimals, timing 1 decimal.

## 6. Resource constraints (Mac: ~7 GB RAM, 2 CPU)

- Experiments run strictly sequentially: one technique at a time, one model at a time.
- `X_train`/`y_train` loaded once per technique; resampled arrays released (`del` + `gc.collect()`)
  before the next technique. Never hold two resampled datasets simultaneously.
- `X_test`/`y_test` loaded once per run (55 MB), treated read-only.
- Feature dtype int64 (Person 1 fidelity): train ≈ 221 MB; largest resampled train (SMOTE-ENN path)
  ≈ 370 MB. Peak expected during RandomForest fit; well under budget on a quiet machine.
- Checkpointing (see §8) means no experiment is ever rerun unnecessarily.

## 7. Pilot

- Stratified subsample of **20,000 rows** from the analytical dataset (seed 42), then the same
  80/20 stratified split → 16,000 train / 4,000 test. Rationale: preserves the 17% positive rate
  (~3,400 minority rows — far above SMOTE's `k_neighbors=5` minimum), exercises every code path
  including SMOTE generation and ENN/Tomek removal, runs in seconds, memory-negligible.
- Pilot validates: code correctness, imbalanced-learn 0.14.2 API compatibility, dtypes,
  determinism (resample twice → identical), no test leakage (fingerprint), metric math,
  output formats, checkpoint/restart (second run must skip everything), per-op timings.
- Pilot artifacts live in `pilot/` and are NEVER merged into `results/`. Pilot metrics are not
  scientific results.

## 8. Checkpoint / resume

- `results/checkpoint.json` + `results/results_person3.csv` (append-only).
- A (technique, model) pair is complete iff its row exists in the CSV. Reruns skip complete pairs;
  `--force` is required to redo. Partial/corrupt rows are never written (row appended only after
  successful completion; predictions parquet written to temp then atomically renamed).
- Resampling is deterministic (`random_state=42` everywhere supported); resume re-derives the
  resampled train rather than storing multi-GB intermediates on disk.
- Any run aborts if the test fingerprint mismatches the manifest.

## 9. Audit (per technique)

Before/after row counts, class counts before/after, rows removed, rows added/generated,
final train size, resample wall time, model fit/predict times, errors/warnings,
test-fingerprint confirmation. For SMOTE-involving techniques: Person 1's feature-validity
checks (finite values; synthetic values within observed per-feature [min, max]; strict
fractional/unobserved-code scan — reported as observed, never claimed without verification).

## 10. Outputs

- `results/results_person3.csv` — header `Technique,Model,Accuracy,Precision,Recall,F1,ROC_AUC,PR_AUC,TN,FP,FN,TP,fit_predict_sec`.
- `results/class_distributions.json` — per technique: technique, train_before, train_after, added_samples, final_train_size (+ rows_removed for undersamplers).
- `results/test_predictions_{enn,tomek,smote_tomek,smote_enn}.parquet` — `y_true`, `{Model}__pred`, `{Model}__proba` × 3.
- `results/audit.json`, `results/run.log`, `results/environment.json`, `results/checkpoint.json`.
- `workbook/Person_3_results_workbook.xlsx` — sheets: ENN, Tomek_Links, SMOTE_Tomek, SMOTE_ENN,
  Combined_results, Class_distributions, Protocol_config (+ run log text sheet).

## 11. Reproducibility

Pinned in `requirements.txt`; recorded in `results/environment.json` and the Protocol_config sheet:
Python, scikit-learn, imbalanced-learn (target 0.14.2, Person 1's version), XGBoost, pandas,
NumPy, pyarrow, openpyxl, plus platform/CPU/RAM and all seeds.

## 12. Open methodological points (require approval, not assumptions)

1. Resampler hyperparameters: proposal in §3 (defaults + `random_state=42`; SMOTE component =
   Person 1's exact SMOTE). Needs explicit sign-off since the protocol names techniques only.
2. `StandardScaler` fitted on resampled train follows Person 1's *code*; the report text says
   "training fold only". Proposal: follow the code. Needs sign-off.
3. Person 3 delivers the 4 techniques only (12 experiments); no "No Balancing" baseline
   (Person 1's scope).
