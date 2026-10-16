# Person 3: Class Imbalance Resampling Techniques for BRFSS Diabetes Prediction

**Status**: ✓ COMPLETE — All 12 experiments finished, all artifacts validated, production-ready.

## Overview

Person 3 evaluated four class-imbalance resampling techniques applied to BRFSS diabetes prediction:

| Technique | Type | Mechanism | Best For |
|-----------|------|-----------|----------|
| **ENN** | Undersampling | Removes noisy majority samples via k-NN voting | Balanced F1 (0.4420) |
| **Tomek Links** | Undersampling | Removes borderline majority samples | High precision (0.6023), high accuracy (0.8354) |
| **SMOTE-Tomek** | Over+Under | SMOTE synthetic + Tomek removal | High recall (0.7319) |
| **SMOTE-ENN** | Over+Under | SMOTE synthetic + ENN removal | Balanced accuracy (0.8239) |

Each technique was evaluated with three models: Logistic Regression, Random Forest, and XGBoost.

**Result**: 12 complete experiments (4 techniques × 3 models).

---

## Dataset & Protocol

**Data Source**: BRFSS 2015–2024 (corrected)

**Training Set**: 1,837,734 rows (82.9% negative / 17.1% positive)
**Test Set**: 459,434 rows (frozen, never resampled)

**Predictors**: 15 behavioral/clinical features (integer codes)
**Target**: DIABETE4 (No=0, Yes=1)

**Split**: Stratified 80/20 with `random_state=42`

**Test Fingerprint** (SHA-256):
```
70e180c659c74efbb6ac6368f5b061c9a94cf7efc10a707f52abcf29280b0d11
```

---

## Results

### Overall Performance

**Best Scores by Metric**:
- **Accuracy**: 0.8354 (Tomek Links + XGBoost)
- **Precision**: 0.6023 (Tomek Links + RandomForest)
- **Recall**: 0.7319 (SMOTE-Tomek + RandomForest)
- **F1 Score**: 0.4420 (ENN + XGBoost)
- **ROC-AUC**: 0.7803 (Tomek Links + XGBoost)
- **PR-AUC**: 0.4176 (Tomek Links + XGBoost)

### Complete Results Table

See `results/results_person3.csv` (12 rows with all metrics).

### Visualizations

All publication-ready figures in `figures/`:

1. `01_f1_comparison.png/.pdf` — F1 across techniques/models
2. `02_recall_comparison.png/.pdf` — Recall across techniques/models
3. `03_precision_comparison.png/.pdf` — Precision across techniques/models
4. `04_pr_auc_comparison.png/.pdf` — PR-AUC across techniques/models
5. `05_roc_auc_comparison.png/.pdf` — ROC-AUC across techniques/models
6. `06_accuracy_comparison.png/.pdf` — Accuracy across techniques/models
7. `07_class_distribution.png/.pdf` — Before/after class counts per technique
8. `08_precision_recall_tradeoff.png/.pdf` — Precision vs. recall by technique

---

## File Structure

```
person3-brfss/
├── README_PERSON3.md              # This file
├── PROTOCOL.md                    # Frozen methodology
├── requirements.txt               # Python dependencies
│
├── data/
│   ├── manifest.json              # Dataset metadata + fingerprint
│   ├── X_train.npy / y_train.npy  # Training set (1.8M samples)
│   └── X_test.npy / y_test.npy    # Test set (frozen, 459k samples)
│
├── src/
│   ├── 01_prepare.py              # Dataset preparation
│   ├── 02_run.py                  # Original main runner
│   ├── 02_run_with_staged_smote_enn.py  # Updated runner (SMOTE-ENN staged)
│   ├── 03_workbook.py             # Workbook generation
│   ├── common.py                  # Shared utilities
│   ├── smote_enn_staged.py        # Staged SMOTE-ENN implementation
│   └── validate_smote_enn_staged.py    # Pilot validation
│
├── results/
│   ├── results_person3.csv        # ✓ PRIMARY: 12 experiment results
│   ├── test_predictions_enn.parquet
│   ├── test_predictions_tomek.parquet
│   ├── test_predictions_smote_tomek.parquet
│   ├── test_predictions_smote_enn.parquet
│   ├── class_distributions.json   # Before/after class counts
│   ├── audit.json                 # Resampling audit trail
│   ├── checkpoint.json            # Experiment status
│   ├── environment.json           # Python packages + platform
│   ├── artifact_inventory.json    # File inventory
│   ├── Person_3_Results_and_Discussion.md
│   ├── Person_3_Final_Report.md
│   └── GROUP_INTEGRATION_COMPATIBILITY.txt
│
├── figures/
│   ├── 01_f1_comparison.png/.pdf
│   ├── 02_recall_comparison.png/.pdf
│   ├── 03_precision_comparison.png/.pdf
│   ├── 04_pr_auc_comparison.png/.pdf
│   ├── 05_roc_auc_comparison.png/.pdf
│   ├── 06_accuracy_comparison.png/.pdf
│   ├── 07_class_distribution.png/.pdf
│   └── 08_precision_recall_tradeoff.png/.pdf
│
└── workbook/
    └── Person_3_results_workbook.xlsx  # Excel with all results
```

---

## How to Use

### View Results

**Quick look**:
```bash
cat results/results_person3.csv
```

**Full report**:
```bash
less results/Person_3_Final_Report.md
```

**Excel workbook**:
```bash
open workbook/Person_3_results_workbook.xlsx
```

### Inspect Predictions

```python
import pandas as pd

# Load predictions for a specific technique
df = pd.read_parquet('results/test_predictions_enn.parquet')
print(df.columns)  # y_true, LogisticRegression__pred, ..., XGBoost__proba
print(df.shape)    # (459434, 7)
```

### Regenerate Workbook

```bash
python3 src/03_workbook.py \
  --results-dir results \
  --data-dir data \
  --out workbook/Person_3_results_workbook.xlsx
```

### Reproduce Full Analysis

```bash
# Re-run all 12 experiments
python3 src/02_run_with_staged_smote_enn.py --data-dir data --out-dir results

# Verify checkpoint
cat results/checkpoint.json
```

---

## Methodology Notes

### Resampling Techniques

All from imbalanced-learn 0.14.2 with `sampling_strategy="auto"`:

- **ENN**: `EditedNearestNeighbours(n_neighbors=3, kind_sel="all")`
- **Tomek Links**: `TomekLinks()`
- **SMOTE-Tomek**: `SMOTETomek(random_state=42, smote=SMOTE(random_state=42, k_neighbors=5))`
- **SMOTE-ENN**: `SMOTEENN(random_state=42, smote=SMOTE(...), enn=EditedNearestNeighbours(...))`

### Models

- **Logistic Regression**: `solver="lbfgs", max_iter=1000, random_state=42` + StandardScaler (fitted on resampled train)
- **Random Forest**: `n_estimators=100, max_depth=12, n_jobs=-1, random_state=42`
- **XGBoost**: `n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=42`

### Key Points

1. **Test set frozen**: Never resampled, balanced, or augmented. Fingerprint verified before/after experiments.
2. **Deterministic**: `random_state=42` throughout ensures reproducibility.
3. **Staged SMOTE-ENN**: Implemented as explicit two-stage pipeline with checkpointing for robustness. 
   Scientific operation unchanged; produces bit-identical results to sequential application.
4. **All metrics computed on frozen test set** at decision threshold 0.5.

---

## Environment

**Python**: 3.12.x
**scikit-learn**: 1.5.x
**imbalanced-learn**: 0.14.2
**XGBoost**: Latest (2024)
**Platform**: macOS (Apple Silicon), 16 GB RAM

See `results/environment.json` for exact versions.

---

## Limitations

1. **Single train/test split** — no cross-validation variance assessment
2. **Hyperparameters fixed per protocol** — not grid-tuned
3. **SMOTE synthetic samples** — not validated for clinical realism
4. **Decision threshold fixed at 0.5** — no threshold optimization
5. **Test set retains original imbalance** — results don't reflect balanced test scenarios

---

## Integration with Person 1 & Person 2

Person 3 results are **directly mergeable** with Person 1 experiments:

✓ Same dataset (BRFSS 2015–2024)
✓ Same train/test split (80/20, random_state=42)
✓ Same 15 predictors (integer codes)
✓ Same 3 models (LR, RF, XGB)
✓ Same metrics and CSV format
✓ Same evaluation methodology

**Caveats**:
- Confirm Person 1 used `random_state=42` for split
- Exchange test-set fingerprints to verify identical data
- SMOTE-ENN staging is engineering detail; scientific operation identical

See `results/GROUP_INTEGRATION_COMPATIBILITY.txt` for detailed alignment.

---

## Quick Reference

| What | Where |
|------|-------|
| Results (12 experiments) | `results/results_person3.csv` |
| Complete report | `results/Person_3_Final_Report.md` |
| Excel workbook | `workbook/Person_3_results_workbook.xlsx` |
| Visualizations | `figures/` (PNG + PDF) |
| Test predictions | `results/test_predictions_*.parquet` |
| Protocol spec | `PROTOCOL.md` |
| Integration notes | `results/GROUP_INTEGRATION_COMPATIBILITY.txt` |

---

## Contact & Questions

All 12 experiments are complete and verified. For questions about methodology, results, or reproduction, 
refer to:

1. **PROTOCOL.md** — Detailed methodology specification
2. **Person_3_Final_Report.md** — Comprehensive findings and interpretation
3. **SOURCE CODE** — `src/02_run_with_staged_smote_enn.py` and `src/common.py`

---

**Status**: Production-ready, thesis-ready, merge-ready. ✓

