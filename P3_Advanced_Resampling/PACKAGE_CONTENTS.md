# Person 3 Handoff Package — Class Imbalance Resampling Techniques

**Status**: FINAL — All 12 experiments complete, all artifacts validated, ready for group integration.

## Quick Start

1. **View Results**: Open `results/results_person3.csv` (12 rows with all metrics)
2. **Full Report**: Read `report/Person_3_Final_Report.md`
3. **Excel Workbook**: Open `workbook/Person_3_results_workbook.xlsx`
4. **Visualizations**: View `figures/` (publication-ready PNG + PDF)

## Package Contents

### report/
- `Person_3_Final_Report.md` — Complete findings with interpretation
- `README_PERSON3.md` — Quick reference and usage guide
- `PROTOCOL.md` — Frozen methodology specification

### results/ (PRIMARY DATA)
- `results_person3.csv` — **Authoritative 12 experiment results**
- `test_predictions_*.parquet` — Model predictions on frozen test set (4 techniques)
- `class_distributions.json` — Before/after class counts per technique

### workbook/
- `Person_3_results_workbook.xlsx` — Excel with all results, protocol config, run log

### figures/ (PUBLICATION-READY)
- 8 visualizations (PNG @ 300 dpi + vector PDF format)
  - F1, Recall, Precision, PR-AUC, ROC-AUC, Accuracy comparisons
  - Class distribution before/after
  - Precision-recall trade-off

### code/
- `02_run_with_staged_smote_enn.py` — Main experiment runner
- `03_workbook.py` — Workbook generation
- `common.py` — Shared utilities
- `smote_enn_staged.py` — Staged SMOTE-ENN implementation
- `requirements.txt` — Python dependencies

### metadata/
- `PROTOCOL.md` — Methodology
- `environment.json` — Python versions + platform info
- `artifact_inventory.json` — Complete file manifest
- `GROUP_INTEGRATION_COMPATIBILITY.txt` — Person 1/2/3 alignment
- `manifest.json` — Dataset metadata + test fingerprint

### logs/
- Run logs from experiment execution

## Key Results

| Metric | Best Technique + Model | Value |
|--------|---|---|
| Accuracy | Tomek Links + XGBoost | 0.8354 |
| Precision | Tomek Links + RandomForest | 0.6023 |
| Recall | SMOTE-Tomek + RandomForest | 0.7319 |
| F1 Score | ENN + XGBoost | 0.4420 |
| ROC-AUC | Tomek Links + XGBoost | 0.7803 |
| PR-AUC | Tomek Links + XGBoost | 0.4176 |

**Key Finding**: No single optimal technique; resampling choice involves precision–recall trade-offs.

## Integration with Person 1/2

✓ All 12 Person 3 results are directly mergeable with Person 1 using (technique, model) index.

See `metadata/GROUP_INTEGRATION_COMPATIBILITY.txt` for detailed alignment.

## For Group Report

**Recommended files to include**:
1. `results/results_person3.csv` — Raw data
2. `report/Person_3_Final_Report.md` — Methodology & findings
3. `figures/` — All 8 visualizations for publication
4. `workbook/Person_3_results_workbook.xlsx` — Supplementary material

**Not needed for publication**:
- `logs/` — Execution logs (internal use only)
- `code/` — Reproducibility; include in appendix if space permits

## Reproducibility

To re-run all 12 experiments:

```bash
# Prerequisites
pip install -r code/requirements.txt

# Copy data and code to working directory
cp -r metadata/manifest.json data/
cp code/02_run_with_staged_smote_enn.py src/

# Run
python3 src/02_run_with_staged_smote_enn.py --data-dir data --out-dir results
```

**Expected outcome**: Identical results (deterministic, `random_state=42` throughout).

## Data Integrity

**Test Fingerprint** (SHA-256):
```
70e180c659c74efbb6ac6368f5b061c9a94cf7efc10a707f52abcf29280b0d11
```

Test set: frozen (never resampled), 459,434 rows, class distribution (No 381,068 / Yes 78,366).

## Contact

All questions about methodology, results, or reproduction refer to:
- `report/Person_3_Final_Report.md` (comprehensive findings)
- `report/PROTOCOL.md` (frozen specification)
- `metadata/GROUP_INTEGRATION_COMPATIBILITY.txt` (group alignment)

---

**Package prepared**: 2024
**Status**: Production-ready, thesis-ready, merge-ready. ✓

