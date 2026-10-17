# Person 4 — Integration & Evaluation

Integrated, statistically tested comparison of the three class-imbalance approaches on the
identical frozen BRFSS 2015–2024 test set (459,434 rows, SHA-256 `70e180c6…`):

- **P1 — basic oversampling:** No Balancing (baseline), Random Oversampling, SMOTE, ADASYN
- **P2 — advanced oversampling:** Borderline-SMOTE, KMeans-SMOTE, SVM-SMOTE†, SMOTE-N†
- **P3 — cleaning & hybrid:** Tomek Links, ENN, SMOTE-Tomek, SMOTE-ENN

**36 result rows = 12 techniques × 3 classifiers**, every row traced to a validated source
artifact and re-verified (`results/input_validation.json`).

## Contents

| Path | What |
|---|---|
| `PROTOCOL_P4.md` | Frozen integration rules, fairness register, statistical methods |
| `report/Person_4_Integration_Report.md` | **The final deliverable** — plain-language research report |
| `workbook/Person_4_Integration_Workbook.xlsx` | Concise results workbook (11 sheets) |
| `figures/` | 6 figures × (PNG + PDF) |
| `results/` | combined_results.csv, statistical matrices, validation reports, report_tables/ |
| `metadata/` | sources.json (provenance registry), environment.json |
| `code/` | Reproducible pipeline (see below) |

## Start here

1. Open `report/Person_4_Integration_Report.md` — executive summary, fairness assessment,
   integrated tables, statistical analysis, findings, limitations, traceability.
2. The workbook mirrors the report tables plus the full 198-row McNemar matrix.

## Reproduce

```powershell
$py = "P2_Advanced_Oversampling/.venv/Scripts/python.exe"
& $py P4_Integration_Evaluation/code/01_verify_inputs.py       # gate 1: 82 checks
& $py P4_Integration_Evaluation/code/02_integrate_results.py
& $py P4_Integration_Evaluation/code/03_statistical_tests.py --bootstrap
& $py P4_Integration_Evaluation/code/04_figures.py
& $py P4_Integration_Evaluation/code/05_workbook.py
& $py P4_Integration_Evaluation/code/06_build_report.py
& $py P4_Integration_Evaluation/code/07_validate_p4.py         # gate 2
```

P1–P3 folders are never modified; gate 2 verifies their git status is clean.

## Headline (see report §1 for detail)

Resampling roughly doubles minority-class F1 (0.153–0.217 → 0.392–0.445). **Random
Oversampling is the F1 champion for all three classifiers**; recall peaks with ADASYN
(0.772); **ENN** preserves accuracy while reaching F1 0.442 (XGBoost); **Tomek Links ≈
baseline** (statistically indistinguishable, best ROC-AUC). Resampling mainly moves the
0.5-threshold operating point rather than improving probability ranking (PR-AUC XGB:
ROS 0.4176 vs baseline 0.4176). † = trained on a 50k subsample (documented deviation).
