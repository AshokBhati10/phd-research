"""
STEP 9 - Full quality-control audit. Runs after all experiments + visualizations.
Checks: dataset integrity, leakage, reproducibility markers, results correctness,
visualization presence, master-dataset immutability.
Appends PASS/FAIL lines to validation.log. Exits nonzero if any FAIL.
"""
import json, os, sys
import numpy as np
import pandas as pd

BASE = "/home/hatch/workspace/Person_1_Class_Imbalance"
log = [f"# STEP 9 QC AUDIT"]
fails = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    log.append(f"[{status}] {name} {detail}".rstrip())
    print(f"[{status}] {name} {detail}".rstrip(), flush=True)
    if not cond:
        fails.append(name)

# --- 1. expected files ---
expected = [
    "01_Data_Inspection/column_inventory.csv", "01_Data_Inspection/inspection_log.txt",
    "02_Preprocessing/X_train.parquet", "02_Preprocessing/X_test.parquet",
    "02_Preprocessing/y_train.parquet", "02_Preprocessing/y_test.parquet",
    "02_Preprocessing/protocol_config.json",
    "03_Baseline/results_no_balancing.csv", "03_Baseline/class_distribution.json",
    "03_Baseline/test_predictions.parquet",
    "04_Random_Oversampling/results_random_oversampling.csv",
    "04_Random_Oversampling/class_distribution.json", "04_Random_Oversampling/test_predictions.parquet",
    "05_SMOTE/results_smote.csv", "05_SMOTE/class_distribution.json", "05_SMOTE/test_predictions.parquet",
    "06_ADASYN/results_adasyn.csv", "06_ADASYN/class_distribution.json", "06_ADASYN/test_predictions.parquet",
    "07_Combined_Results/combined_results.csv", "07_Combined_Results/class_distributions.json",
    "09_Code/01_data_inspection.py", "09_Code/02_preprocess_split.py",
    "09_Code/03_experiments.py", "09_Code/04_visualizations.py", "09_Code/05_qc.py",
    "10_Final_Report/Person_1_Final_Report.md",
]
for f in expected:
    check(f"file present: {f}", os.path.exists(f"{BASE}/{f}"))
for i in range(1, 13):
    pngs = [f for f in os.listdir(f"{BASE}/08_Visualizations") if f.startswith(f"{i:02d}_")]
    check(f"visualization {i:02d} present", len(pngs) == 1 and pngs[0].endswith(".png"),
          pngs[0] if pngs else "")

# --- 2. master dataset untouched ---
master = "/home/hatch/workspace/brfss_data/brfss_master.parquet"
check("master dataset exists", os.path.exists(master))
check("master dataset size unchanged (8647149 bytes)",
      os.path.getsize(master) == 8647149, f"size={os.path.getsize(master)}")

# --- 3. test set untouched across all experiments ---
y_test = pd.read_parquet(f"{BASE}/02_Preprocessing/y_test.parquet")["y"].values
X_test_cols = list(pd.read_parquet(f"{BASE}/02_Preprocessing/X_test.parquet").columns)
for sub in ["03_Baseline", "04_Random_Oversampling", "05_SMOTE", "06_ADASYN"]:
    pdf = pd.read_parquet(f"{BASE}/{sub}/test_predictions.parquet")
    check(f"{sub}: test y_true identical to frozen test set",
          (pdf["y_true"].values == y_test).all(), f"n={len(pdf)}")
    for m in ["LogisticRegression", "RandomForest", "XGBoost"]:
        check(f"{sub}: {m} proba has no NaN/inf", np.isfinite(pdf[f"{m}__proba"].values).all())
        check(f"{sub}: {m} proba in [0,1]",
              bool(((pdf[f"{m}__proba"].values >= 0) & (pdf[f"{m}__proba"].values <= 1)).all()))

# --- 4. combined results correctness ---
comb = pd.read_csv(f"{BASE}/07_Combined_Results/combined_results.csv")
check("combined results has 12 rows (4 techniques x 3 models)", len(comb) == 12, f"rows={len(comb)}")
check("all 4 techniques present",
      set(comb["Technique"]) == {"No Balancing", "Random Oversampling", "SMOTE", "ADASYN"})
check("all 3 models present",
      set(comb["Model"]) == {"LogisticRegression", "RandomForest", "XGBoost"})
for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]:
    check(f"{metric}: no NaN/inf, in [0,1]",
          bool(np.isfinite(comb[metric]).all() and (comb[metric] >= 0).all() and (comb[metric] <= 1).all()))
# confusion-matrix totals == test size; metric math sanity
for _, r in comb.iterrows():
    cm_sum = r["TN"] + r["FP"] + r["FN"] + r["TP"]
    check(f"CM total == test size ({r['Technique']}/{r['Model']})",
          cm_sum == 499641, f"total={cm_sum}")
    acc = (r["TN"] + r["TP"]) / cm_sum
    check(f"Accuracy == (TN+TP)/n ({r['Technique']}/{r['Model']})", abs(acc - r["Accuracy"]) < 1e-9)
    p, rc = r["Precision"], r["Recall"]
    if p + rc > 0:
        f1 = 2 * p * rc / (p + rc)
        check(f"F1 == 2PR/(P+R) ({r['Technique']}/{r['Model']})", abs(f1 - r["F1"]) < 1e-6)

# --- 5. class-distribution consistency ---
dists = json.load(open(f"{BASE}/07_Combined_Results/class_distributions.json"))
maj = {int(k): int(v) for k, v in dists["No Balancing"]["train_before"].items()}[0]
minc = {int(k): int(v) for k, v in dists["No Balancing"]["train_before"].items()}[1]
check("baseline train distribution matches split (No=1658530, Yes=340032)",
      maj == 1658530 and minc == 340032)
for tech in ["Random Oversampling", "SMOTE"]:
    ta = {int(k): int(v) for k, v in dists[tech]["train_after"].items()}
    check(f"{tech}: minority resampled to majority size ({maj})", ta[1] == maj,
          f"after={ta}")
ada = {int(k): int(v) for k, v in dists["ADASYN"]["train_after"].items()}
check("ADASYN: documented non-parity (minority < majority)",
      0 < ada[1] <= maj, f"after={ada}")
for tech in ["Random Oversampling", "SMOTE", "ADASYN"]:
    added = dists[tech]["added_samples"]
    ta = {int(k): int(v) for k, v in dists[tech]["train_after"].items()}
    check(f"{tech}: added == after-before", added == (ta[0] + ta[1]) - (maj + minc),
          f"added={added}")

# --- 6. reproducibility markers ---
cfg = json.load(open(f"{BASE}/02_Preprocessing/protocol_config.json"))
check("random_state=42 recorded in config", cfg["random_state"] == 42)
check("train/test sizes recorded", cfg["n_train"] == 1998562 and cfg["n_test"] == 499641)
check("15 features recorded", cfg["n_features"] == 15 and len(cfg["features"]) == 15)
check("test set size == frozen fingerprint", len(y_test) == 499641 and int(y_test.sum()) == 85008)

with open(f"{BASE}/validation.log", "a") as f:
    f.write("\n" + "\n".join(log) + "\n")
print(f"\nQC complete: {len(fails)} FAILURES" if fails else "\nQC complete: ALL CHECKS PASSED")
sys.exit(1 if fails else 0)
