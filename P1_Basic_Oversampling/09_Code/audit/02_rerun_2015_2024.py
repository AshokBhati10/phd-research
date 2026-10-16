"""
AUDIT 1b - Corrected 2015-2024 rerun.
Builds the corrected working set (master minus YEAR==2014) and reproduces the
ORIGINAL protocol exactly: same 15 features, same integer encoding, stratified
80/20 split seed 42, same models/hyperparameters, same resamplers seed 42.
Resampling is applied to TRAINING data only. Test set is frozen and never
resampled; its fingerprint is re-verified after every experiment.
Outputs go ONLY under 11_Audit_2015_2024_Rerun/. Original folders 01-10 untouched.
"""
import json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

MASTER = "/home/hatch/workspace/brfss_data/brfss_master.parquet"
BASE = "/home/hatch/workspace/Person_1_Class_Imbalance"
OUT = f"{BASE}/11_Audit_2015_2024_Rerun"
for sub in ["02_preprocessing_audit", "03_baseline_audit", "04_ros_audit",
            "05_smote_audit", "06_adasyn_audit", "07_combined_results_audit"]:
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
RNG = 42
FEATURES = ["EXERANY2", "_MICHD", "ASTHMA3", "ADDEPEV3", "CHCKDNY2",
            "MARITAL", "_EDUCAG", "DRNKANY6", "_AGEG5YR", "_BMI5CAT",
            "_INCOMG1", "_RFSMOK3", "GENHLTH", "HAVARTH4", "SEXVAR"]
TARGET = "DIABETE4"

print("Loading master (read-only)...", flush=True)
df = pd.read_parquet(MASTER)
n_master = len(df)
assert n_master == 2498203

# --- corrected working set: 2015-2024 ---
dropped = int((df["YEAR"] == 2014).sum())
dfc = df.loc[df["YEAR"] != 2014].copy()
assert len(dfc) == n_master - dropped
assert dfc["YEAR"].min() >= 2015 and dfc["YEAR"].max() == 2024
print(f"YEAR==2014 rows excluded: {dropped:,} -> corrected set: {len(dfc):,} rows (2015-2024)", flush=True)

# --- identical preprocessing ---
assert dfc["_AGEG5YR"].min() >= 5.0
assert set(dfc["DIABETE4"].unique()) == {"No", "Yes"}
assert dfc.isna().sum().sum() == 0
dfc = dfc.loc[dfc["_AGEG5YR"] >= 5].copy()
X = dfc[FEATURES].astype(int)
y = (dfc[TARGET] == "Yes").astype(int)
assert TARGET not in X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RNG, stratify=y, shuffle=True)
assert len(np.intersect1d(X_train.index, X_test.index)) == 0
assert list(X_train.columns) == list(X_test.columns)

FP = (len(X_test), int(y_test.sum()), int((y_test == 0).sum()))
print(f"NEW frozen test set (2015-2024): n={FP[0]}, No={FP[2]}, Yes={FP[1]}", flush=True)

def dist(v):
    c = v.value_counts().to_dict(); n = len(v)
    return {"No": int(c.get(0, 0)), "Yes": int(c.get(1, 0)),
            "pct_No": round(c.get(0, 0)/n*100, 3), "pct_Yes": round(c.get(1, 0)/n*100, 3)}

X_train.to_parquet(f"{OUT}/02_preprocessing_audit/X_train.parquet")
X_test.to_parquet(f"{OUT}/02_preprocessing_audit/X_test.parquet")
y_train.to_frame("y").to_parquet(f"{OUT}/02_preprocessing_audit/y_train.parquet")
y_test.to_frame("y").to_parquet(f"{OUT}/02_preprocessing_audit/y_test.parquet")
config = {
    "audit": "corrected 2015-2024 rerun (YEAR==2014 excluded)",
    "master_rows": n_master, "dropped_2014_rows": dropped,
    "working_rows": len(dfc), "random_state": RNG,
    "split": "stratified 80/20 train/test, shuffle=True",
    "target": TARGET, "features": FEATURES, "n_features": 15,
    "exclude_YEAR": "YEAR kept as metadata only",
    "feature_encoding": "integer-coded categoricals kept as int (no one-hot); "
                        "standard SMOTE/ADASYN on integer codes (same as original)",
    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "train_dist": dist(y_train), "test_dist": dist(y_test),
    "models": {
        "LogisticRegression": {"max_iter": 1000, "random_state": 42, "solver": "lbfgs",
                               "scaling": "StandardScaler fitted on TRAIN only"},
        "RandomForest": {"n_estimators": 100, "max_depth": 12, "n_jobs": -1, "random_state": 42},
        "XGBoost": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
                    "eval_metric": "logloss", "n_jobs": -1, "random_state": 42}},
    "resampling": {"random_state": 42, "sampling_strategy": "auto (minority -> majority)",
                   "scope": "TRAIN ONLY; test set never touched"},
}
with open(f"{OUT}/02_preprocessing_audit/protocol_config_audit.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"split: train={len(X_train):,} test={len(X_test):,}", flush=True)

def make_models():
    return {
        "LogisticRegression": ("scaled", LogisticRegression(max_iter=1000, random_state=RNG)),
        "RandomForest": ("raw", RandomForestClassifier(n_estimators=100, max_depth=12,
                                                       n_jobs=-1, random_state=RNG)),
        "XGBoost": ("raw", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8,
                                        tree_method="hist", eval_metric="logloss",
                                        n_jobs=-1, random_state=RNG)),
    }

def evaluate(name, model, kind, Xr, yr, tech_key, proba_store):
    t = time.time()
    if kind == "scaled":
        scaler = StandardScaler().fit(Xr)
        model.fit(scaler.transform(Xr), yr)
        y_pred = model.predict(scaler.transform(X_test))
        y_proba = model.predict_proba(scaler.transform(X_test))[:, 1]
    else:
        model.fit(Xr, yr)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    proba_store[name] = {"y_pred": y_pred.astype(np.int8), "y_proba": y_proba.astype(np.float32)}
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    assert cm.sum() == len(y_test), f"CM total {cm.sum()} != test size {len(y_test)}"
    tn, fp, fn, tp = cm.ravel()
    row = {"Technique": tech_key, "Model": name,
           "Accuracy": accuracy_score(y_test, y_pred),
           "Precision": precision_score(y_test, y_pred, zero_division=0),
           "Recall": recall_score(y_test, y_pred, zero_division=0),
           "F1": f1_score(y_test, y_pred, zero_division=0),
           "ROC_AUC": roc_auc_score(y_test, y_proba),
           "PR_AUC": average_precision_score(y_test, y_proba),
           "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
           "fit_predict_sec": round(time.time() - t, 1)}
    for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]:
        assert np.isfinite(row[k]) and 0.0 <= row[k] <= 1.0, f"invalid {k} for {name}"
    p, r = row["Precision"], row["Recall"]
    if p + r > 0:
        assert abs(row["F1"] - 2*p*r/(p+r)) < 1e-9, "F1 inconsistent with P/R"
    print(f"  {name}: acc={row['Accuracy']:.4f} prec={row['Precision']:.4f} "
          f"rec={row['Recall']:.4f} f1={row['F1']:.4f} roc={row['ROC_AUC']:.4f} "
          f"prauc={row['PR_AUC']:.4f} [{row['fit_predict_sec']}s]", flush=True)
    return row

def check_scope(Xr, yr, tech):
    assert list(Xr.columns) == list(X_train.columns), f"{tech}: feature mismatch"
    assert Xr.shape[1] == 15, f"{tech}: feature count changed"
    assert yr.isna().sum() == 0 and np.isfinite(np.asarray(Xr)).all(), f"{tech}: NaN/inf"
    assert (len(X_test), int(y_test.sum()), int((y_test == 0).sum())) == FP, \
        f"{tech}: TEST SET CHANGED!"

def run_experiment(tech_key, subdir, sampler=None):
    print(f"\n=== {tech_key} (2015-2024) ===", flush=True)
    if sampler is None:
        Xr, yr = X_train.copy(), y_train.copy(); added = 0
    else:
        t = time.time()
        Xr, yr = sampler.fit_resample(X_train, y_train)
        added = len(Xr) - len(X_train)
        print(f"  resampling took {time.time()-t:.1f}s; added {added:,} samples; "
              f"train {len(X_train):,} -> {len(Xr):,}", flush=True)
    before = y_train.value_counts().to_dict()
    after = pd.Series(yr).value_counts().to_dict()
    check_scope(Xr, yr, tech_key)
    rows, proba_store = [], {}
    for name, (kind, model) in make_models().items():
        rows.append(evaluate(name, model, kind, Xr, yr, tech_key, proba_store))
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/{subdir}/results_{tech_key.lower().replace(' ', '_')}_2015_2024.csv", index=False)
    prob_df = pd.DataFrame({"y_true": y_test.values})
    for mname, d in proba_store.items():
        prob_df[f"{mname}__pred"] = d["y_pred"]
        prob_df[f"{mname}__proba"] = d["y_proba"]
    prob_df.to_parquet(f"{OUT}/{subdir}/test_predictions_2015_2024.parquet")
    assert (prob_df["y_true"].values == y_test.values).all(), "y_true drift!"
    dist_out = {"technique": tech_key, "scope": "2015-2024",
                "train_before": {int(k): int(v) for k, v in before.items()},
                "train_after": {int(k): int(v) for k, v in after.items()},
                "added_samples": int(added), "final_train_size": int(len(Xr))}
    with open(f"{OUT}/{subdir}/class_distribution_2015_2024.json", "w") as f:
        json.dump(dist_out, f, indent=2)
    print(f"  saved {subdir}/ (CM totals validated, test fingerprint unchanged)", flush=True)
    return res, dist_out

experiments = [
    ("No Balancing", "03_baseline_audit", None),
    ("Random Oversampling", "04_ros_audit",
     RandomOverSampler(sampling_strategy="auto", random_state=RNG)),
    ("SMOTE", "05_smote_audit",
     SMOTE(sampling_strategy="auto", random_state=RNG, k_neighbors=5)),
    ("ADASYN", "06_adasyn_audit",
     ADASYN(sampling_strategy="auto", random_state=RNG, n_neighbors=5)),
]
all_results, all_dist = [], {}
for tech_key, subdir, sampler in experiments:
    res, dist = run_experiment(tech_key, subdir, sampler)
    all_results.append(res); all_dist[tech_key] = dist

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(f"{OUT}/07_combined_results_audit/combined_results_2015_2024.csv", index=False)
with open(f"{OUT}/07_combined_results_audit/class_distributions_2015_2024.json", "w") as f:
    json.dump(all_dist, f, indent=2)
print("\nCORRECTED 2015-2024 RERUN COMPLETE. Consolidated table:", flush=True)
print(combined[["Technique", "Model", "Accuracy", "Precision", "Recall", "F1",
                "ROC_AUC", "PR_AUC"]].to_string(index=False), flush=True)
print(f"FINAL test fingerprint: n={len(X_test)}, No={int((y_test==0).sum())}, "
      f"Yes={int(y_test.sum())} (identical across all 4 experiments)", flush=True)
