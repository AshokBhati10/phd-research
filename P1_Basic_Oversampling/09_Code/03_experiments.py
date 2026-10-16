"""
STEPS 3-6 - Four class-imbalance experiments on the SAME frozen test set.
  1. No Balancing (baseline)
  2. Random Oversampling (ROS)
  3. SMOTE (standard; see mixed-data decision below)
  4. ADASYN
Resampling is applied to the TRAINING data only. Test set is never modified.
Models: LogisticRegression (StandardScaler on train), RandomForest, XGBoost.
Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix.

MIXED-DATA DECISION: all 15 BRFSS features are integer-coded categoricals.
SMOTENC was attempted first but imbalanced-learn 0.14.2 REFUSES all-categorical
feature sets ("SMOTE-NC is not designed to work only with categorical features.
It requires some numerical features."). Therefore standard SMOTE is used on the
integer codes - the convention in published BRFSS-SMOTE studies. Synthetic values
are validated to stay within observed per-feature [min, max] with no NaN/inf.
ADASYN has no categorical-aware variant; applied identically and validated."""
import json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

BASE = "/home/hatch/workspace/Person_1_Class_Imbalance"
PRE = f"{BASE}/02_Preprocessing"
RNG = 42

X_train = pd.read_parquet(f"{PRE}/X_train.parquet")
X_test = pd.read_parquet(f"{PRE}/X_test.parquet")
y_train = pd.read_parquet(f"{PRE}/y_train.parquet")["y"].astype(int)
y_test = pd.read_parquet(f"{PRE}/y_test.parquet")["y"].astype(int)

TEST_FINGERPRINT = (len(X_test), int(y_test.sum()), int((y_test == 0).sum()))
print(f"Frozen test set: n={TEST_FINGERPRINT[0]}, No={TEST_FINGERPRINT[2]}, Yes={TEST_FINGERPRINT[1]}",
      flush=True)

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

def evaluate(name, model, kind, Xtr, ytr, out_dir, proba_store, tech_key):
    t = time.time()
    if kind == "scaled":
        scaler = StandardScaler().fit(Xtr)          # fitted on TRAIN (resampled) only
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(X_test)
        model.fit(Xtr_s, ytr)
        y_pred = model.predict(Xte_s)
        y_proba = model.predict_proba(Xte_s)[:, 1]
    else:
        model.fit(Xtr, ytr)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    proba_store[name] = {"y_pred": y_pred.astype(np.int8), "y_proba": y_proba.astype(np.float32)}
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    # --- validation: confusion matrix totals == test size ---
    assert cm.sum() == len(y_test), f"CM total {cm.sum()} != test size {len(y_test)}"
    tn, fp, fn, tp = cm.ravel()
    row = {
        "Technique": tech_key,
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
        "PR_AUC": average_precision_score(y_test, y_proba),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "fit_predict_sec": round(time.time() - t, 1),
    }
    # --- validation: metric sanity ---
    for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]:
        assert np.isfinite(row[k]), f"non-finite {k} for {name}"
        assert 0.0 <= row[k] <= 1.0, f"{k} out of range for {name}"
    p, r = row["Precision"], row["Recall"]
    if p + r > 0:
        assert abs(row["F1"] - 2 * p * r / (p + r)) < 1e-9, "F1 inconsistent with P/R"
    print(f"  {name}: acc={row['Accuracy']:.4f} prec={row['Precision']:.4f} "
          f"rec={row['Recall']:.4f} f1={row['F1']:.4f} roc={row['ROC_AUC']:.4f} "
          f"prauc={row['PR_AUC']:.4f} [{row['fit_predict_sec']}s]", flush=True)
    return row

def check_train_only(Xr, yr, tech):
    """Resampling-scope validation."""
    assert list(Xr.columns) == list(X_train.columns), f"{tech}: feature mismatch"
    assert Xr.shape[1] == X_train.shape[1] == 15, f"{tech}: feature count changed"
    assert yr.isna().sum() == 0 and np.isfinite(Xr.values).all(), f"{tech}: NaN/inf in resampled data"
    assert len(X_test) == TEST_FINGERPRINT[0], f"{tech}: TEST SET CHANGED!"
    assert int(y_test.sum()) == TEST_FINGERPRINT[1], f"{tech}: TEST LABELS CHANGED!"

def run_experiment(tech_key, subdir, sampler=None, smote_variant=None):
    print(f"\n=== {tech_key} ===", flush=True)
    if sampler is None:
        Xr, yr = X_train.copy(), y_train.copy()
        added = 0
    else:
        t = time.time()
        Xr, yr = sampler.fit_resample(X_train, y_train)
        added = len(Xr) - len(X_train)
        print(f"  resampling took {time.time()-t:.1f}s; added {added:,} samples; "
              f"train {len(X_train):,} -> {len(Xr):,}", flush=True)
    # class distribution bookkeeping
    before = y_train.value_counts().to_dict()
    after = pd.Series(yr).value_counts().to_dict()
    check_train_only(Xr, yr, tech_key)
    # synthetic-value validity: within observed global min/max per feature
    for c in Xr.columns:
        assert Xr[c].min() >= X_train[c].min() - 1e-9 and Xr[c].max() <= X_train[c].max() + 1e-9, \
            f"{tech_key}: feature {c} out of observed range"
    rows = []
    proba_store = {}
    for name, (kind, model) in make_models().items():
        rows.append(evaluate(name, model, kind, Xr, yr, subdir, proba_store, tech_key))
    res = pd.DataFrame(rows)
    res.to_csv(f"{BASE}/{subdir}/results_{tech_key.lower().replace(' ', '_')}.csv", index=False)
    # persist test-set predictions/probabilities for ROC & PR curves (Step 8)
    prob_df = pd.DataFrame({"y_true": y_test.values})
    for mname, d in proba_store.items():
        prob_df[f"{mname}__pred"] = d["y_pred"]
        prob_df[f"{mname}__proba"] = d["y_proba"]
    prob_df.to_parquet(f"{BASE}/{subdir}/test_predictions.parquet")
    assert (prob_df["y_true"].values == y_test.values).all(), "y_true drift!"
    dist = {"technique": tech_key, "train_before": {int(k): int(v) for k, v in before.items()},
            "train_after": {int(k): int(v) for k, v in after.items()},
            "added_samples": int(added), "final_train_size": int(len(Xr))}
    with open(f"{BASE}/{subdir}/class_distribution.json", "w") as f:
        json.dump(dist, f, indent=2)
    print(f"  saved {subdir}/ (CM totals validated, metrics finite)", flush=True)
    return res, dist

experiments = [
    ("No Balancing", "03_Baseline", None),
    ("Random Oversampling", "04_Random_Oversampling",
     RandomOverSampler(sampling_strategy="auto", random_state=RNG)),
    ("SMOTE", "05_SMOTE",
     SMOTE(sampling_strategy="auto", random_state=RNG, k_neighbors=5)),
    ("ADASYN", "06_ADASYN",
     ADASYN(sampling_strategy="auto", random_state=RNG, n_neighbors=5)),
]

all_results, all_dist = [], {}
for tech_key, subdir, sampler in experiments:
    res, dist = run_experiment(tech_key, subdir, sampler)
    all_results.append(res); all_dist[tech_key] = dist

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(f"{BASE}/07_Combined_Results/combined_results.csv", index=False)
with open(f"{BASE}/07_Combined_Results/class_distributions.json", "w") as f:
    json.dump(all_dist, f, indent=2)
print("\nALL EXPERIMENTS COMPLETE. Combined table:")
print(combined[["Technique", "Model", "Accuracy", "Precision", "Recall", "F1",
                "ROC_AUC", "PR_AUC"]].to_string(index=False))
