"""
STEP 2 - Preprocessing + stratified train/test split (seed 42).
Saves: X_train, X_test, y_train, y_test + protocol_config.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OUT = "/home/hatch/workspace/Person_1_Class_Imbalance/02_Preprocessing"
SRC = "/home/hatch/workspace/brfss_data/brfss_master.parquet"
RNG = 42

TARGET = "DIABETE4"
FEATURES = ["EXERANY2", "_MICHD", "ASTHMA3", "ADDEPEV3", "CHCKDNY2",
            "MARITAL", "_EDUCAG", "DRNKANY6", "_AGEG5YR", "_BMI5CAT",
            "_INCOMG1", "_RFSMOK3", "GENHLTH", "HAVARTH4", "SEXVAR"]

df = pd.read_parquet(SRC)
n0 = len(df)

# --- Validation: master is already standardized per documented protocol ---
# 1. _AGEG5YR 7/9 are VALID age-category codes; the 7/9 filter in the raw-XPT
#    preprocessing.py must NOT be applied here (would drop ~20% valid rows).
# 2. Age restriction >=5 : already satisfied (min==5.0).
assert df["_AGEG5YR"].min() >= 5.0, "age restriction violated"
# 3. DIABETE4 already standardized to No/Yes.
assert set(df["DIABETE4"].unique()) == {"No", "Yes"}
# 4. No missing values.
assert df.isna().sum().sum() == 0, "unexpected missing values"

df = df.loc[df["_AGEG5YR"] >= 5].copy()          # age restriction (keeps all rows)
df = df.dropna(subset=FEATURES + [TARGET]).copy()  # complete-case (no-op)

X = df[FEATURES].copy()
y = (df[TARGET] == "Yes").astype(int)            # 1 = diabetes Yes
# keep integer-coded categoricals as int for SMOTENC categorical indices
X = X.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RNG, stratify=y, shuffle=True)

# --- Validation of the split ---
assert len(X_train) + len(X_test) == len(X)
assert len(np.intersect1d(X_train.index, X_test.index)) == 0, "index overlap!"
assert set(X_train.columns) == set(FEATURES)
assert TARGET not in X_train.columns, "target leaked into predictors!"
# same feature representation for all experiments (frozen here)
assert list(X_train.columns) == list(X_test.columns)

def dist(v):
    c = v.value_counts().to_dict()
    n = len(v)
    return {"No": int(c.get(0, 0)), "Yes": int(c.get(1, 0)),
            "pct_No": round(c.get(0, 0)/n*100, 3), "pct_Yes": round(c.get(1, 0)/n*100, 3)}

config = {
    "random_state": RNG,
    "split": "stratified 80/20 train/test, shuffle=True",
    "target": TARGET, "positive_class": "Yes -> 1", "negative_class": "No -> 0",
    "features": FEATURES, "n_features": len(FEATURES),
    "exclude_YEAR": "YEAR kept as metadata only (not a patient-level predictor; "
                    "consistent with documented RESEARCH_VARIABLES list)",
    "feature_encoding": "integer-coded categoricals kept as int; no one-hot "
                        "(SMOTENC categorical-aware; tree models native)",
    "n_total": n0, "n_after_preprocessing": len(df),
    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "train_dist": dist(y_train), "test_dist": dist(y_test),
    "models": {
        "LogisticRegression": {"max_iter": 1000, "random_state": 42, "solver": "lbfgs",
                               "scaling": "StandardScaler fitted on TRAIN only"},
        "RandomForest": {"n_estimators": 100, "max_depth": 12, "n_jobs": -1, "random_state": 42},
        "XGBoost": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
                    "eval_metric": "logloss", "n_jobs": -1, "random_state": 42},
    },
    "resampling": {"random_state": 42, "sampling_strategy": "auto (minority -> majority)",
                   "scope": "TRAIN ONLY; test set never touched"},
}
X_train.to_parquet(f"{OUT}/X_train.parquet"); X_test.to_parquet(f"{OUT}/X_test.parquet")
y_train.to_frame("y").to_parquet(f"{OUT}/y_train.parquet"); y_test.to_frame("y").to_parquet(f"{OUT}/y_test.parquet")
with open(f"{OUT}/protocol_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(json.dumps({k: config[k] for k in
      ["n_total", "n_after_preprocessing", "n_train", "n_test", "train_dist", "test_dist"]}, indent=2))
print("train/test split OK: no overlap, no leakage, frozen feature set")
