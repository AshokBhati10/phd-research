"""Shared constants and helpers for the Person 2 BRFSS pipeline (advanced oversampling).

Aligned to Person 1's final protocol (corrected 2015-2024 scope) and to Person 3's
frozen implementation. See PROTOCOL_P2.md. This module implements the protocol;
it does not decide it.
"""
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RNG = 42
TARGET = "DIABETE4"
FEATURES = [
    "EXERANY2", "_MICHD", "ASTHMA3", "ADDEPEV3", "CHCKDNY2", "MARITAL",
    "_EDUCAG", "DRNKANY6", "_AGEG5YR", "_BMI5CAT", "_INCOMG1", "_RFSMOK3",
    "GENHLTH", "HAVARTH4", "SEXVAR",
]
TECHNIQUES = ["Borderline-SMOTE", "KMeans-SMOTE", "SVM-SMOTE", "SMOTE-N"]
TECHNIQUE_SLUGS = {
    "Borderline-SMOTE": "borderline_smote",
    "KMeans-SMOTE": "kmeans_smote",
    "SVM-SMOTE": "svm_smote",
    "SMOTE-N": "smoten",
}
BASELINE = "No Balancing"
BASELINE_SLUG = "no_balancing"
MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]

RESULTS_HEADER = (
    "Technique,Model,Accuracy,Precision,Recall,F1,ROC_AUC,PR_AUC,"
    "TN,FP,FN,TP,fit_predict_sec"
)

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parent
P1_ROOT = REPO / "P1_Basic_Oversampling"
P1_SPLITS = P1_ROOT / "11_Audit_2015_2024_Rerun" / "02_preprocessing_audit"
P1_PREDICTIONS = {
    "No Balancing": P1_ROOT / "11_Audit_2015_2024_Rerun" / "03_baseline_audit" / "test_predictions_2015_2024.parquet",
    "Random Oversampling": P1_ROOT / "11_Audit_2015_2024_Rerun" / "04_ros_audit" / "test_predictions_2015_2024.parquet",
    "SMOTE": P1_ROOT / "11_Audit_2015_2024_Rerun" / "05_smote_audit" / "test_predictions_2015_2024.parquet",
    "ADASYN": P1_ROOT / "11_Audit_2015_2024_Rerun" / "06_adasyn_audit" / "test_predictions_2015_2024.parquet",
}

# Frozen corrected-scope reference (Person 1 final / Person 3 protocol)
EXPECTED_TEST_FINGERPRINT = "70e180c659c74efbb6ac6368f5b061c9a94cf7efc10a707f52abcf29280b0d11"
EXPECTED_TRAIN_SIZE = 1_837_734
EXPECTED_TEST_SIZE = 459_434
EXPECTED_TRAIN_COUNTS = {0: 1_524_272, 1: 313_462}
EXPECTED_TEST_COUNTS = {0: 381_068, 1: 78_366}


# ---------------------------------------------------------------------------
# Resamplers (frozen per PROTOCOL_P2.md section 3)
# ---------------------------------------------------------------------------
def make_resampler(technique):
    """Return the imbalanced-learn resampler for a Person 2 technique.

    Parameter provenance (documented, not invented):
      - sampling_strategy="auto": as in Person 1 / Person 3.
      - random_state=42: where the estimator supports it.
      - Borderline-SMOTE: library defaults (kind="borderline-1", k_neighbors=5,
        m_neighbors=10).
      - KMeans-SMOTE: kmeans_estimator=100 and cluster_balance_threshold=0.2 are
        REQUIRED on this all-categorical feature space; library defaults
        (8 clusters, threshold 0.5) raise RuntimeError (no qualifying clusters).
        Documented deviation, approved for the P2 protocol.
      - SVM-SMOTE / SMOTE-N: library defaults; executed on the shared stratified
        50k train subsample (O(n^2) cost/memory -- disclosed deviation).
    """
    from imblearn.over_sampling import BorderlineSMOTE, KMeansSMOTE, SMOTEN, SVMSMOTE

    if technique == "Borderline-SMOTE":
        return BorderlineSMOTE(sampling_strategy="auto", random_state=RNG)
    if technique == "KMeans-SMOTE":
        return KMeansSMOTE(
            sampling_strategy="auto",
            kmeans_estimator=100,
            cluster_balance_threshold=0.2,
            random_state=RNG,
        )
    if technique == "SVM-SMOTE":
        return SVMSMOTE(sampling_strategy="auto", random_state=RNG)
    if technique == "SMOTE-N":
        return SMOTEN(sampling_strategy="auto", random_state=RNG)
    raise ValueError(f"Unknown technique: {technique}")


def describe_resampler(technique):
    """Human-readable parameter list for audit / Protocol_config."""
    params = make_resampler(technique).get_params()
    source = {"sampling_strategy", "random_state", "kmeans_estimator",
              "cluster_balance_threshold"}
    return "; ".join(
        f"{k}={params[k]!r} [{'source-specified' if k in source else 'library-default'}]"
        for k in sorted(params)
    )


# ---------------------------------------------------------------------------
# Models (frozen per PROTOCOL_P2.md section 4 -- Person 1's exact configuration)
# ---------------------------------------------------------------------------
def make_models():
    """Return {model_name: (scaling, estimator)}.

    "scaled": StandardScaler fitted on the RESAMPLED training data (Person 1's
    actual code: StandardScaler().fit(Xr) after resampling), used for
    LogisticRegression only. "raw": integer codes used directly.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    return {
        "LogisticRegression": (
            "scaled",
            LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RNG),
        ),
        "RandomForest": (
            "raw",
            RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1,
                                   random_state=RNG),
        ),
        "XGBoost": (
            "raw",
            XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", eval_metric="logloss",
                n_jobs=-1, random_state=RNG,
            ),
        ),
    }


# ---------------------------------------------------------------------------
# Metrics (Person 1 / Person 3 definitions)
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba):
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "PR_AUC": average_precision_score(y_true, y_proba),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def format_result_row(technique, model, metrics, fit_predict_sec):
    return (
        f"{technique},{model},"
        f"{metrics['Accuracy']:.10f},{metrics['Precision']:.10f},"
        f"{metrics['Recall']:.10f},{metrics['F1']:.10f},"
        f"{metrics['ROC_AUC']:.10f},{metrics['PR_AUC']:.10f},"
        f"{metrics['TN']},{metrics['FP']},{metrics['FN']},{metrics['TP']},"
        f"{fit_predict_sec:.1f}"
    )


# ---------------------------------------------------------------------------
# Fingerprint / leakage guard (identical algorithm to Person 3's common.py)
# ---------------------------------------------------------------------------
def fingerprint(X, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(X)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(y).ravel()).tobytes())
    return h.hexdigest()


def class_counts(y):
    y = np.asarray(y).ravel()
    return {"0": int((y == 0).sum()), "1": int((y == 1).sum())}


# ---------------------------------------------------------------------------
# JSON / checkpoint helpers
# ---------------------------------------------------------------------------
def load_json(path, default=None):
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def completed_pairs(results_csv):
    pairs = set()
    if os.path.exists(results_csv):
        with open(results_csv, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    pairs.add((parts[0], parts[1]))
    return pairs


def append_result_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        if new:
            f.write(RESULTS_HEADER + "\n")
        f.write(row + "\n")
        f.flush()
        os.fsync(f.fileno())


def drop_technique_rows(path, technique):
    """Remove all CSV rows for a technique (used before an incomplete rerun)."""
    path = Path(path)
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [lines[0]] + [ln for ln in lines[1:] if not ln.startswith(technique + ",")]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Synthetic-row audit (Person 1 feature-validity checks + class bookkeeping)
# ---------------------------------------------------------------------------
def audit_resample(technique, X_orig, y_orig, X_res, y_res, feature_names):
    """Audit a resampling result.

    Returns class bookkeeping plus feature-validity checks on appended synthetic
    rows: finiteness, observed range, unobserved category codes, fractional values.
    """
    Xo = np.asarray(X_orig)
    Xr = np.asarray(X_res)
    yo = np.asarray(y_orig).ravel()
    yr = np.asarray(y_res).ravel()

    before = class_counts(yo)
    after = class_counts(yr)
    n_before, n_after = len(Xo), len(Xr)
    n_synthetic = n_after - n_before if n_after >= n_before else 0

    audit = {
        "technique": technique,
        "dtype": str(Xr.dtype),
        "train_before": before,
        "train_after": after,
        "added_samples": int(n_after - n_before),
        "rows_removed": int(max(0, n_before - n_after)),
        "final_train_size": int(n_after),
        "all_finite": bool(np.isfinite(Xr).all()),
        "no_nan": bool(not np.isnan(Xr.astype(np.float64)).any()),
        "synthetic_checks": None,
    }

    if n_synthetic <= 0:
        return audit

    # imblearn over-samplers append synthetic rows after the original matrix.
    base_matches = bool(np.array_equal(Xr[:n_before], Xo))
    audit["original_rows_preserved"] = base_matches
    if not base_matches:
        audit["synthetic_checks"] = {"skipped": "original rows not a prefix; append semantics not verified"}
        return audit

    synth = Xr[n_before:]
    per_feature = {}
    total_unobserved = 0
    total_out_of_range = 0
    total_fractional = 0
    for j, name in enumerate(feature_names):
        col = synth[:, j]
        obs = np.unique(Xo[:, j])
        lo, hi = int(obs.min()), int(obs.max())
        unobserved = int(np.count_nonzero(~np.isin(col, obs)))
        out_of_range = int(np.count_nonzero((col < lo) | (col > hi)))
        fractional = int(np.count_nonzero(col != np.floor(col)))
        total_unobserved += unobserved
        total_out_of_range += out_of_range
        total_fractional += fractional
        per_feature[name] = {
            "observed_min": lo, "observed_max": hi,
            "synthetic_min": int(col.min()), "synthetic_max": int(col.max()),
            "out_of_range": out_of_range,
            "unobserved_codes": unobserved,
            "fractional_values": fractional,
        }

    audit["synthetic_checks"] = {
        "n_synthetic_rows": int(n_synthetic),
        "finite": bool(np.isfinite(synth).all()),
        "out_of_range_values": total_out_of_range,
        "unobserved_code_values": total_unobserved,
        "fractional_values": total_fractional,
        "per_feature": per_feature,
    }
    return audit


# ---------------------------------------------------------------------------
# Environment record
# ---------------------------------------------------------------------------
def environment_record():
    import platform

    rec = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        rec["ram_total_gb"] = round(vm.total / 1e9, 2)
        rec["ram_available_gb"] = round(vm.available / 1e9, 2)
    except ImportError:
        pass
    for mod in ["numpy", "pandas", "sklearn", "imblearn", "xgboost", "pyarrow", "openpyxl", "scipy"]:
        try:
            m = __import__(mod)
            rec[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            rec[mod] = "NOT INSTALLED"
    return rec


def log(msg, log_path=None):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def rss_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e9, 2)
    except ImportError:
        return None


def release(*objs):
    for obj in objs:
        del obj
    gc.collect()
