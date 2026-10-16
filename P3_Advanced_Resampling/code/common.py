"""Shared constants and helpers for the Person 3 BRFSS pipeline.

Methodology is frozen in PROTOCOL.md. This module implements it; it does not decide it.
"""
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

RNG = 42
TARGET = "DIABETE4"
FEATURES = [
    "EXERANY2", "_MICHD", "ASTHMA3", "ADDEPEV3", "CHCKDNY2", "MARITAL",
    "_EDUCAG", "DRNKANY6", "_AGEG5YR", "_BMI5CAT", "_INCOMG1", "_RFSMOK3",
    "GENHLTH", "HAVARTH4", "SEXVAR",
]
TECHNIQUES = ["ENN", "Tomek Links", "SMOTE-Tomek", "SMOTE-ENN"]
TECHNIQUE_SLUGS = {"ENN": "enn", "Tomek Links": "tomek", "SMOTE-Tomek": "smote_tomek", "SMOTE-ENN": "smote_enn"}
MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]

RESULTS_HEADER = ("Technique,Model,Accuracy,Precision,Recall,F1,ROC_AUC,PR_AUC,"
                  "TN,FP,FN,TP,fit_predict_sec")


# ----------------------------------------------------------------------------
# Resamplers (frozen per PROTOCOL.md §3)
# ----------------------------------------------------------------------------
def make_resampler(technique):
    """Return the imbalanced-learn resampler for a Person 3 technique.

    Parameter provenance (documented, not invented):
      - sampling_strategy="auto": source-specified (user brief §3).
      - random_state=42: source-specified wherever the estimator supports it.
      - SMOTE sub-component: Person 1's exact SMOTE
        (SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)).
      - Everything else: imbalanced-learn library defaults.
    """
    from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.over_sampling import SMOTE

    # Person 1's exact SMOTE configuration (verified from 09_Code/03_experiments.py)
    smote_p1 = SMOTE(sampling_strategy="auto", random_state=RNG, k_neighbors=5)

    if technique == "ENN":
        # EditedNearestNeighbours exposes no random_state in imbalanced-learn 0.14.2
        # (deterministic KNN-based removal). Defaults: n_neighbors=3, kind_sel="all".
        return EditedNearestNeighbours(sampling_strategy="auto")
    if technique == "Tomek Links":
        # TomekLinks exposes no random_state (deterministic pair removal).
        return TomekLinks(sampling_strategy="auto")
    if technique == "SMOTE-Tomek":
        return SMOTETomek(
            sampling_strategy="auto",
            random_state=RNG,
            smote=smote_p1,
            tomek=TomekLinks(sampling_strategy="auto"),
        )
    if technique == "SMOTE-ENN":
        return SMOTEENN(
            sampling_strategy="auto",
            random_state=RNG,
            smote=smote_p1,
            enn=EditedNearestNeighbours(sampling_strategy="auto"),
        )
    raise ValueError(f"Unknown technique: {technique}")


def describe_resampler(technique):
    """Human-readable parameter provenance for audit / Protocol_config."""
    rs = make_resampler(technique)
    params = rs.get_params()
    # Annotate which params are source-specified vs library defaults.
    # "smote"/"tomek"/"enn" (and their nested params) are source-specified because
    # the SMOTE sub-component is Person 1's exact SMOTE configuration.
    source = {"sampling_strategy", "random_state", "smote", "tomek", "enn"}
    lines = []
    for k in sorted(params):
        v = params[k]
        tag = "source-specified" if (k in source or k.split("__")[0] in source) else "library-default"
        lines.append(f"{k}={v!r} [{tag}]")
    return "; ".join(lines)


# ----------------------------------------------------------------------------
# Models (frozen per PROTOCOL.md §4 — Person 1's exact configuration)
# ----------------------------------------------------------------------------
def make_models():
    """Return {model_name: (scaling, estimator)}.

    scaling == "scaled": StandardScaler fitted on the RESAMPLED training data
    (Person 1's actual code: StandardScaler().fit(Xr) after resampling), used for
    LogisticRegression only. "raw": integer codes used directly.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        "LogisticRegression": (
            "scaled",
            LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RNG),
        ),
        "RandomForest": (
            "raw",
            RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RNG),
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


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, average_precision_score,
                                 confusion_matrix)
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
    return (f"{technique},{model},"
            f"{metrics['Accuracy']:.10f},{metrics['Precision']:.10f},"
            f"{metrics['Recall']:.10f},{metrics['F1']:.10f},"
            f"{metrics['ROC_AUC']:.10f},{metrics['PR_AUC']:.10f},"
            f"{metrics['TN']},{metrics['FP']},{metrics['FN']},{metrics['TP']},"
            f"{fit_predict_sec:.1f}")


# ----------------------------------------------------------------------------
# Fingerprint / leakage guard
# ----------------------------------------------------------------------------
def fingerprint(X, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


def class_counts(y):
    y = np.asarray(y)
    return {"0": int((y == 0).sum()), "1": int((y == 1).sum())}


# ----------------------------------------------------------------------------
# Checkpoint helpers
# ----------------------------------------------------------------------------
def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed": {}, "test_fingerprint": None, "versions": {}}


def save_checkpoint(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def completed_pairs(results_csv):
    pairs = set()
    if os.path.exists(results_csv):
        with open(results_csv) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    pairs.add((parts[0], parts[1]))
    return pairs


# ----------------------------------------------------------------------------
# Environment record
# ----------------------------------------------------------------------------
def environment_record():
    import platform
    rec = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    for mod in ["numpy", "pandas", "sklearn", "imblearn", "xgboost", "pyarrow", "openpyxl"]:
        try:
            m = __import__(mod)
            rec[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            rec[mod] = "NOT INSTALLED"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    rec["mem_total_kb"] = int(line.split()[1])
                    break
    except OSError:
        pass
    rec["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    return rec


def log(msg, log_path=None):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    if log_path:
        with open(log_path, "a") as f:
            f.write(line + "\n")


def release(*names):
    for _ in names:
        pass
    gc.collect()
