"""Shared constants and loaders for the P4 integration.

P4 is strictly read-only with respect to P1/P2/P3: every number in the combined
tables is traced to a validated source artifact registered here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[1]

MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]

P1 = REPO / "P1_Basic_Oversampling"
P2 = REPO / "P2_Advanced_Oversampling"
P3 = REPO / "P3_Advanced_Resampling"
P1_AUD = P1 / "11_Audit_2015_2024_Rerun"
P1_COMBINED_CSV = P1_AUD / "07_combined_results_audit" / "combined_results_2015_2024.csv"
P2_RESULTS_CSV = P2 / "results" / "results_person2.csv"
P2_BASELINE_CSV = P2 / "results" / "results_baseline_no_balancing.csv"
P3_RESULTS_CSV = P3 / "results" / "results_person3.csv"
P2_DIST_JSON = P2 / "results" / "class_distributions.json"
P3_DIST_JSON = P3 / "results" / "class_distributions.json"

EXPECTED_TEST_FINGERPRINT = "70e180c659c74efbb6ac6368f5b061c9a94cf7efc10a707f52abcf29280b0d11"
P1_SPLITS = P1_AUD / "02_preprocessing_audit"

DAGGER = "\u2020"  # †


# ---------------------------------------------------------------------------
# Source registry: one entry per (person, technique)
# ---------------------------------------------------------------------------
def _p1(tech, audit_dir, display=None, caveat=""):
    d = P1_AUD / audit_dir
    return {
        "person": "P1", "family": "A. Basic oversampling" if tech != "No Balancing" else "0. Baseline",
        "technique": tech, "display": display or tech, "caveat": caveat,
        "results_csv": str(P1_COMBINED_CSV), "csv_label": tech,
        "predictions": str(d / "test_predictions_2015_2024.parquet"),
        "dist_json": str(d / "class_distribution_2015_2024.json"), "dist_key": None,
        "train_source": "full corrected train split (1,837,734 rows)",
    }


def _p2(tech, slug, display=None, caveat="", baseline=False):
    return {
        "person": "P2", "family": "B. Advanced oversampling" if not baseline else "0. Baseline (replication)",
        "technique": tech, "display": display or tech, "caveat": caveat,
        "results_csv": str(P2_BASELINE_CSV if baseline else P2_RESULTS_CSV), "csv_label": tech,
        "predictions": str(P2 / "results" / f"test_predictions_{slug}.parquet"),
        "dist_json": str(P2_DIST_JSON), "dist_key": tech,
        "train_source": ("raw corrected train split (no resampling)" if baseline
                         else "full corrected train split (1,837,734 rows)" if tech in
                         ("Borderline-SMOTE", "KMeans-SMOTE") else
                         "shared stratified 50k subsample (41,472/8,528, seed 42)"),
    }


def _p3(tech, slug, display=None, caveat=""):
    return {
        "person": "P3", "family": "C. Cleaning & hybrid", "technique": tech,
        "display": display or tech, "caveat": caveat,
        "results_csv": str(P3_RESULTS_CSV), "csv_label": tech,
        "predictions": str(P3 / "results" / f"test_predictions_{slug}.parquet"),
        "dist_json": str(P3_DIST_JSON), "dist_key": tech,
        "train_source": "full corrected train split (1,837,734 rows); cleaning removes majority/overlap rows",
    }


SOURCES = [
    _p1("No Balancing", "03_baseline_audit"),
    _p1("Random Oversampling", "04_ros_audit"),
    _p1("SMOTE", "05_smote_audit"),
    _p1("ADASYN", "06_adasyn_audit"),
    _p2("Borderline-SMOTE", "borderline_smote"),
    _p2("KMeans-SMOTE", "kmeans_smote",
        caveat="non-default params required (kmeans_estimator=100, cluster_balance_threshold=0.2)"),
    _p2("SVM-SMOTE", "svm_smote", display=f"SVM-SMOTE {DAGGER}",
        caveat="trained on 50k subsample (O(n^2) SVM fit); out_step extrapolation"),
    _p2("SMOTE-N", "smoten", display=f"SMOTE-N {DAGGER}",
        caveat="trained on 50k subsample (O(n^2) memory); categorical-native generation"),
    _p2("No Balancing", "no_balancing", display="No Balancing (P2 replication)", baseline=True,
        caveat="replication of P1's baseline; LR/RF bit-identical, XGB ~2e-3 cross-machine"),
    _p3("Tomek Links", "tomek"),
    _p3("ENN", "enn"),
    _p3("SMOTE-Tomek", "smote_tomek"),
    _p3("SMOTE-ENN", "smote_enn", caveat="staged two-stage implementation (documented equivalent)"),
]

CANONICAL_ORDER = [
    "No Balancing",
    "Random Oversampling", "SMOTE", "ADASYN",
    "Borderline-SMOTE", "KMeans-SMOTE", "SVM-SMOTE", "SMOTE-N",
    "Tomek Links", "ENN", "SMOTE-Tomek", "SMOTE-ENN",
]

FAMILY_OF = {s["technique"]: s["family"].split(". ")[1] for s in SOURCES
             if s["technique"] != "No Balancing"}
FAMILY_OF["No Balancing"] = "Baseline"


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


def load_predictions(src):
    return pd.read_parquet(src["predictions"])


def load_source_metrics(src):
    """Return the 3 rows for a technique from its registered results CSV."""
    df = pd.read_csv(src["results_csv"])
    label = src["csv_label"]
    if "Technique" in df.columns:
        sub = df[df["Technique"] == label]
    else:  # pragma: no cover
        raise ValueError(f"no Technique column in {src['results_csv']}")
    if len(sub) != 3:
        raise ValueError(f"expected 3 rows for {label} in {src['results_csv']}, got {len(sub)}")
    return sub


def load_dist(src):
    data = json.loads(Path(src["dist_json"]).read_text(encoding="utf-8"))
    if src["dist_key"]:
        return data[src["dist_key"]]
    return data


def class_counts_from_dist(d):
    return (int(d["train_before"]["0"]), int(d["train_before"]["1"]),
            int(d["train_after"]["0"]), int(d["train_after"]["1"]),
            int(d.get("added_samples", 0)), int(d.get("removed_samples", d.get("rows_removed", 0))),
            int(d["final_train_size"]))


# ---------------------------------------------------------------------------
def mcnemar(a, b, y):
    b_cnt = int(np.count_nonzero((a == y) & (b != y)))
    c_cnt = int(np.count_nonzero((a != y) & (b == y)))
    stat = ((b_cnt - c_cnt) ** 2) / (b_cnt + c_cnt) if (b_cnt + c_cnt) else 0.0
    from scipy.stats import chi2
    return b_cnt, c_cnt, float(stat), float(chi2.sf(stat, 1))


def holm(pvals):
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values (same order)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def auc_bootstrap_ci(y, proba_a, proba_b, metric="roc", n_boot=500, seed=42):
    """Paired bootstrap CI for metric(A) - metric(B) on the same test rows."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    fn = roc_auc_score if metric == "roc" else average_precision_score
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs[i] = fn(yy, proba_a[idx]) - fn(yy, proba_b[idx])
    diffs = diffs[np.isfinite(diffs)]
    return {
        "metric": metric.upper() + "_AUC",
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_boot": int(len(diffs)),
        "seed": seed,
    }


def fmt(x, dp=4):
    return f"{x:.{dp}f}"
