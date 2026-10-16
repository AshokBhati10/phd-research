"""Regenerate the 12 publication-quality plots from the CORRECTED 2015-2024
results and corrected test predictions. Outputs to 08_Visualizations_FINAL_2015_2024/.
Honest axes (bars from 0), no distortion."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

BASE = "/home/hatch/workspace/Person_1_Class_Imbalance"
A = f"{BASE}/11_Audit_2015_2024_Rerun"
OUT = f"{BASE}/08_Visualizations_FINAL_2015_2024"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 150})

TECHS = ["No Balancing", "Random Oversampling", "SMOTE", "ADASYN"]
MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]
MODEL_LABELS = {"LogisticRegression": "LR", "RandomForest": "RF", "XGBoost": "XGB"}
SUB = {"No Balancing": "03_baseline_audit", "Random Oversampling": "04_ros_audit",
       "SMOTE": "05_smote_audit", "ADASYN": "06_adasyn_audit"}
COLORS = {"No Balancing": "#7f7f7f", "Random Oversampling": "#1f77b4",
          "SMOTE": "#2ca02c", "ADASYN": "#d62728"}
RESNAME = {"No Balancing": "results_no_balancing_2015_2024.csv",
           "Random Oversampling": "results_random_oversampling_2015_2024.csv",
           "SMOTE": "results_smote_2015_2024.csv",
           "ADASYN": "results_adasyn_2015_2024.csv"}

df = pd.read_csv(f"{A}/07_combined_results_audit/combined_results_2015_2024.csv")
assert len(df) == 12, f"expected 12 rows, got {len(df)}"
assert (df["TN"] + df["FP"] + df["FN"] + df["TP"] == 459434).all(), "CM total mismatch"
probs = {t: pd.read_parquet(f"{A}/{SUB[t]}/test_predictions_2015_2024.parquet") for t in TECHS}
for t in TECHS:
    assert (probs[t]["y_true"].values == probs[TECHS[0]]["y_true"].values).all(), \
        f"test-set drift in {t}"
dists = {t: json.load(open(f"{A}/{SUB[t]}/class_distribution_2015_2024.json")) for t in TECHS[1:]}

# --- 1-4: class distributions ---
def bar_two(ax, labels, vals, title, note):
    ax.bar(labels, vals, color=["#1f77b4", "#d62728"], edgecolor="black")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}\n({v/sum(vals)*100:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title(title)
    ax.set_ylabel("Observations")
    ax.text(0.98, 0.96, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, style="italic")

fig, ax = plt.subplots(figsize=(7, 5))
bar_two(ax, ["No diabetes", "Diabetes"],
        [1905340, 391828],
        "Class distribution — BRFSS 2015–2024 analytical dataset (n = 2,297,168)",
        "Imbalance ratio 4.86:1")
fig.tight_layout(); fig.savefig(f"{OUT}/01_class_dist_analytical.png"); plt.close(fig)

for i, (t, fn) in enumerate([("Random Oversampling", "02"), ("SMOTE", "03"), ("ADASYN", "04")], start=2):
    d = dists[t]["train_after"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_two(ax, ["No diabetes", "Diabetes"], [d["0"], d["1"]],
            f"Training class distribution after {t} (n = {d['0']+d['1']:,})",
            f"+{dists[t]['added_samples']:,} synthetic/replicated rows")
    fig.tight_layout(); fig.savefig(f"{OUT}/{fn}_class_dist_{t.lower().replace(' ', '_')}.png"); plt.close(fig)

# --- 5: confusion matrices (4 techniques x 3 models) ---
fig, axes = plt.subplots(4, 3, figsize=(11, 14))
for r, t in enumerate(TECHS):
    for c, m in enumerate(MODELS):
        row = df[(df.Technique == t) & (df.Model == m)].iloc[0]
        cm = np.array([[row.TN, row.FP], [row.FN, row.TP]])
        ax = axes[r, c]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred No", "Pred Yes"]); ax.set_yticklabels(["True No", "True Yes"])
        ax.set_title(f"{t} — {MODEL_LABELS[m]}")
        for (ii, jj), v in np.ndenumerate(cm):
            ax.text(jj, ii, f"{int(v):,}", ha="center", va="center", fontsize=11,
                    color="white" if v > cm.max() / 2 else "black")
        ax.text(1.02, -0.12, f"acc={row.Accuracy:.3f} F1={row.F1:.3f}", transform=ax.transAxes,
                fontsize=9, ha="left")
fig.suptitle("Confusion matrices — corrected 2015–2024 test set (n = 459,434 per cell)", y=0.995)
fig.tight_layout(); fig.savefig(f"{OUT}/05_confusion_matrices.png"); plt.close(fig)

# --- 6/7: ROC and PR curves (2x2 grid, 3 models each) ---
for kind, fname, title in [("roc", "06_roc_curves.png", "ROC curves"),
                           ("pr", "07_pr_curves.png", "Precision–Recall curves")]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, t in zip(axes.flat, TECHS):
        p = probs[t]
        for m in MODELS:
            y_true, y_proba = p["y_true"].values, p[f"{m}__proba"].values
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                ax.plot(fpr, tpr, label=f"{MODEL_LABELS[m]} (AUC={auc(fpr, tpr):.3f})")
            else:
                prec, rec, _ = precision_recall_curve(y_true, y_proba)
                ax.plot(rec, prec,
                        label=f"{MODEL_LABELS[m]} (AP={average_precision_score(y_true, y_proba):.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        else:
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(t); ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle(f"{title} — corrected 2015–2024 test set (n = 459,434)", y=0.995)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}"); plt.close(fig)

# --- 8-12: metric comparison bar charts ---
def grouped(metric, fname, title, ylim=None):
    x = np.arange(len(TECHS)); w = 0.22
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(MODELS):
        vals = [df[(df.Technique == t) & (df.Model == m)].iloc[0][metric] for t in TECHS]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=MODEL_LABELS[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([t.replace(" ", "\n") for t in TECHS])
    ax.set_ylabel(title); ax.set_title(f"{title} by technique and model — 2015–2024 test set")
    ax.set_ylim(ylim or (0, max(df[metric]) * 1.18)); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}"); plt.close(fig)

grouped("Accuracy", "08_accuracy_comparison.png", "Accuracy")
grouped("Recall", "09_recall_comparison.png", "Recall (sensitivity)")
grouped("F1", "10_f1_comparison.png", "F1-score")
grouped("ROC_AUC", "11_roc_auc_comparison.png", "ROC-AUC", ylim=(0.70, 0.80))
grouped("PR_AUC", "12_pr_auc_comparison.png", "PR-AUC", ylim=(0, 0.50))
