"""
STEP 8 - Research-quality visualizations (12 requested plots).
Publication-friendly: labeled, honest axes (bars start at 0), consistent palette.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

BASE = "/home/hatch/workspace/Person_1_Class_Imbalance"
VIZ = f"{BASE}/08_Visualizations"
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 13, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9})

TECHNIQUES = ["No Balancing", "Random Oversampling", "SMOTE", "ADASYN"]
SUBDIRS = {"No Balancing": "03_Baseline", "Random Oversampling": "04_Random_Oversampling",
           "SMOTE": "05_SMOTE", "ADASYN": "06_ADASYN"}
MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]
MODEL_COLORS = {"LogisticRegression": "#1f77b4", "RandomForest": "#2ca02c", "XGBoost": "#d62728"}

combined = pd.read_csv(f"{BASE}/07_Combined_Results/combined_results.csv")
dists = json.load(open(f"{BASE}/07_Combined_Results/class_distributions.json"))

# ---------- 1-4: class distributions ----------
def class_dist_plot(key, fname, title, note=None):
    d = dists[key]
    ta = {int(k): int(v) for k, v in d["train_after"].items()}  # JSON keys are strings
    counts = [ta[0], ta[1]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["No (majority)", "Yes (minority)"], counts,
                  color=["#4e79a7", "#e15759"], edgecolor="black")
    ax.set_ylabel("Number of training samples")
    ax.set_title(title)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{c:,}",
                ha="center", va="bottom", fontsize=10)
    if note:
        ax.text(0.98, 0.96, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
    ax.set_ylim(0, max(counts) * 1.15)
    fig.tight_layout(); fig.savefig(f"{VIZ}/{fname}", bbox_inches="tight"); plt.close(fig)

before = dists["No Balancing"]["train_before"]
before = {int(k): int(v) for k, v in before.items()}
class_dist_plot("No Balancing", "01_class_dist_original.png",
                "Original training class distribution (no balancing)",
                note=f"Imbalance ratio {before[0]/before[1]:.2f}:1")
class_dist_plot("Random Oversampling", "02_class_dist_ros.png",
                "Training class distribution after Random Oversampling",
                note=f"+{dists['Random Oversampling']['added_samples']:,} replicated samples")
class_dist_plot("SMOTE", "03_class_dist_smote.png",
                "Training class distribution after SMOTE",
                note=f"+{dists['SMOTE']['added_samples']:,} synthetic samples")
ad = dists["ADASYN"]; ada = {int(k): int(v) for k, v in ad["train_after"].items()}
class_dist_plot("ADASYN", "04_class_dist_adasyn.png",
                "Training class distribution after ADASYN",
                note=f"+{ad['added_samples']:,} synthetic samples\n(minority {ada[1]:,} < majority {ada[0]:,}: ADASYN does not force exact parity)")

# ---------- 5: confusion matrices (4 x 3 grid) ----------
fig, axes = plt.subplots(4, 3, figsize=(13, 15), sharex=True, sharey=True)
for i, tech in enumerate(TECHNIQUES):
    sub = combined[combined["Technique"] == tech]
    for j, model in enumerate(MODELS):
        r = sub[sub["Model"] == model].iloc[0]
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        ax = axes[i, j]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred No", "Pred Yes"], yticklabels=["True No", "True Yes"],
                    cbar=False, annot_kws={"fontsize": 10})
        ax.set_title(f"{tech} / {model}", fontsize=11)
        assert cm.sum() == 499641, "CM total != test size"
fig.suptitle("Confusion matrices on the shared untouched test set (n = 499,641)", fontsize=14, y=1.005)
fig.tight_layout(); fig.savefig(f"{VIZ}/05_confusion_matrices.png", bbox_inches="tight"); plt.close(fig)

# ---------- 6: ROC curves ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
for ax, tech in zip(axes.flat, TECHNIQUES):
    pdf = pd.read_parquet(f"{BASE}/{SUBDIRS[tech]}/test_predictions.parquet")
    y_true = pdf["y_true"].values
    for model in MODELS:
        proba = pdf[f"{model}__proba"].values
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = combined[(combined["Technique"] == tech) & (combined["Model"] == model)]["ROC_AUC"].iloc[0]
        ax.plot(fpr, tpr, label=f"{model} (AUC={auc:.4f})", color=MODEL_COLORS[model], lw=1.8)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_title(tech); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.suptitle("ROC curves per imbalance technique (same test set)", fontsize=14)
fig.tight_layout(); fig.savefig(f"{VIZ}/06_roc_curves.png", bbox_inches="tight"); plt.close(fig)

# ---------- 7: Precision-Recall curves ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
base_rate = 85008 / 499641
for ax, tech in zip(axes.flat, TECHNIQUES):
    pdf = pd.read_parquet(f"{BASE}/{SUBDIRS[tech]}/test_predictions.parquet")
    y_true = pdf["y_true"].values
    for model in MODELS:
        proba = pdf[f"{model}__proba"].values
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = combined[(combined["Technique"] == tech) & (combined["Model"] == model)]["PR_AUC"].iloc[0]
        ax.plot(rec, prec, label=f"{model} (AP={ap:.4f})", color=MODEL_COLORS[model], lw=1.8)
    ax.axhline(base_rate, color="k", ls="--", lw=1, label=f"Baseline ({base_rate:.3f})")
    ax.set_title(tech); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(loc="upper right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.suptitle("Precision-Recall curves per imbalance technique (same test set)", fontsize=14)
fig.tight_layout(); fig.savefig(f"{VIZ}/07_pr_curves.png", bbox_inches="tight"); plt.close(fig)

# ---------- 8-12: metric comparison grouped bars ----------
def metric_bar(metric, fname, title):
    order = combined["Technique"].unique()
    x = np.arange(len(order)); w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k, model in enumerate(MODELS):
        vals = [combined[(combined["Technique"] == t) & (combined["Model"] == model)][metric].iloc[0]
                for t in order]
        bars = ax.bar(x + (k - 1) * w, vals, width=w, label=model, color=MODEL_COLORS[model],
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=12)
    ax.set_ylabel(metric.replace("_", "-")); ax.set_title(title)
    ax.set_ylim(0, 1.02)  # honest axis starting at 0
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{VIZ}/{fname}", bbox_inches="tight"); plt.close(fig)

metric_bar("Accuracy", "08_accuracy_comparison.png", "Accuracy by technique and model (test set, n=499,641)")
metric_bar("Recall", "09_recall_comparison.png", "Recall (sensitivity) by technique and model")
metric_bar("F1", "10_f1_comparison.png", "F1-score by technique and model")
metric_bar("ROC_AUC", "11_roc_auc_comparison.png", "ROC-AUC by technique and model")
metric_bar("PR_AUC", "12_pr_auc_comparison.png", "PR-AUC (average precision) by technique and model")

print("12 visualizations saved to", VIZ)
