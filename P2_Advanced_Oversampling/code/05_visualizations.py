"""STEP 5 -- Person 2 figures (Person 1/3 style)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

OUT = C.BASE / "results"
FIG = C.BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TECH_ORDER = [C.BASELINE] + C.TECHNIQUES
LABELS = {
    C.BASELINE: "No Balancing",
    "Borderline-SMOTE": "Borderline-SMOTE",
    "KMeans-SMOTE": "KMeans-SMOTE",
    "SVM-SMOTE": "SVM-SMOTE \u2020",
    "SMOTE-N": "SMOTE-N \u2020",
}
COLORS = {"LogisticRegression": "#4C72B0", "RandomForest": "#55A868", "XGBoost": "#C44E52"}
SLUGS = {C.BASELINE: C.BASELINE_SLUG, **C.TECHNIQUE_SLUGS}


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{stem}.png/.pdf")


def metric_bars(df, metric, stem, title):
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(TECH_ORDER))
    w = 0.26
    for k, model in enumerate(C.MODELS):
        vals = []
        for tech in TECH_ORDER:
            row = df[(df["Technique"] == tech) & (df["Model"] == model)]
            vals.append(float(row[metric].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (k - 1) * w, vals, w, label=model, color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[t] for t in TECH_ORDER], fontsize=9)
    ax.set_ylabel(metric.replace("_", "-"))
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    save(fig, stem)


def main():
    results = pd.read_csv(OUT / "results_person2.csv")
    baseline = pd.read_csv(OUT / "results_baseline_no_balancing.csv")
    df = pd.concat([baseline, results], ignore_index=True)
    dists = C.load_json(OUT / "class_distributions.json", {}) or {}

    for metric, stem, title in [
        ("F1", "01_f1_comparison", "F1-score by technique and classifier (frozen corrected test set)"),
        ("Recall", "02_recall_comparison", "Recall (minority class) by technique and classifier"),
        ("Precision", "03_precision_comparison", "Precision by technique and classifier"),
        ("PR_AUC", "04_pr_auc_comparison", "PR-AUC by technique and classifier"),
        ("ROC_AUC", "05_roc_auc_comparison", "ROC-AUC by technique and classifier"),
        ("Accuracy", "06_accuracy_comparison", "Accuracy by technique and classifier"),
    ]:
        metric_bars(df, metric, stem, title)

    # --- class distributions -------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    xs, labels = [], []
    pos = 0
    for tech in TECH_ORDER:
        d = dists.get(tech)
        if not d:
            continue
        for phase, tag in ((d["train_before"], "before"), (d["train_after"], "after")):
            n0, n1 = phase.get("0", 0), phase.get("1", 0)
            ax.bar(pos, n0 / 1e6, color="#8C8C8C", width=0.8)
            ax.bar(pos, n1 / 1e6, bottom=n0 / 1e6, color="#C44E52", width=0.8)
            xs.append(pos)
            labels.append(f"{LABELS[tech]}\n{tag}")
            pos += 1
        pos += 0.6
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
    ax.set_ylabel("Rows (millions)")
    ax.set_title("Training class distribution before/after resampling", fontweight="bold")
    ax.legend(["No (majority)", "Yes (minority)"], fontsize=8)
    save(fig, "07_class_distribution")

    # --- confusion matrices --------------------------------------------------
    fig, axes = plt.subplots(len(TECH_ORDER), 3, figsize=(12, 15))
    for i, tech in enumerate(TECH_ORDER):
        p = OUT / f"test_predictions_{SLUGS[tech]}.parquet"
        preds = pd.read_parquet(p)
        y = preds["y_true"].to_numpy()
        for j, model in enumerate(C.MODELS):
            ax = axes[i, j]
            pred = preds[f"{model}__pred"].to_numpy()
            tn = int(((y == 0) & (pred == 0)).sum())
            fp = int(((y == 0) & (pred == 1)).sum())
            fn = int(((y == 1) & (pred == 0)).sum())
            tp = int(((y == 1) & (pred == 1)).sum())
            cm = np.array([[tn, fp], [fn, tp]])
            im = ax.imshow(cm, cmap="Blues")
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, f"{cm[r, c]:,}", ha="center", va="center", fontsize=8,
                            color="white" if cm[r, c] > cm.max() / 2 else "black")
            ax.set_xticks([0, 1], ["No", "Yes"])
            ax.set_yticks([0, 1], ["No", "Yes"])
            ax.set_title(f"{LABELS[tech]} \u00b7 {model}", fontsize=9)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
    fig.suptitle("Confusion matrices \u2014 frozen corrected test set (n=459,434)",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "08_confusion_matrices")

    # --- PR / ROC curves (XGBoost) ------------------------------------------
    model = "XGBoost"
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for tech in TECH_ORDER:
        preds = pd.read_parquet(OUT / f"test_predictions_{SLUGS[tech]}.parquet")
        y = preds["y_true"].to_numpy()
        p = preds[f"{model}__proba"].to_numpy()
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, lw=1.4, label=LABELS[tech])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curves \u2014 {model}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "09_pr_curves")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for tech in TECH_ORDER:
        preds = pd.read_parquet(OUT / f"test_predictions_{SLUGS[tech]}.parquet")
        y = preds["y_true"].to_numpy()
        p = preds[f"{model}__proba"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.4, label=LABELS[tech])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves \u2014 {model}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "10_roc_curves")


if __name__ == "__main__":
    main()
