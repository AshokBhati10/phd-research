"""P4 STEP 4 -- Six figures (PNG + PDF each).

F1 headline; accuracy-recall Pareto; ROC/PR-AUC bars; resampling mechanics;
family-champion curves; best-model heatmap.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_p4 as C

OUT = C.BASE / "results"
FIG = C.BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {"LogisticRegression": "#4C72B0", "RandomForest": "#55A868", "XGBoost": "#C44E52"}
FAMILY_COLORS = {"Baseline": "#7F7F7F", "Basic oversampling": "#4C72B0",
                 "Advanced oversampling": "#DD8452", "Cleaning & hybrid": "#55A868"}

comb = pd.read_csv(OUT / "combined_results.csv")
order = [t for t in C.CANONICAL_ORDER]
comb["ord"] = comb["Technique"].map({t: i for i, t in enumerate(order)})
comb = comb.sort_values(["ord", "Model"])


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{stem}.png/.pdf")


# --- Figure 1: F1 headline ---------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 4.6))
x = np.arange(len(order))
w = 0.26
for k, model in enumerate(C.MODELS):
    vals = [float(comb[(comb["Technique"] == t) & (comb["Model"] == model)]["F1"].iloc[0]) for t in order]
    ax.bar(x + (k - 1) * w, vals, w, label=model, color=MODEL_COLORS[model])
labels = []
for t in order:
    fam = C.FAMILY_OF[t]
    prefix = {"Baseline": "", "Basic oversampling": "A. ", "Advanced oversampling": "B. ",
              "Cleaning & hybrid": "C. "}[fam]
    labels.append(("P1 " if prefix in ("A. ", "") and t != "Tomek Links" else
                   "P3 " if fam == "Cleaning & hybrid" else
                   "P2 " if fam == "Advanced oversampling" else "") + prefix + t)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.4, rotation=28, ha="right")
for fam, color in FAMILY_COLORS.items():
    pass
ax.set_ylabel("F1 (minority class)")
ax.set_title("F1-score across all 12 techniques x 3 classifiers (same frozen test set, n=459,434)",
             fontweight="bold")
ax.legend(fontsize=8, ncol=3)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "01_f1_headline")

# --- Figure 2: accuracy-recall Pareto ---------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 5.8))
for _, r in comb.iterrows():
    scaled = r["Technique"] == "SVM-SMOTE" or r["Technique"] == "SMOTE-N"
    ax.scatter(r["Recall"], r["Accuracy"], s=46,
               color=FAMILY_COLORS[r["Family"]],
               marker={"LogisticRegression": "o", "RandomForest": "s", "XGBoost": "^"}[r["Model"]],
               edgecolor="black", linewidth=0.3,
               alpha=0.9)
for t in ["No Balancing", "Random Oversampling", "SVM-SMOTE", "ENN"]:
    g = comb[comb["Technique"] == t]
    r = g.loc[g["F1"].idxmax()]
    ax.annotate(r["Technique"], (r["Recall"], r["Accuracy"]),
                textcoords="offset points", xytext=(7, 5), fontsize=7.6, fontweight="bold")
best = comb.loc[comb.groupby("Model")["F1"].idxmax()]
handles = [plt.Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in FAMILY_COLORS.items()]
ax.legend(handles=handles, fontsize=8, title="Family")
ax.set_xlabel("Recall (minority class, Yes = diabetes)")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs recall trade-off - no technique dominates both axes", fontweight="bold")
ax.grid(alpha=0.3)
save(fig, "02_accuracy_recall_pareto")

# --- Figure 3: ROC-AUC and PR-AUC bars --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
for ax, metric, title in [(axes[0], "ROC_AUC", "ROC-AUC"), (axes[1], "PR_AUC", "PR-AUC (average precision)")]:
    vals = [float(comb[(comb["Technique"] == t) & (comb["Model"] == "XGBoost")][metric].iloc[0])
            for t in order]
    colors = [FAMILY_COLORS[C.FAMILY_OF[t]] for t in order]
    ax.bar(x, vals, 0.7, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[i].replace("P1 ", "").replace("P2 ", "").replace("P3 ", "")
                        for i in range(len(order))], fontsize=6.8, rotation=32, ha="right")
    ax.set_title(f"{title} - XGBoost (best single model)", fontweight="bold", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    lo = min(vals) - 0.02
    ax.set_ylim(max(0, lo), min(1.0, max(vals) + 0.015))
handles = [plt.Line2D([], [], marker="s", ls="", color=v, label=k) for k, v in FAMILY_COLORS.items()]
axes[1].legend(handles=handles, fontsize=7, title="Family")
save(fig, "03_auc_ranking_quality")

# --- Figure 4: resampling mechanics -----------------------------------------
eff = pd.read_csv(OUT / "resampling_effects.csv")
eff["ord"] = eff["Technique"].map({t: i for i, t in enumerate(order)})
eff = eff.sort_values("ord")
fig, ax = plt.subplots(figsize=(11, 4.4))
xs = np.arange(len(eff))
ax.bar(xs - 0.2, eff["Rows_added"] / 1e6, 0.4, color="#DD8452", label="rows added (oversampled)")
ax.bar(xs + 0.2, eff["Rows_removed"] / 1e6, 0.4, color="#55A868", label="rows removed (cleaned)")
ax2 = ax.twinx()
ax2.plot(xs, eff["Final_train_size"] / 1e6, "o--", color="#333333", ms=4, lw=1,
         label="final train size (right axis)")
ax2.set_ylabel("Final train size (millions)", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels([t + (" \u2020" if t in ("SVM-SMOTE", "SMOTE-N") else "") for t in eff["Technique"]],
                   fontsize=7.4, rotation=28, ha="right")
ax.set_ylabel("Rows added / removed (millions)")
ax.set_title("What each technique does to the 1.84M-row training set (train-only; test untouched)",
             fontweight="bold")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "04_resampling_mechanics")

# --- Figure 5: family champions ROC + PR ------------------------------------
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, roc_auc_score
fam_champion = {fam: comb[comb["Family"] == fam].groupby("Technique")["F1"].mean().idxmax()
                for fam in ["Basic oversampling", "Advanced oversampling", "Cleaning & hybrid"]}

parq = {src["technique"]: src for src in C.SOURCES}
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
model = "XGBoost"
for fam, tech in fam_champion.items():
    src = next(s for s in C.SOURCES if s["technique"] == tech)
    df = C.load_predictions(src)
    y = df["y_true"].to_numpy()
    p = df[f"{model}__proba"].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, lw=1.6, color=FAMILY_COLORS[fam],
                 label=f"{tech} (ROC-AUC {roc_auc_score(y, p):.4f})")
    prec, rec, _ = precision_recall_curve(y, p)
    axes[1].plot(rec, prec, lw=1.6, color=FAMILY_COLORS[fam],
                 label=f"{tech} (PR-AUC {average_precision_score(y, p):.4f})")
src = next(s for s in C.SOURCES if s["technique"] == "No Balancing")
df = C.load_predictions(src)
y = df["y_true"].to_numpy()
p = df[f"{model}__proba"].to_numpy()
fpr, tpr, _ = roc_curve(y, p)
axes[0].plot(fpr, tpr, "k--", lw=1.2, label=f"No Balancing (ROC-AUC {roc_auc_score(y, p):.4f})")
prec, rec, _ = precision_recall_curve(y, p)
axes[1].plot(rec, prec, "k--", lw=1.2, label=f"No Balancing (PR-AUC {average_precision_score(y, p):.4f})")
axes[0].set_title(f"ROC - family champions ({model})", fontweight="bold", fontsize=10)
axes[1].set_title(f"Precision-recall - family champions ({model})", fontweight="bold", fontsize=10)
for ax, xl, yl in [(axes[0], "False positive rate", "True positive rate"),
                   (axes[1], "Recall", "Precision")]:
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.legend(fontsize=7.4, loc="lower right" if ax is axes[0] else "upper right")
    ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "05_family_champion_curves")

# --- Figure 6: best-model heatmap -------------------------------------------
metrics6 = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
gm = comb[comb["Model"] == "XGBoost"].set_index("Technique").loc[order]
fig, ax = plt.subplots(figsize=(7.4, 5.4))
data = gm[metrics6].values
im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.1, vmax=0.85)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7.2)
ax.set_xticks(range(len(metrics6)), metrics6, fontsize=8, rotation=20)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([t if t != "No Balancing" else "No Balancing (base)" for t in order], fontsize=7.6)
ax.set_title("XGBoost view: all techniques x 6 metrics (best classifier overall)", fontweight="bold")
fig.colorbar(im, ax=ax, shrink=0.75)
fig.tight_layout()
save(fig, "06_metric_heatmap")

print("figures complete")
