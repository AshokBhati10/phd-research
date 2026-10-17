"""P4 STEP 2 -- Integrate the three persons' validated results.

Builds (all traceable to registered source artifacts):
  results/combined_results.csv      39 rows = baseline + 12 techniques x 3 models
  results/family_summary.csv        per (family, model) means + best technique
  results/winners_tradeoffs.csv     metric winners per model + accuracy/recall trade-off
  results/resampling_effects.csv    train size before/after per technique
  results/report_tables/            markdown tables used verbatim by the report
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_p4 as C

OUT = C.BASE / "results"
TABLES = OUT / "report_tables"
TABLES.mkdir(parents=True, exist_ok=True)

MAIN_COLUMNS = [
    "Family", "Person", "Technique", "Display", "Model",
    "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC",
    "TN", "FP", "FN", "TP", "Fit_predict_sec",
    "Train_source", "Caveat", "Source_results", "Source_predictions",
]

P1_FIT = {"No Balancing": 0}


def build_combined():
    rows = []
    for src in C.SOURCES:
        if src["person"] == "P2" and src["technique"] == "No Balancing":
            continue  # replication row: used in verification, not a separate table row
        csv_rows = C.load_source_metrics(src)
        for _, r in csv_rows.iterrows():
            rows.append({
                "Family": C.FAMILY_OF[src["technique"]],
                "Person": src["person"],
                "Technique": src["technique"],
                "Display": src["display"],
                "Model": r["Model"],
                "Accuracy": float(r["Accuracy"]), "Precision": float(r["Precision"]),
                "Recall": float(r["Recall"]), "F1": float(r["F1"]),
                "ROC_AUC": float(r["ROC_AUC"]), "PR_AUC": float(r["PR_AUC"]),
                "TN": int(r["TN"]), "FP": int(r["FP"]), "FN": int(r["FN"]), "TP": int(r["TP"]),
                "Fit_predict_sec": float(r["fit_predict_sec"]),
                "Train_source": src["train_source"],
                "Caveat": src["caveat"],
                "Source_results": Path(src["results_csv"]).name + ("" if src["person"] != "P1" else " (row Technique='" + src["csv_label"] + "')"),
                "Source_predictions": Path(src["predictions"]).name,
                "fit_ref_sec": src["person"],
            })
    df = pd.DataFrame(rows)
    order = {t: i for i, t in enumerate(C.CANONICAL_ORDER)}
    morder = {m: i for i, m in enumerate(C.MODELS)}
    df = df.sort_values(by=["Technique", "Model"], key=lambda s: s.map(order if s.name == "Technique" else morder))
    df = df.reset_index(drop=True)
    assert len(df) == 36, len(df)  # 1 baseline + 11 resampling techniques x 3 models
    df.to_csv(OUT / "combined_results.csv", index=False)
    return df


def build_family_summary(df):
    rows = []
    for fam, g in df.groupby("Family", sort=False):
        for m in C.MODELS:
            gm = g[g["Model"] == m]
            row = {"Family": fam, "Model": m, "n_techniques": len(gm)}
            for k in C.METRICS:
                row[f"{k}_mean"] = float(gm[k].mean())
                row[f"{k}_best_technique"] = gm.loc[gm[k].idxmax(), "Technique"]
                row[f"{k}_best"] = float(gm[k].max())
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "family_summary.csv", index=False)
    return out


def build_winners(df):
    rows = []
    for m in C.MODELS:
        gm = df[df["Model"] == m]
        for k in C.METRICS:
            best = gm.loc[gm[k].idxmax()]
            rows.append({"Model": m, "Metric": k, "Winner": best["Display"],
                         "Value": float(best[k]), "Person": best["Person"],
                         "Family": best["Family"], "Runner_up": gm.sort_values(k, ascending=False).iloc[1]["Display"],
                         "Runner_up_value": float(gm.sort_values(k, ascending=False).iloc[1][k])})
        base = gm[gm["Technique"] == "No Balancing"].iloc[0]
        best_f1 = gm.loc[gm["F1"].idxmax()]
        rows.append({"Model": m, "Metric": "F1_gain_over_baseline", "Winner": best_f1["Display"],
                     "Value": float(best_f1["F1"] - base["F1"]), "Person": best_f1["Person"],
                     "Family": best_f1["Family"], "Runner_up": "baseline F1",
                     "Runner_up_value": float(base["F1"])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "winners_tradeoffs.csv", index=False)
    return out


def build_mean_ranks(df):
    rows = []
    for m in C.MODELS:
        gm = df[df["Model"] == m].set_index("Technique")
        for k in C.METRICS:
            ranks = gm[k].rank(ascending=False)
            for tech, r in ranks.items():
                rows.append({"Model": m, "Metric": k, "Technique": tech, "Rank": int(r),
                             "Mean_rank_family": C.FAMILY_OF[tech]})
    out = pd.DataFrame(rows)
    mg = out.groupby(["Model", "Technique"], sort=False)["Rank"].mean().reset_index()
    mg["Mean_rank"] = mg["Rank"].round(2)
    mg["Family"] = mg["Technique"].map(C.FAMILY_OF)
    mg = mg.sort_values(["Model", "Mean_rank"])
    mg.to_csv(OUT / "mean_ranks.csv", index=False)
    return mg


def build_resampling_effects():
    rows = []
    for src in C.SOURCES:
        if src["person"] == "P2" and src["technique"] == "No Balancing":
            continue
        d = C.load_dist(src)
        n0b, n1b, n0a, n1a, added, removed, final = C.class_counts_from_dist(d)
        rows.append({
            "Family": C.FAMILY_OF[src["technique"]], "Person": src["person"],
            "Technique": src["technique"], "Display": src["display"],
            "Train_No_before": n0b, "Train_Yes_before": n1b,
            "Train_No_after": n0a, "Train_Yes_after": n1a,
            "Rows_added": added, "Rows_removed": removed, "Final_train_size": final,
            "Net_change": final - (n0b + n1b),
            "Train_source": src["train_source"],
            "Class_counts_source": Path(src["dist_json"]).name,
        })
    out = pd.DataFrame(rows)
    order = {t: i for i, t in enumerate(C.CANONICAL_ORDER)}
    out = out.sort_values("Technique", key=lambda s: s.map(order)).reset_index(drop=True)
    out.to_csv(OUT / "resampling_effects.csv", index=False)
    return out


def md_table(df, float_dp=4, columns=None):
    cols = columns or list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.{float_dp}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report_tables(df, fam, win, eff, ranks):
    main_cols = ["Display", "Person", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    (TABLES / "main_results.md").write_text(md_table(df, columns=main_cols), encoding="utf-8")

    f = fam.copy()
    f["Family"] = f["Family"]
    fam_cols = ["Family", "Model", "n_techniques", "F1_mean", "Recall_mean", "Accuracy_mean",
                "ROC_AUC_mean", "PR_AUC_mean", "F1_best_technique"]
    (TABLES / "family_summary.md").write_text(md_table(f, columns=fam_cols), encoding="utf-8")

    w = win.copy()
    (TABLES / "winners.md").write_text(
        md_table(w[w["Metric"].isin(["Accuracy", "Recall", "F1", "ROC_AUC", "PR_AUC"])],
                 columns=["Model", "Metric", "Winner", "Value", "Family"]), encoding="utf-8")

    eff_cols = ["Display", "Family", "Rows_added", "Rows_removed", "Net_change", "Final_train_size", "Train_source"]
    e = eff.copy()
    e["Display"] = e["Display"].str.replace(" (P2 replication)", "", regex=False)
    (TABLES / "resampling_effects.md").write_text(md_table(e, columns=eff_cols), encoding="utf-8")

    rk = ranks.copy()
    rk_top = rk[rk["Mean_rank"] <= 6]
    (TABLES / "mean_ranks.md").write_text(
        md_table(rk_top, columns=["Model", "Technique", "Family", "Mean_rank"]), encoding="utf-8")


def main():
    df = build_combined()
    fam = build_family_summary(df)
    win = build_winners(df)
    ranks = build_mean_ranks(df)
    eff = build_resampling_effects()
    write_report_tables(df, fam, win, eff, ranks)

    print(f"combined rows: {len(df)} | techniques: {df['Technique'].nunique()} | families: {df['Family'].nunique()}")
    print("\nF1 winners per model:")
    for m in C.MODELS:
        r = win[(win["Model"] == m) & (win["Metric"] == "F1")].iloc[0]
        print(f"  {m:20s} {r['Winner']:24s} F1={r['Value']:.4f} ({r['Family']})")
    print("\nFamily means (F1):")
    for _, r in fam[fam["Model"] == "XGBoost"].iterrows():
        print(f"  {r['Family']:22s} XGBoost F1_mean={r['F1_mean']:.4f}")


if __name__ == "__main__":
    main()
