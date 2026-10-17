"""P4 STEP 3 -- Statistical comparison on the shared frozen test set.

Per classifier: pairwise McNemar tests over all C(12,2)=66 technique pairs
(198 total), Holm-corrected within each model, with effect sizes (accuracy and
recall differences). Optional --bootstrap adds paired bootstrap 95% CIs for
ROC/PR-AUC differences between per-model champions and their main rivals
(500 resamples, seed 42).

Outputs:
  results/statistical_tests_mcnemar.csv   (full 198-row matrix)
  results/statistical_tests_summary.csv
  results/auc_bootstrap_cis.csv           (with --bootstrap)
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_p4 as C

OUT = C.BASE / "results"


def family_pair_group(fa, fb):
    if fa == fb:
        return f"within-family ({fa})"
    return "cross-family"


def champion_for(df, model, metric="F1"):
    gm = df[(df["Model"] == model)]
    row = gm.loc[gm[metric].idxmax()]
    return row["Technique"]


def main(bootstrap=False):
    preds = {src["technique"]: C.load_predictions(src)
             for src in C.SOURCES
             if not (src["person"] == "P2" and src["technique"] == "No Balancing")}
    y = preds["No Balancing"]["y_true"].to_numpy()
    n_pos = int((y == 1).sum())
    techniques = list(preds)

    rows = []
    for model in C.MODELS:
        pairs = list(combinations(techniques, 2))
        rec = {}
        for t in techniques:
            d = preds[t]
            p = d[f"{model}__pred"].to_numpy()
            rec[t] = {"acc": float((p == y).mean()), "recall": float(((p == 1) & (y == 1)).sum() / n_pos)}
        raw = []
        for a, b in pairs:
            pa = preds[a][f"{model}__pred"].to_numpy()
            pb = preds[b][f"{model}__pred"].to_numpy()
            bc, cb, stat, p = C.mcnemar(pa, pb, y)
            raw.append({
                "Model": model,
                "Tech_A": a, "Tech_B": b,
                "Group": family_pair_group(C.FAMILY_OF[a], C.FAMILY_OF[b]),
                "A_right_B_wrong": bc, "A_wrong_B_right": cb,
                "chi2": round(stat, 3), "p_value": p,
                "Accuracy_diff_A_minus_B": rec[a]["acc"] - rec[b]["acc"],
                "Recall_diff_A_minus_B": rec[a]["recall"] - rec[b]["recall"],
            })
        rv = C.holm([r["p_value"] for r in raw])
        for r, pv in zip(raw, rv):
            r["p_holm"] = float(pv)
            r["significant_holm"] = bool(pv < 0.05)
            rows.append(r)

    mdf = pd.DataFrame(rows)
    mdf.to_csv(OUT / "statistical_tests_mcnemar.csv", index=False)

    # --- summary -------------------------------------------------------------
    comb = pd.read_csv(OUT / "combined_results.csv")
    summary_rows = []
    for model in C.MODELS:
        mm = mdf[mdf["Model"] == model]
        champ = champion_for(comb, model)
        base_mask = (mm["Tech_A"] == "No Balancing") | (mm["Tech_B"] == "No Balancing")
        base_rows = []
        for _, r in mm[base_mask].iterrows():
            other = r["Tech_B"] if r["Tech_A"] == "No Balancing" else r["Tech_A"]
            rd = r["Recall_diff_A_minus_B"] * (1 if other != r["Tech_A"] else -1)
            rd = r["Recall_diff_A_minus_B"] if r["Tech_A"] == other else (
                -r["Recall_diff_A_minus_B"] if r["Tech_A"] == "No Balancing" else r["Recall_diff_A_minus_B"])
            # normalize: diff = other - baseline
            rd = (r["Recall_diff_A_minus_B"] if r["Tech_A"] == "No Balancing"
                  else -r["Recall_diff_A_minus_B"])
            base_rows.append({"Model": model, "Technique": other,
                              "p_holm_vs_baseline": r["p_holm"],
                              "significant_vs_baseline": r["significant_holm"],
                              "Recall_gain_over_baseline": rd})
        fam_champs = {}
        for fam in ["Basic oversampling", "Advanced oversampling", "Cleaning & hybrid"]:
            sub = comb[(comb["Model"] == model) & (comb["Family"] == fam)]
            fam_champs[fam] = sub.loc[sub["F1"].idxmax(), "Technique"]
        summary_rows.append({
            "Model": model,
            "F1_champion": champ,
            "Family_champions": "; ".join(f"{f}: {t}" for f, t in fam_champs.items()),
            "Techniques_beating_baseline_F1_holm": int(sum(
                1 for b in base_rows if b["significant_vs_baseline"])),
            "Techniques_vs_baseline_total": len(base_rows),
            "Baseline_recalls_beaten": "; ".join(
                f"{b['Technique']} (+{b['Recall_gain_over_baseline']:.3f})"
                for b in base_rows if b["significant_vs_baseline"]),
        })
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(OUT / "statistical_tests_summary.csv", index=False)

    print(f"McNemar tests: {len(mdf)} | significant (Holm 0.05): {int(mdf['significant_holm'].sum())}")
    for _, r in sdf.iterrows():
        print(f"  {r['Model']:20s} F1 champion: {r['F1_champion']:22s} | "
              f"{r['Techniques_beating_baseline_F1_holm']}/{r['Techniques_vs_baseline_total']} beat baseline (Holm)")

    if not bootstrap:
        return

    # --- bootstrap AUC CIs: champion vs baseline and other family champions ---
    y8 = y.astype(np.int8)
    Cara = []
    jobs = []
    for model in C.MODELS:
        champ = champion_for(comb, model)
        rivals = ["No Balancing"]
        fam_champs = {}
        for fam in ["Basic oversampling", "Advanced oversampling", "Cleaning & hybrid"]:
            sub = comb[(comb["Model"] == model) & (comb["Family"] == fam)]
            fam_champs[fam] = sub.loc[sub["F1"].idxmax(), "Technique"]
        for fam, t in fam_champs.items():
            if t != champ:
                rivals.append(t)
        for t in rivals:
            jobs.append((model, champ, t))

    for model, a, b in jobs:
        pa = preds[a][f"{model}__proba"].to_numpy()
        pb = preds[b][f"{model}__proba"].to_numpy()
        for metric in ("roc", "pr"):
            ci = C.auc_bootstrap_ci(y8, pa, pb, metric=metric, n_boot=500, seed=42)
            Cara.append({"Model": model, "Tech_A": a, "Tech_B": b, **ci})
            print(f"  bootstrap {model}: {a} vs {b} [{metric.upper()}_AUC] "
                  f"diff={ci['mean_diff']:+.4f} CI95=[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")
    bdf = pd.DataFrame(Cara)
    bdf.to_csv(OUT / "auc_bootstrap_cis.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true")
    args = ap.parse_args()
    main(bootstrap=args.bootstrap)
