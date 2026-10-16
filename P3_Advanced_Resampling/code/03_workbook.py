"""Stage 3: build the Person 3 results workbook (Person 1 structure).

Sheets: one per technique (ENN, Tomek_Links, SMOTE_Tomek, SMOTE_ENN),
Combined_results, Class_distributions, Protocol_config, Run_log.

Usage:
    python3 03_workbook.py --results-dir ../results --out ../workbook/Person_3_results_workbook.xlsx
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TECHNIQUES, MODELS, RESULTS_HEADER, RNG, FEATURES, TARGET, describe_resampler

SHEET_NAMES = {"ENN": "ENN", "Tomek Links": "Tomek_Links",
               "SMOTE-Tomek": "SMOTE_Tomek", "SMOTE-ENN": "SMOTE_ENN"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--data-dir", required=True, help="directory holding manifest.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pandas as pd

    results_csv = os.path.join(args.results_dir, "results_person3.csv")
    df = pd.read_csv(results_csv)
    with open(os.path.join(args.results_dir, "class_distributions.json")) as f:
        class_dists = json.load(f)
    with open(os.path.join(args.results_dir, "audit.json")) as f:
        audit = json.load(f)
    with open(os.path.join(args.data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(args.results_dir, "environment.json")) as f:
        env = json.load(f)
    run_log = ""
    lp = os.path.join(args.results_dir, "run.log")
    if os.path.exists(lp):
        with open(lp) as f:
            run_log = f.read()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as w:
        for tech in TECHNIQUES:
            sub = df[df["Technique"] == tech].copy()
            sub.to_excel(w, sheet_name=SHEET_NAMES[tech], index=False)
        df.to_excel(w, sheet_name="Combined_results", index=False)

        cd_rows = []
        for tech in TECHNIQUES:
            d = class_dists[tech]
            cd_rows.append({
                "Technique": tech,
                "Train_No_before": d["train_before"]["0"],
                "Train_Yes_before": d["train_before"]["1"],
                "Train_No_after": d["train_after"]["0"],
                "Train_Yes_after": d["train_after"]["1"],
                "Added_samples": d["added_samples"],
                "Removed_samples": d.get("removed_samples", 0),
                "Final_train_size": d["final_train_size"],
            })
        pd.DataFrame(cd_rows).to_excel(w, sheet_name="Class_distributions", index=False)

        proto_rows = [
            ("dataset_scope", manifest["scope"]),
            ("n_analytical", manifest["n_analytical"]),
            ("n_2014_excluded", manifest["n_2014_excluded"]),
            ("target", TARGET),
            ("target_encoding", "No -> 0; Yes -> 1"),
            ("features", ", ".join(FEATURES)),
            ("n_features", len(FEATURES)),
            ("feature_representation", manifest["feature_dtype"]),
            ("YEAR", "metadata only, not a predictor"),
            ("split", "stratified 80/20 train/test, shuffle=True"),
            ("random_state", RNG),
            ("n_train", manifest["n_train"]),
            ("n_test", manifest["n_test"]),
            ("train_class_counts", json.dumps(manifest["train_class_counts"])),
            ("test_class_counts", json.dumps(manifest["test_class_counts"])),
            ("test_fingerprint_sha256", manifest["test_fingerprint_sha256"]),
            ("test_set_policy", "frozen; never resampled/balanced/augmented; fingerprint verified every run"),
        ]
        for tech in TECHNIQUES:
            proto_rows.append((f"resampler_{SHEET_NAMES[tech]}", describe_resampler(tech)))
        proto_rows += [
            ("model_LogisticRegression",
             "max_iter=1000, solver=lbfgs, random_state=42; StandardScaler fitted on RESAMPLED train (Person 1 code)"),
            ("model_RandomForest",
             "n_estimators=100, max_depth=12, n_jobs=-1, random_state=42"),
            ("model_XGBoost",
             "n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, "
             "colsample_bytree=0.8, tree_method=hist, eval_metric=logloss, n_jobs=-1, random_state=42"),
            ("decision_threshold", 0.5),
            ("metrics", "Accuracy, Precision, Recall, F1, ROC_AUC, PR_AUC (average precision), "
                        "TN, FP, FN, TP, fit_predict_sec"),
        ]
        for k in ["python", "numpy", "pandas", "sklearn", "imblearn", "xgboost", "pyarrow", "openpyxl"]:
            proto_rows.append((f"pkg_{k}", env.get(k)))
        proto_rows += [
            ("platform", env.get("platform")),
            ("cpu_count", env.get("cpu_count")),
            ("mem_total_kb", env.get("mem_total_kb")),
        ]
        pd.DataFrame(proto_rows, columns=["Parameter", "Value"]).to_excel(
            w, sheet_name="Protocol_config", index=False)

        pd.DataFrame({"run_log": run_log.splitlines()}).to_excel(
            w, sheet_name="Run_log", index=False)

    print(f"workbook written: {args.out}")


if __name__ == "__main__":
    main()
