"""STEP 3 -- Paired McNemar tests on the frozen corrected-scope test set.

Comparisons (per classifier, exact same test rows):
  * within Person 2: 5 runs (baseline + 4 techniques) pairwise
  * Person 2 techniques vs Person 1's corrected reference runs
    (No Balancing, Random Oversampling, SMOTE, ADASYN)

Statistic: chi2 = (b - c)^2 / (b + c), df = 1 (no continuity correction, same
as the original Person 2 analysis). Bonferroni alpha = 0.05 / n_tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def load_p2(out):
    preds = {}
    for tech, slug in C.TECHNIQUE_SLUGS.items():
        p = out / f"test_predictions_{slug}.parquet"
        if p.exists():
            preds[f"P2: {tech}"] = pd.read_parquet(p)
    p = out / f"test_predictions_{C.BASELINE_SLUG}.parquet"
    if p.exists():
        preds[f"P2: {C.BASELINE}"] = pd.read_parquet(p)
    return preds


def load_p1():
    preds = {}
    for tech, path in C.P1_PREDICTIONS.items():
        if path.exists():
            preds[f"P1: {tech}"] = pd.read_parquet(path)
    return preds


def mcnemar(a, b, y):
    b_cnt = int(np.count_nonzero((a == y) & (b != y)))
    c_cnt = int(np.count_nonzero((a != y) & (b == y)))
    stat = ((b_cnt - c_cnt) ** 2) / (b_cnt + c_cnt) if (b_cnt + c_cnt) else 0.0
    return b_cnt, c_cnt, stat, float(chi2.sf(stat, 1))


def main():
    out = C.BASE / "results"
    p2 = load_p2(out)
    p1 = load_p1()
    all_preds = {**p2, **p1}
    if len(p2) != 5:
        raise SystemExit(f"expected 5 P2 prediction files, found {len(p2)}: {list(p2)}")

    y_ref = None
    for name, df in all_preds.items():
        y = df["y_true"].to_numpy()
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y, y_ref):
            raise SystemExit(f"y_true mismatch in {name}")

    rows = []
    p2_names = list(p2)
    p1_names = list(p1)

    for model in C.MODELS:
        # within Person 2
        for i in range(len(p2_names)):
            for j in range(i + 1, len(p2_names)):
                na, nb = p2_names[i], p2_names[j]
                a = p2[na][f"{model}__pred"].to_numpy()
                b = p2[nb][f"{model}__pred"].to_numpy()
                bc, cb, stat, p = mcnemar(a, b, y_ref)
                rows.append({"Model": model, "Group_A": "P2", "Tech_A": na,
                             "Group_B": "P2", "Tech_B": nb,
                             "A_right_B_wrong": bc, "A_wrong_B_right": cb,
                             "chi2": round(stat, 3), "p_value": p})
        # Person 2 techniques vs Person 1 references
        for na in p2_names:
            for nb in p1_names:
                a = p2[na][f"{model}__pred"].to_numpy()
                b = p1[nb][f"{model}__pred"].to_numpy()
                bc, cb, stat, p = mcnemar(a, b, y_ref)
                rows.append({"Model": model, "Group_A": "P2", "Tech_A": na,
                             "Group_B": "P1", "Tech_B": nb,
                             "A_right_B_wrong": bc, "A_wrong_B_right": cb,
                             "chi2": round(stat, 3), "p_value": p})

    df = pd.DataFrame(rows)
    alpha = 0.05 / len(df)
    df["significant_bonf"] = df["p_value"] < alpha
    df.to_csv(out / "mcnemar_results.csv", index=False)
    C.log(f"McNemar: {len(df)} paired tests, Bonferroni alpha={alpha:.6f}, "
          f"significant={int(df['significant_bonf'].sum())}/{len(df)}")
    C.log(f"  saved {out / 'mcnemar_results.csv'}")


if __name__ == "__main__":
    main()
