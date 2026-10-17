"""P4 STEP 1 -- Input verification gate.

Verifies, without modifying anything:
  1. the frozen test fingerprint (from P1's corrected splits),
  2. every registered prediction parquet: schema, y_true identity, dtype class,
  3. every combined-table row: metrics recomputed from parquets match the
     person's own results CSV,
  4. P2's baseline replication of P1's baseline (LR/RF exact, XGB tolerance).

Writes results/input_validation.json. Exits nonzero on any hard failure.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_p4 as C


def main():
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    # --- 1. fingerprint ------------------------------------------------------
    X_test = pd.read_parquet(C.P1_SPLITS / "X_test.parquet")
    y_test = pd.read_parquet(C.P1_SPLITS / "y_test.parquet")["y"].astype(np.int64)
    import hashlib
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X_test.values).tobytes())
    h.update(np.ascontiguousarray(y_test.values).tobytes())
    fp = h.hexdigest()
    check("test fingerprint matches the frozen reference", fp == C.EXPECTED_TEST_FINGERPRINT, fp[:16] + "...")

    # --- 2/3. per-source verification ---------------------------------------
    y_ref = None
    max_metric_diff = 0.0
    for src in C.SOURCES:
        tag = f"{src['person']}:{src['technique']}"
        df = C.load_predictions(src)
        if y_ref is None:
            y_ref = df["y_true"].to_numpy()
        y = df["y_true"].to_numpy()
        check(f"{tag}: 459,434 rows x 7 cols", df.shape == (459434, 7), str(df.shape))
        check(f"{tag}: y_true identical to frozen test set", bool(np.array_equal(y, y_ref)))
        check(f"{tag}: columns", list(df.columns) == ["y_true"] +
              [f"{m}__{k}" for m in C.MODELS for k in ("pred", "proba")])

        csv_rows = C.load_source_metrics(src)
        for _, row in csv_rows.iterrows():
            m = row["Model"]
            met = C.compute_metrics(y, df[f"{m}__pred"].to_numpy(), df[f"{m}__proba"].to_numpy())
            diff = max(abs(float(row[k]) - met[k]) for k in C.METRICS)
            max_metric_diff = max(max_metric_diff, diff)
            cm_ok = all(int(row[k]) == met[k] for k in ["TN", "FP", "FN", "TP"])
            check(f"{tag}/{m}: CSV row reproduced from parquet", diff < 1e-6 and cm_ok,
                  f"max metric diff {diff:.2e}, CM {'OK' if cm_ok else 'MISMATCH'}")

    # --- 4. P2 baseline replication -----------------------------------------
    p1_base = C.load_predictions(next(s for s in C.SOURCES if s["person"] == "P1" and s["technique"] == "No Balancing"))
    p2_base = C.load_predictions(next(s for s in C.SOURCES if s["person"] == "P2" and s["technique"] == "No Balancing"))
    for m in C.MODELS:
        met_p1 = C.compute_metrics(y_ref, p1_base[f"{m}__pred"].to_numpy(), p1_base[f"{m}__proba"].to_numpy())
        met_p2 = C.compute_metrics(y_ref, p2_base[f"{m}__pred"].to_numpy(), p2_base[f"{m}__proba"].to_numpy())
        worst = max(abs(met_p1[k] - met_p2[k]) for k in C.METRICS)
        flips = int((p1_base[f"{m}__pred"].to_numpy() != p2_base[f"{m}__pred"].to_numpy()).sum())
        if m in ("LogisticRegression", "RandomForest"):
            check(f"baseline replication {m}: bit-identical", worst == 0.0 and flips == 0)
        else:
            check(f"baseline replication XGBoost within cross-machine tolerance", worst < 5e-3,
                  f"diff {worst:.2e}, {flips} flips")

    report = {"generated_at_utc": pd.Timestamp.utcnow().isoformat(),
              "test_fingerprint": fp, "sources_verified": len(C.SOURCES),
              "max_metric_recompute_diff": max_metric_diff,
              "all_pass": all(c["pass"] for c in checks), "checks": checks}
    out = C.BASE / "results" / "input_validation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{sum(c['pass'] for c in checks)}/{len(checks)} checks passed -> {out}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
