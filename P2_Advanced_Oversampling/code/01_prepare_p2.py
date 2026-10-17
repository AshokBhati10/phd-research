"""STEP 1 -- Verify the frozen corrected-scope splits and write P2 metadata.

Loads Person 1's corrected 2015-2024 splits (the same arrays Person 3 used),
verifies the SHA-256 test fingerprint, regenerates the shared stratified 50k
train subsample used by SVM-SMOTE / SMOTE-N, and writes:
  metadata/manifest.json
  metadata/environment.json
  results/subsample_X_50k.parquet
  results/subsample_y_50k.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main():
    C.log("=== P2 prepare: verifying frozen corrected-scope splits ===")
    files = {
        "X_train": C.P1_SPLITS / "X_train.parquet",
        "X_test": C.P1_SPLITS / "X_test.parquet",
        "y_train": C.P1_SPLITS / "y_train.parquet",
        "y_test": C.P1_SPLITS / "y_test.parquet",
    }
    for name, path in files.items():
        if not path.exists():
            raise SystemExit(f"missing split file: {path}")
        C.log(f"  {name}: {path}")

    X_train = pd.read_parquet(files["X_train"])
    X_test = pd.read_parquet(files["X_test"])
    y_train = pd.read_parquet(files["y_train"])["y"].astype(np.int64)
    y_test = pd.read_parquet(files["y_test"])["y"].astype(np.int64)

    # --- structural validation -------------------------------------------------
    assert list(X_train.columns) == C.FEATURES, "feature list/order mismatch"
    assert list(X_test.columns) == C.FEATURES, "test feature list/order mismatch"
    assert X_train.dtypes.eq(np.int64).all(), f"non-int64 train dtypes: {X_train.dtypes}"
    assert X_test.dtypes.eq(np.int64).all(), f"non-int64 test dtypes: {X_test.dtypes}"
    assert X_train.shape == (C.EXPECTED_TRAIN_SIZE, 15), f"train shape {X_train.shape}"
    assert X_test.shape == (C.EXPECTED_TEST_SIZE, 15), f"test shape {X_test.shape}"
    assert y_train.value_counts().sort_index().to_dict() == C.EXPECTED_TRAIN_COUNTS
    assert y_test.value_counts().sort_index().to_dict() == C.EXPECTED_TEST_COUNTS
    assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0
    assert C.TARGET not in X_train.columns and "YEAR" not in X_train.columns

    # --- fingerprint (must equal Person 3's manifest) --------------------------
    fp = C.fingerprint(X_test.values, y_test.values)
    C.log(f"  test fingerprint: {fp}")
    if fp != C.EXPECTED_TEST_FINGERPRINT:
        raise SystemExit(
            "TEST FINGERPRINT MISMATCH -- refusing to produce results.\n"
            f"  expected {C.EXPECTED_TEST_FINGERPRINT}\n  got      {fp}"
        )
    C.log("  fingerprint matches Person 3 manifest (70e180c6...)")

    # --- shared stratified 50k train subsample (SVM-SMOTE / SMOTE-N) -----------
    Xsub, _, ysub, _ = train_test_split(
        X_train, y_train, train_size=50_000, stratify=y_train, random_state=C.RNG,
    )
    Xsub = Xsub.reset_index(drop=True)
    ysub = ysub.reset_index(drop=True)
    assert len(Xsub) == 50_000
    sub_counts = ysub.value_counts().sort_index().to_dict()
    C.log(f"  shared 50k subsample: {sub_counts}")

    out = C.BASE / "results"
    Xsub.to_parquet(out / "subsample_X_50k.parquet", index=False)
    ysub.to_frame("y").to_parquet(out / "subsample_y_50k.parquet", index=False)

    manifest = {
        "scope": "BRFSS 2015-2024 (YEAR 2014 excluded), corrected Person 1 scope",
        "source": str(C.P1_SPLITS),
        "target": C.TARGET,
        "features": C.FEATURES,
        "n_features": len(C.FEATURES),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_counts": C.class_counts(y_train),
        "test_counts": C.class_counts(y_test),
        "test_fingerprint_sha256": fp,
        "expected_test_fingerprint": C.EXPECTED_TEST_FINGERPRINT,
        "fingerprint_match_person3": fp == C.EXPECTED_TEST_FINGERPRINT,
        "split_protocol": "stratified 80/20, random_state=42, shuffle=True (Person 1)",
        "subsample_50k": {
            "n_rows": int(len(Xsub)),
            "counts": {str(k): int(v) for k, v in sub_counts.items()},
            "seed": C.RNG,
            "method": "train_test_split(train_size=50000, stratify=y, random_state=42)",
            "used_by": ["SVM-SMOTE", "SMOTE-N"],
            "fingerprint_sha256": C.fingerprint(Xsub.values, ysub.values),
        },
        "random_state": C.RNG,
        "generated_at_utc": C.environment_record()["recorded_at_utc"],
    }
    C.save_json(C.BASE / "metadata" / "manifest.json", manifest)
    C.save_json(C.BASE / "metadata" / "environment.json", C.environment_record())

    C.log("=== prepare complete: manifest + environment written ===")
    C.log(f"  train {len(X_train):,} / test {len(X_test):,} / fingerprint OK")


if __name__ == "__main__":
    main()
