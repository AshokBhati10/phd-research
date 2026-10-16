"""Stage 2: sequential, resumable experiment runner (Person 3 techniques x 3 models).

MODIFICATIONS:
- SMOTE-ENN now uses explicit two-stage pipeline (smote_enn_staged.py)
- SMOTE output is checkpointed in results/.smote_enn_checkpoint/
- If SMOTE-ENN fails at ENN stage, SMOTE checkpoint is preserved for resume
- All other techniques unchanged

- Strictly sequential: one technique at a time, one model at a time.
- Resampling touches TRAINING data only. The test set is loaded once, treated
  read-only, and its SHA-256 fingerprint is verified before/after every technique.
- Checkpoint: a (technique, model) pair is complete iff its row exists in the
  results CSV. Reruns skip complete pairs. --force is required to redo.
- Resampling is deterministic (random_state=42 wherever supported), so a resume
  re-derives the resampled train instead of storing multi-GB intermediates.
- Memory: resampled arrays are released after each technique (del + gc.collect()).

Usage:
    python3 02_run_with_staged_smote_enn.py --data-dir ../data --out-dir ../results
    python3 02_run_with_staged_smote_enn.py --data-dir ../pilot/data --out-dir ../pilot --force
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    RNG, TECHNIQUES, TECHNIQUE_SLUGS, MODELS, RESULTS_HEADER,
    make_resampler, describe_resampler, make_models, compute_metrics,
    format_result_row, fingerprint, class_counts,
    load_checkpoint, save_checkpoint, completed_pairs,
    environment_record, log,
)
from sklearn.preprocessing import StandardScaler
from smote_enn_staged import run_smote_enn_staged, log_staged


def audit_validity(X_orig, X_res, feature_names, smote_involved):
    """Feature-validity checks in the spirit of Person 1's audit.

    Returns a dict; never claims validity it did not verify.
    For SMOTE-involving techniques: every value must be finite, within the
    observed per-feature [min, max] of the original train, and use only
    codes observed in the original train (strict scan).
    """
    Xr = np.asarray(X_res)
    Xo = np.asarray(X_orig)
    out = {
        "n_features": int(Xr.shape[1]),
        "finite": bool(np.isfinite(Xr).all()),
        "feature_count_ok": bool(Xr.shape[1] == Xo.shape[1]),
    }
    if smote_involved:
        in_range = True
        unobserved = 0
        for j in range(Xr.shape[1]):
            lo, hi = float(Xo[:, j].min()), float(Xo[:, j].max())
            col = Xr[:, j]
            if not (float(col.min()) >= lo - 1e-9 and float(col.max()) <= hi + 1e-9):
                in_range = False
            unseen = np.setdiff1d(np.unique(col), np.unique(Xo[:, j]))
            unobserved += int(unseen.size)
        out["in_observed_range"] = bool(in_range)
        out["unobserved_codes"] = unobserved
    return out


def run_model(model_name, scaling, estimator, Xr, yr, X_test, y_test):
    """Fit on resampled train, evaluate on frozen test. Returns (metrics, fit_predict_sec, y_pred, y_proba)."""
    t0 = time.perf_counter()
    if scaling == "scaled":
        # Person 1's actual implementation: scaler fitted on the RESAMPLED train.
        scaler = StandardScaler().fit(Xr)
        Xr_use = scaler.transform(Xr)
        Xt_use = scaler.transform(X_test)
    else:
        Xr_use, Xt_use = Xr, X_test
    estimator.fit(Xr_use, yr)
    y_pred = estimator.predict(Xt_use)
    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(Xt_use)[:, 1]
    else:  # pragma: no cover - all three models expose predict_proba
        y_proba = estimator.decision_function(Xt_use)
    dt = time.perf_counter() - t0
    metrics = compute_metrics(y_test, y_pred, y_proba)
    return metrics, dt, y_pred, y_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force", action="store_true", help="discard previous results and rerun")
    ap.add_argument("--only-technique", default=None, choices=TECHNIQUES)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logp = args.log or os.path.join(args.out_dir, "run.log")

    results_csv = os.path.join(args.out_dir, "results_person3.csv")
    ckpt_path = os.path.join(args.out_dir, "checkpoint.json")
    audit_path = os.path.join(args.out_dir, "audit.json")
    env_path = os.path.join(args.out_dir, "environment.json")
    cd_path = os.path.join(args.out_dir, "class_distributions.json")
    
    # Checkpoint dir for SMOTE-ENN staged pipeline
    smote_enn_ckpt_dir = os.path.join(args.out_dir, ".smote_enn_checkpoint")

    if args.force:
        for p in [results_csv, ckpt_path, audit_path]:
            if os.path.exists(p):
                os.remove(p)
        for f in os.listdir(args.out_dir):
            if f.startswith("test_predictions_") and f.endswith(".parquet"):
                os.remove(os.path.join(args.out_dir, f))
        if os.path.exists(smote_enn_ckpt_dir):
            import shutil
            shutil.rmtree(smote_enn_ckpt_dir)
        log("--force: previous results cleared", logp)

    with open(os.path.join(args.data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    expected_fp = manifest["test_fingerprint_sha256"]

    log("loading frozen test set (read-only)", logp)
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))
    fp = fingerprint(X_test, y_test)
    assert fp == expected_fp, "TEST FINGERPRINT MISMATCH AT STARTUP - aborting"
    log(f"test fingerprint verified: {fp[:16]}... n={len(y_test)} {class_counts(y_test)}", logp)

    with open(env_path, "w") as f:
        json.dump(environment_record(), f, indent=2)

    ckpt = load_checkpoint(ckpt_path)
    ckpt["test_fingerprint"] = fp
    ckpt["versions"] = environment_record()
    save_checkpoint(ckpt_path, ckpt)

    done = completed_pairs(results_csv)
    log(f"resuming: {len(done)}/12 pairs already complete", logp)
    audit = {}
    if os.path.exists(audit_path):
        with open(audit_path) as f:
            audit = json.load(f)

    techniques = [t for t in TECHNIQUES if args.only_technique is None or t == args.only_technique]

    for tech in techniques:
        models_todo = [m for m in MODELS if (tech, m) not in done]
        if not models_todo:
            log(f"[{tech}] all models complete - skipping", logp)
            continue

        log(f"[{tech}] loading training data", logp)
        X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
        n_before = len(y_train)
        cc_before = class_counts(y_train)

        # ---- resample (TRAIN ONLY) ----
        if tech == "SMOTE-ENN":
            # Use staged pipeline for SMOTE-ENN
            log(f"[{tech}] Using staged two-stage pipeline (SMOTE → ENN)", logp)
            try:
                Xr, yr, staged_metadata = run_smote_enn_staged(
                    X_train, y_train, 
                    checkpoint_dir=smote_enn_ckpt_dir,
                    logp=logp
                )
                n_after = len(yr)
                cc_after = class_counts(yr)
                resample_sec = staged_metadata["total_elapsed_sec"]
                log(f"[{tech}] staged pipeline complete: {n_before} -> {n_after} {cc_after}", logp)
            except Exception as e:  # noqa: BLE001
                audit[tech] = {
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                    "train_before": {"n": n_before, "class_counts": cc_before},
                    "note": "SMOTE checkpoint preserved in .smote_enn_checkpoint for resume"
                }
                with open(audit_path, "w") as f:
                    json.dump(audit, f, indent=2)
                log(f"[{tech}] STAGED PIPELINE FAILED: {e} - checkpoint preserved", logp)
                del X_train, y_train
                gc.collect()
                continue
        else:
            # Standard resampling for other techniques
            resampler = make_resampler(tech)
            log(f"[{tech}] resampling with {type(resampler).__name__} ...", logp)
            t0 = time.perf_counter()
            try:
                Xr, yr = resampler.fit_resample(X_train, y_train)
            except Exception as e:  # noqa: BLE001
                audit[tech] = {
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                    "train_before": {"n": n_before, "class_counts": cc_before}
                }
                with open(audit_path, "w") as f:
                    json.dump(audit, f, indent=2)
                log(f"[{tech}] RESAMPLING FAILED: {e} - recorded in audit, continuing", logp)
                del X_train, y_train
                gc.collect()
                continue
            resample_sec = time.perf_counter() - t0
            n_after = len(yr)
            cc_after = class_counts(yr)
            log(f"[{tech}] resampled in {resample_sec:.1f}s: {n_before} -> {n_after} {cc_after}", logp)

        smote_involved = tech in ("SMOTE-Tomek", "SMOTE-ENN")
        validity = audit_validity(X_train, Xr, None, smote_involved)
        log(f"[{tech}] validity: {validity}", logp)
        assert validity["finite"] and validity["feature_count_ok"], f"[{tech}] validity failed"

        # test set untouched?
        assert fingerprint(X_test, y_test) == expected_fp, "TEST SET CHANGED DURING RESAMPLING"

        # ---- train models sequentially ----
        models_cfg = make_models()
        preds = {"y_true": y_test}
        tech_audit_models = audit.get(tech, {}).get("models", {})
        for m in models_todo:
            scaling, est = models_cfg[m]
            log(f"[{tech}][{m}] training ...", logp)
            try:
                metrics, dt, y_pred, y_proba = run_model(m, scaling, est, Xr, yr, X_test, y_test)
            except Exception as e:  # noqa: BLE001
                tech_audit_models[m] = {
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()
                }
                log(f"[{tech}][{m}] TRAINING FAILED: {e} - recorded, continuing", logp)
                continue
            row = format_result_row(tech, m, metrics, dt)
            new_file = not os.path.exists(results_csv)
            with open(results_csv, "a") as f:
                if new_file:
                    f.write(RESULTS_HEADER + "\n")
                f.write(row + "\n")
            done.add((tech, m))
            ckpt["completed"][f"{tech}|{m}"] = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            save_checkpoint(ckpt_path, ckpt)
            preds[f"{m}__pred"] = y_pred.astype(np.int64)
            preds[f"{m}__proba"] = y_proba.astype(np.float64)
            tech_audit_models[m] = {"fit_predict_sec": round(dt, 1)}
            log(f"[{tech}][{m}] done in {dt:.1f}s: "
                f"acc={metrics['Accuracy']:.4f} f1={metrics['F1']:.4f}", logp)
            del y_pred, y_proba
            gc.collect()

        # ---- predictions parquet (atomic write) ----
        if len(preds) > 1:
            slug = TECHNIQUE_SLUGS[tech]
            final_p = os.path.join(args.out_dir, f"test_predictions_{slug}.parquet")
            tmp_p = final_p + ".tmp"
            pd.DataFrame(preds).to_parquet(tmp_p, index=False)
            os.replace(tmp_p, final_p)
            log(f"[{tech}] predictions parquet written: {final_p}", logp)

        # ---- audit entry ----
        resampler_name = "SMOTEENN (staged: SMOTE → ENN)" if tech == "SMOTE-ENN" else type(resampler).__name__
        audit[tech] = {
            "technique": tech,
            "resampler": resampler_name,
            "resampler_params": describe_resampler(tech) if tech != "SMOTE-ENN" else "Person 1 SMOTE (k_neighbors=5) + ENN (n_neighbors=3, staged pipeline)",
            "train_before": {"n": n_before, "class_counts": cc_before},
            "train_after": {"n": n_after, "class_counts": cc_after},
            "rows_removed": max(0, n_before - n_after),
            "rows_added": max(0, n_after - n_before),
            "final_train_size": n_after,
            "resample_sec": round(resample_sec, 1),
            "validity": validity,
            "models": tech_audit_models,
            "test_fingerprint_ok": bool(fingerprint(X_test, y_test) == expected_fp),
            "errors": [],
        }
        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=2)

        # ---- class_distributions.json (Person 1 format) ----
        cd = {}
        if os.path.exists(cd_path):
            with open(cd_path) as f:
                cd = json.load(f)
        cd[tech] = {
            "technique": tech,
            "train_before": cc_before,
            "train_after": cc_after,
            "added_samples": max(0, n_after - n_before),
            "removed_samples": max(0, n_before - n_after),
            "final_train_size": n_after,
        }
        with open(cd_path, "w") as f:
            json.dump(cd, f, indent=2)

        # ---- release resampled data before next technique ----
        del X_train, y_train, Xr, yr, preds
        gc.collect()
        log(f"[{tech}] technique complete; memory released", logp)

    assert fingerprint(X_test, y_test) == expected_fp, "TEST FINGERPRINT MISMATCH AT END"
    log(f"run finished: {len(completed_pairs(results_csv))}/12 pairs complete; "
        f"test fingerprint intact", logp)


if __name__ == "__main__":
    main()

