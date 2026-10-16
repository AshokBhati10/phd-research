"""Robust two-stage SMOTE-ENN pipeline with checkpointing and monitoring.

This module implements SMOTE-ENN as an explicit two-stage pipeline:
  Stage 1: SMOTE (oversampling)
  Stage 2: ENN (undersampling)

Checkpointing allows resume if Stage 2 fails (SMOTE output can be reused).
Monitoring logs provide visibility into each stage's progress and resource usage.

Scientific definition preserved:
  - SMOTE component: Person 1 exact configuration
  - ENN component: EditedNearestNeighbours with n_neighbors=3 (library default)
  - Both stages run sequentially on training data only
  - Test set never modified
"""
import gc
import json
import os
import time
import traceback
from datetime import datetime, timezone

import numpy as np

from common import RNG


def log_staged(msg, logp=None):
    """Log with ISO timestamp, matching common.log() style."""
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    if logp:
        with open(logp, "a") as f:
            f.write(line + "\n")


def class_counts_simple(y):
    """Return class distribution as dict."""
    y = np.asarray(y)
    return {str(int(c)): int((y == c).sum()) for c in np.unique(y)}


def stage1_smote(X_train, y_train, checkpoint_dir=None, logp=None):
    """
    Stage 1: SMOTE oversampling.
    
    Returns: (X_smote, y_smote, smote_metadata_dict)
    
    If checkpoint_dir provided, saves SMOTE output and metadata for resume.
    """
    from imblearn.over_sampling import SMOTE
    
    smote_checkpoint_path = None
    smote_metadata_path = None
    if checkpoint_dir:
        smote_checkpoint_path = os.path.join(checkpoint_dir, "smote_X.npy")
        smote_metadata_path = os.path.join(checkpoint_dir, "smote_metadata.json")
        if os.path.exists(smote_checkpoint_path):
            log_staged("[SMOTE] Loading cached SMOTE output from checkpoint", logp)
            X_smote = np.load(smote_checkpoint_path, allow_pickle=False)
            y_smote = np.load(os.path.join(checkpoint_dir, "smote_y.npy"), allow_pickle=False)
            with open(smote_metadata_path) as f:
                metadata = json.load(f)
            log_staged(f"[SMOTE] Loaded from cache: {X_smote.shape[0]} rows, {class_counts_simple(y_smote)}", logp)
            return X_smote, y_smote, metadata
    
    n_before = len(y_train)
    cc_before = class_counts_simple(y_train)
    
    log_staged(f"[SMOTE] Input: {n_before:,} rows, class dist: {cc_before}", logp)
    log_staged("[SMOTE] Starting SMOTE (Person 1 configuration: sampling_strategy='auto', k_neighbors=5)", logp)
    
    smote = SMOTE(sampling_strategy="auto", random_state=RNG, k_neighbors=5)
    t0 = time.perf_counter()
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    smote_sec = time.perf_counter() - t0
    
    n_after = len(y_smote)
    cc_after = class_counts_simple(y_smote)
    n_added = n_after - n_before
    
    log_staged(f"[SMOTE] Completed in {smote_sec:.1f}s", logp)
    log_staged(f"[SMOTE] Output: {n_after:,} rows, class dist: {cc_after}", logp)
    log_staged(f"[SMOTE] Samples added: {n_added:,}", logp)
    
    metadata = {
        "stage": "SMOTE",
        "n_before": n_before,
        "n_after": n_after,
        "class_counts_before": cc_before,
        "class_counts_after": cc_after,
        "samples_added": n_added,
        "elapsed_sec": round(smote_sec, 1),
    }
    
    # Save checkpoint if requested
    if checkpoint_dir:
        log_staged(f"[SMOTE] Saving checkpoint to {checkpoint_dir}", logp)
        np.save(smote_checkpoint_path, X_smote, allow_pickle=False)
        np.save(os.path.join(checkpoint_dir, "smote_y.npy"), y_smote, allow_pickle=False)
        with open(smote_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        log_staged("[SMOTE] Checkpoint saved", logp)
    
    return X_smote, y_smote, metadata


def stage2_enn(X_smote, y_smote, checkpoint_dir=None, logp=None):
    """
    Stage 2: ENN undersampling on SMOTE-expanded data.
    
    Returns: (X_final, y_final, enn_metadata_dict)
    """
    from imblearn.under_sampling import EditedNearestNeighbours
    
    n_before = len(y_smote)
    cc_before = class_counts_simple(y_smote)
    
    log_staged(f"[ENN] Input: {n_before:,} rows, class dist: {cc_before}", logp)
    log_staged("[ENN] Starting EditedNearestNeighbours (sampling_strategy='auto', n_neighbors=3)", logp)
    
    enn = EditedNearestNeighbours(sampling_strategy="auto", n_neighbors=3)
    
    t0 = time.perf_counter()
    X_final, y_final = enn.fit_resample(X_smote, y_smote)
    enn_sec = time.perf_counter() - t0
    
    n_after = len(y_final)
    cc_after = class_counts_simple(y_final)
    n_removed = n_before - n_after
    
    log_staged(f"[ENN] Completed in {enn_sec:.1f}s", logp)
    log_staged(f"[ENN] Output: {n_after:,} rows, class dist: {cc_after}", logp)
    log_staged(f"[ENN] Samples removed: {n_removed:,}", logp)
    
    metadata = {
        "stage": "ENN",
        "n_before": n_before,
        "n_after": n_after,
        "class_counts_before": cc_before,
        "class_counts_after": cc_after,
        "samples_removed": n_removed,
        "elapsed_sec": round(enn_sec, 1),
    }
    
    return X_final, y_final, metadata


def run_smote_enn_staged(X_train, y_train, checkpoint_dir=None, logp=None):
    """
    Execute full SMOTE-ENN pipeline with checkpointing.
    
    Returns: (X_final, y_final, full_metadata_dict)
    
    If checkpoint_dir provided:
      - SMOTE output is cached; if present on resume, SMOTE is skipped
      - Pipeline can resume from this point if ENN fails
    """
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Stage 1: SMOTE
    try:
        X_smote, y_smote, smote_meta = stage1_smote(X_train, y_train, checkpoint_dir, logp)
    except Exception as e:
        log_staged(f"[SMOTE-ENN] STAGE 1 (SMOTE) FAILED: {type(e).__name__}: {e}", logp)
        log_staged(f"[SMOTE-ENN] Traceback:\n{traceback.format_exc()}", logp)
        raise
    
    # Clear original training data from memory before Stage 2
    del X_train, y_train
    gc.collect()
    log_staged("[SMOTE-ENN] Stage 1 complete, original training data released", logp)
    
    # Stage 2: ENN
    try:
        X_final, y_final, enn_meta = stage2_enn(X_smote, y_smote, checkpoint_dir, logp)
    except Exception as e:
        log_staged(f"[SMOTE-ENN] STAGE 2 (ENN) FAILED: {type(e).__name__}: {e}", logp)
        log_staged(f"[SMOTE-ENN] Traceback:\n{traceback.format_exc()}", logp)
        log_staged(f"[SMOTE-ENN] SMOTE checkpoint preserved at {checkpoint_dir} for resume", logp)
        raise
    
    # Clear SMOTE output from memory before returning
    del X_smote, y_smote
    gc.collect()
    log_staged("[SMOTE-ENN] Stage 2 complete, SMOTE intermediate data released", logp)
    
    # Full metadata - fixed to avoid walrus operator in dict literal
    n_after = len(y_final)
    cc_after = class_counts_simple(y_final)
    
    metadata = {
        "technique": "SMOTE-ENN",
        "stage1_smote": smote_meta,
        "stage2_enn": enn_meta,
        "total_elapsed_sec": round(smote_meta["elapsed_sec"] + enn_meta["elapsed_sec"], 1),
        "final_rows": n_after,
        "final_class_counts": cc_after,
    }
    
    log_staged(f"[SMOTE-ENN] Pipeline complete: {n_after:,} final rows, {cc_after}", logp)
    
    return X_final, y_final, metadata

