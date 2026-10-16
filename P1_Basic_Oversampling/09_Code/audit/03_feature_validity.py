"""
AUDIT 2 - SMOTE/ADASYN synthetic-feature validity audit.
Regenerates SMOTE and ADASYN resampling deterministically (seed 42) on
  (A) the ORIGINAL scope (full master, incl. 2014) and
  (B) the CORRECTED 2015-2024 scope (YEAR==2014 excluded).
Verification of reproduction:
  (A) resampled class counts must equal the saved 05_SMOTE / 06_ADASYN
      class_distribution.json EXACTLY; and X_res[:n_orig] must equal X_train
      EXACTLY, so that rows X_res[n_orig:] are precisely the synthetic samples.
  (B) resampled class counts must equal the audit-rerun JSONs written by
      02_rerun_2015_2024.py (waits for them), and X_train must match the rerun's
      saved X_train parquet.
Validity rule (STRICT, per audit spec): a synthetic value is INVALID if it is
non-integer OR not in the feature's observed valid code set. Being within
[min, max] is NOT sufficient (e.g. 2.5 is invalid for {1,2,3,4}).
No rounding, clipping, or algorithm changes. This script only measures.
"""
import json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN

MASTER = "/home/hatch/workspace/brfss_data/brfss_master.parquet"
BASE = "/home/hatch/workspace/Person_1_Class_Imbalance"
VOUT = f"{BASE}/12_Audit_Feature_Validity"
RNG = 42
FEATURES = ["EXERANY2", "_MICHD", "ASTHMA3", "ADDEPEV3", "CHCKDNY2",
            "MARITAL", "_EDUCAG", "DRNKANY6", "_AGEG5YR", "_BMI5CAT",
            "_INCOMG1", "_RFSMOK3", "GENHLTH", "HAVARTH4", "SEXVAR"]
TARGET = "DIABETE4"


def build_split(df):
    X = df[FEATURES].astype(int)
    y = (df[TARGET] == "Yes").astype(int)
    return train_test_split(X, y, test_size=0.20, random_state=RNG,
                            stratify=y, shuffle=True)


def validity_audit(X_train, X_syn, tech, scope, outdir):
    """X_train: int DataFrame; X_syn: float ndarray of synthetic rows only."""
    os.makedirs(outdir, exist_ok=True)
    n_syn = X_syn.shape[0]
    per_feat, any_invalid = [], np.zeros(n_syn, dtype=bool)
    for j, c in enumerate(FEATURES):
        valid = np.sort(X_train[c].unique())
        v = X_syn[:, j].astype(float)
        is_int = np.isclose(v, np.round(v))
        vr = np.round(v).astype(np.int64)
        in_set = np.isin(vr, valid)
        invalid = (~is_int) | (~in_set)
        any_invalid |= invalid
        per_feat.append({
            "feature": c,
            "valid_codes": ",".join(map(str, valid.tolist())),
            "n_valid_codes": int(len(valid)),
            "n_synthetic": int(n_syn),
            "n_invalid": int(invalid.sum()),
            "pct_invalid": round(float(invalid.sum()) / n_syn * 100, 4),
            "n_fractional_non_integer": int((~is_int).sum()),
            "n_integer_out_of_set": int((is_int & ~in_set).sum()),
            "n_out_of_min_max_range": int((((v < valid.min()) | (v > valid.max()))).sum()),
        })
    pct_rows_any = round(float(any_invalid.sum()) / n_syn * 100, 4)
    pf = pd.DataFrame(per_feat)
    pf.to_csv(f"{outdir}/{tech}_{scope}_per_feature_validity.csv", index=False)
    summary = {
        "technique": tech, "scope": scope,
        "n_synthetic_rows": int(n_syn),
        "n_rows_with_at_least_one_invalid_feature": int(any_invalid.sum()),
        "pct_rows_with_at_least_one_invalid_feature": pct_rows_any,
        "features_with_any_invalid": pf.loc[pf["n_invalid"] > 0, "feature"].tolist(),
        "features_fully_valid": pf.loc[pf["n_invalid"] == 0, "feature"].tolist(),
        "worst_feature": pf.loc[pf["pct_invalid"].idxmax(), "feature"],
        "worst_feature_pct_invalid": float(pf["pct_invalid"].max()),
    }
    with open(f"{outdir}/{tech}_{scope}_validity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # persist synthetic samples for the thesis appendix (float64, exact)
    pd.DataFrame(X_syn, columns=FEATURES).to_parquet(
        f"{outdir}/{tech}_{scope}_synthetic_samples.parquet")
    print(f"[{tech} | {scope}] synthetic rows={n_syn:,}; rows with >=1 invalid "
          f"feature value: {any_invalid.sum():,} ({pct_rows_any}%)", flush=True)
    print(pf[["feature", "valid_codes", "n_invalid", "pct_invalid",
              "n_fractional_non_integer", "n_integer_out_of_set",
              "n_out_of_min_max_range"]].to_string(index=False), flush=True)
    return summary


def resample_and_verify(X_train, y_train, sampler, expected_after, expected_added, tech, scope):
    t = time.time()
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    n_orig = len(X_train)
    # --- reproduction assertions ---
    assert np.array_equal(np.asarray(X_res[:n_orig]), X_train.to_numpy()), \
        f"{tech}/{scope}: X_res[:n_orig] != X_train -> synthetic rows not isolated"
    vc = pd.Series(y_res).value_counts().to_dict()
    got_after = {int(k): int(v) for k, v in vc.items()}
    assert got_after == expected_after, \
        f"{tech}/{scope}: class counts {got_after} != expected {expected_after}"
    assert len(X_res) - n_orig == expected_added, \
        f"{tech}/{scope}: added {len(X_res)-n_orig} != expected {expected_added}"
    print(f"[{tech} | {scope}] REPRODUCTION PASS: class counts {got_after}, "
          f"added {expected_added:,}, resample took {time.time()-t:.1f}s", flush=True)
    return np.asarray(X_res[n_orig:]).astype(float)


print("Loading master (read-only)...", flush=True)
df = pd.read_parquet(MASTER)
assert len(df) == 2498203

# ================= PART A: ORIGINAL SCOPE =================
print("\n=== PART A: original scope (2014-2024) ===", flush=True)
Xtr_o, _, ytr_o, _ = build_split(df)
n_o = len(Xtr_o)
with open(f"{BASE}/05_SMOTE/class_distribution.json") as f:
    d_sm = json.load(f)
with open(f"{BASE}/06_ADASYN/class_distribution.json") as f:
    d_ad = json.load(f)

syn_sm_o = resample_and_verify(
    Xtr_o, ytr_o, SMOTE(sampling_strategy="auto", random_state=RNG, k_neighbors=5),
    {int(k): int(v) for k, v in d_sm["train_after"].items()},
    d_sm["added_samples"], "SMOTE", "original_2014_2024")
syn_ad_o = resample_and_verify(
    Xtr_o, ytr_o, ADASYN(sampling_strategy="auto", random_state=RNG, n_neighbors=5),
    {int(k): int(v) for k, v in d_ad["train_after"].items()},
    d_ad["added_samples"], "ADASYN", "original_2014_2024")

summaries = {"original_2014_2024": {}}
summaries["original_2014_2024"]["SMOTE"] = validity_audit(
    Xtr_o, syn_sm_o, "SMOTE", "original_2014_2024", f"{VOUT}/original_scope")
del syn_sm_o
summaries["original_2014_2024"]["ADASYN"] = validity_audit(
    Xtr_o, syn_ad_o, "ADASYN", "original_2014_2024", f"{VOUT}/original_scope")
del syn_ad_o, Xtr_o, ytr_o

# ================= PART B: CORRECTED SCOPE =================
print("\n=== PART B: corrected scope (2015-2024) ===", flush=True)
RERUN = f"{BASE}/11_Audit_2015_2024_Rerun"
need = [f"{RERUN}/05_smote_audit/class_distribution_2015_2024.json",
        f"{RERUN}/06_adasyn_audit/class_distribution_2015_2024.json"]
t0 = time.time()
while not all(os.path.exists(p) for p in need):
    if time.time() - t0 > 7200:
        raise TimeoutError("timed out waiting for corrected-scope rerun results")
    print(f"  waiting for corrected-scope rerun results... "
          f"({int(time.time()-t0)}s elapsed)", flush=True)
    time.sleep(60)
print("  corrected-scope rerun results found.", flush=True)

dfc = df.loc[df["YEAR"] != 2014].copy()
Xtr_c, _, ytr_c, _ = build_split(dfc)
# verify our recomputed corrected split matches the rerun's saved split
Xtr_saved = pd.read_parquet(f"{RERUN}/02_preprocessing_audit/X_train.parquet")
ytr_saved = pd.read_parquet(f"{RERUN}/02_preprocessing_audit/y_train.parquet")["y"]
assert set(Xtr_c.index) == set(Xtr_saved.index), "corrected X_train indices differ from rerun"
assert np.array_equal(Xtr_c.to_numpy(), Xtr_saved.to_numpy()), "corrected X_train values differ"
print("  corrected-scope split matches rerun's saved X_train (indices + values).", flush=True)

with open(need[0]) as f:
    d_sm_c = json.load(f)
with open(need[1]) as f:
    d_ad_c = json.load(f)
syn_sm_c = resample_and_verify(
    Xtr_c, ytr_c, SMOTE(sampling_strategy="auto", random_state=RNG, k_neighbors=5),
    {int(k): int(v) for k, v in d_sm_c["train_after"].items()},
    d_sm_c["added_samples"], "SMOTE", "corrected_2015_2024")
syn_ad_c = resample_and_verify(
    Xtr_c, ytr_c, ADASYN(sampling_strategy="auto", random_state=RNG, n_neighbors=5),
    {int(k): int(v) for k, v in d_ad_c["train_after"].items()},
    d_ad_c["added_samples"], "ADASYN", "corrected_2015_2024")

summaries["corrected_2015_2024"] = {}
summaries["corrected_2015_2024"]["SMOTE"] = validity_audit(
    Xtr_c, syn_sm_c, "SMOTE", "corrected_2015_2024", f"{VOUT}/corrected_scope")
del syn_sm_c
summaries["corrected_2015_2024"]["ADASYN"] = validity_audit(
    Xtr_c, syn_ad_c, "ADASYN", "corrected_2015_2024", f"{VOUT}/corrected_scope")
del syn_ad_c, Xtr_c, ytr_c

with open(f"{VOUT}/validity_summaries.json", "w") as f:
    json.dump(summaries, f, indent=2)
print("\nFEATURE-VALIDITY AUDIT COMPLETE.", flush=True)
print(json.dumps({s: {t: summaries[s][t]["pct_rows_with_at_least_one_invalid_feature"]
                                   for t in summaries[s]} for s in summaries}, indent=2), flush=True)
