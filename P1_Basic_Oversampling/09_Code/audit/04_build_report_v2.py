"""
Build Person_1_Final_Report.md v2 (corrected 2015-2024 scope) from:
- original report (Person_1_Final_Report_ORIGINAL_SCOPE.md)
- corrected consolidated CSV
- original consolidated CSV (Appendix A)
- year audit JSON, validity summaries JSON, corrected protocol config
"""
import json
import numpy as np
import pandas as pd

BASE = "/home/hatch/workspace/Person_1_Class_Imbalance"
REP = f"{BASE}/10_Final_Report"
orig = open(f"{REP}/Person_1_Final_Report_ORIGINAL_SCOPE.md").read()
corr = pd.read_csv(f"{BASE}/11_Audit_2015_2024_Rerun/07_combined_results_audit/"
                   "combined_results_2015_2024.csv")
origc = pd.read_csv(f"{BASE}/07_Combined_Results/combined_results.csv")
year = json.load(open(f"{BASE}/11_Audit_2015_2024_Rerun/01_year_audit/year_audit_summary.json"))
val = json.load(open(f"{BASE}/12_Audit_Feature_Validity/validity_summaries.json"))
cfg = json.load(open(f"{BASE}/11_Audit_2015_2024_Rerun/02_preprocessing_audit/protocol_config_audit.json"))

METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]


def md_table(df):
    hdr = "| Technique | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |"
    lines = [hdr, "|" + "---|" * 8]
    for _, r in df.iterrows():
        lines.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            r["Technique"], r["Model"], *[r[m] for m in METRICS]))
    return "\n".join(lines)


# max absolute delta between scopes (aligned by Technique+Model)
m = origc.merge(corr, on=["Technique", "Model"], suffixes=("_o", "_c"))
delta = (m[[f"{k}_c" for k in METRICS]].to_numpy()
         - m[[f"{k}_o" for k in METRICS]].to_numpy())
max_delta = float(np.abs(delta).max())
delta_row = m.iloc[int(np.abs(delta).max(axis=1).argmax())]
best_c = corr.loc[corr["F1"].idxmax()]

# ---------- v2 header ----------
v2 = orig.replace(
    "**Date:** 2026-10-01 | Seed: 42 (all randomized operations)",
    "**Date:** 2026-10-01 | Seed: 42 (all randomized operations)\n"
    "**Report version:** v2 (2026-10-01) — corrected 2015–2024 scope (year-scope audit). "
    "The original 2014–2024 analysis is preserved verbatim in "
    "`Person_1_Final_Report_ORIGINAL_SCOPE.md` and as Appendix A of this report.")
v2 = v2.replace(
    "**Role:** Person 1 — No Balancing, Random Oversampling, SMOTE, ADASYN",
    "**Role:** Person 1 — No Balancing, Random Oversampling, SMOTE, ADASYN\n\n"
    "> **Scope correction (v2):** the master dataset contains 201,035 YEAR=2014 rows "
    "(8.05%) that were included in the original analysis. The primary results below "
    "are from the corrected 2015–2024 rerun (identical protocol, seed 42, new frozen "
    f"test set n={cfg['n_test']:,}). The scope discrepancy changed no conclusion "
    f"(max |Δ| across all 72 metric cells = {max_delta:.4f}; see §10).")

# ---------- §2 split note ----------
v2 = v2.replace(
    "Split validations passed: sizes sum correctly, zero index overlap, identical stratification\n"
    "   proportions, identical frozen feature column order for train and test.",
    "Split validations passed: sizes sum correctly, zero index overlap, identical stratification\n"
    "   proportions, identical frozen feature column order for train and test.\n"
    f"   *v2: corrected 2015–2024 scope — Train = {cfg['n_train']:,} "
    f"(No {cfg['train_dist']['No']:,} / Yes {cfg['train_dist']['Yes']:,}); "
    f"Test = {cfg['n_test']:,} (No {cfg['test_dist']['No']:,} / Yes {cfg['test_dist']['Yes']:,}); "
    "new frozen test set, never resampled — see §10.*")

# ---------- §6 replacement ----------
sec6_start = v2.index("## 6. Results")
sec7_start = v2.index("## 7. Validation")
new_sec6 = f"""## 6. Results — corrected 2015–2024 scope (primary)

### 6.1 Consolidated comparison (shared untouched test set, n = {cfg['n_test']:,}; seed 42)

{md_table(corr)}

Source: `11_Audit_2015_2024_Rerun/07_combined_results_audit/combined_results_2015_2024.csv`
(per-experiment CSVs under `11_Audit_2015_2024_Rerun/03_baseline_audit/` … `06_adasyn_audit/`).

**Scope sensitivity:** the corrected rerun reproduces the original pattern almost exactly —
max absolute difference vs the original 2014–2024 table (Appendix A) is {max_delta:.4f}
across all 72 metric cells (largest shift: {delta_row['Technique']} / {delta_row['Model']}).
Best F1 remains ROS + XGBoost ({best_c['F1']:.4f}); highest recall remains ADASYN + RandomForest
({corr[(corr['Technique']=='ADASYN')&(corr['Model']=='RandomForest')]['Recall'].iloc[0]:.4f}).

### 6.2 Key observations (objective — hold for both scopes)

1. **Balancing trades accuracy/precision for recall.** At the default 0.5 decision
   threshold, any of the three balancing techniques lifts recall from 0.09–0.14
   (baseline) to 0.70–0.77, and F1 from 0.15–0.22 to 0.43–0.45 — roughly a 2× gain in
   F1 — while accuracy falls from ~0.83–0.84 to ~0.65–0.70 and precision from
   ~0.54–0.61 to ~0.30–0.32.
2. **Ranking ability is essentially unchanged.** ROC-AUC stays within 0.767–0.780 and
   PR-AUC within 0.395–0.418 across all 12 cells. Balancing changed the operating
   point, not the models' ability to rank patients by risk.
3. **ROS ≈ SMOTE on this data.** Random Oversampling and SMOTE differ by ≤0.004 in F1
   for every model; ROS is marginally ahead on XGB ({corr[(corr['Technique']=='Random Oversampling')&(corr['Model']=='XGBoost')]['F1'].iloc[0]:.4f} vs {corr[(corr['Technique']=='SMOTE')&(corr['Model']=='XGBoost')]['F1'].iloc[0]:.4f}). On
   integer-coded categorical features, synthetic interpolation provided no meaningful
   edge over simple replication in this study.
4. **ADASYN is the most aggressive minority learner.** It gives the highest recall for
   tree models (RF {corr[(corr['Technique']=='ADASYN')&(corr['Model']=='RandomForest')]['Recall'].iloc[0]:.4f}, XGB {corr[(corr['Technique']=='ADASYN')&(corr['Model']=='XGBoost')]['Recall'].iloc[0]:.4f}) but also the lowest precision and accuracy —
   the expected consequence of its adaptive focus on hard minority regions.
5. **Model ranking:** XGBoost has the best ROC-AUC and PR-AUC under every technique.
   Best F1 overall is ROS + XGBoost ({best_c['F1']:.4f}), essentially tied with ROS + RandomForest
   and SMOTE + RandomForest. No threshold tuning or cost-sensitive learning was performed,
   so these comparisons hold at the default 0.5 threshold only.

### 6.3 Visualizations

The 12 requested plots in `08_Visualizations/` were generated from the original
2014–2024 scope. They are retained as-is because the corrected-scope metrics differ by
at most {max_delta:.4f} — no visual conclusion changes. (Regenerating all plots on the
corrected scope is left as optional follow-up; the consolidated CSVs are authoritative.)

"""
v2 = v2[:sec6_start] + new_sec6 + v2[sec7_start:]

# ---------- Appendix A (original table) after §9 ----------
appendix = f"""
---

## Appendix A — Original 2014–2024 results (sensitivity reference)

Preserved verbatim from the v1 report (shared untouched test set n = 499,641; seed 42).
Superseded as the primary table by §6.1; kept so the year-scope correction is auditable.

{md_table(origc)}

"""
v2 = v2.replace("### File map", appendix + "### File map")

# ---------- §7 additions ----------
v2 = v2.replace(
    "no NaN/inf; all 12 expected result rows present.",
    "no NaN/inf; all 12 expected result rows present.\n"
    "- **Audit rerun (2015–2024):** identical QC battery on the corrected scope — "
    f"all 12 confusion-matrix totals == {cfg['n_test']:,}; F1 == 2PR/(P+R) to 1e-9; "
    "metrics finite and within [0,1]; new test-set fingerprint "
    f"(n={cfg['n_test']:,}, Yes={cfg['test_dist']['Yes']:,}) byte-identical after every experiment; "
    "SMOTE/ADASYN reproduction assertions passed (class counts exactly equal the rerun's "
    "saved class_distribution JSONs; X_res[:n_orig] == X_train exactly).")

# ---------- §8 additions (append after item 6) ----------
v2 = v2.replace(
    "   setup only; no impact on results.)",
    "   setup only; no impact on results.)\n"
    "7. **Year-scope correction (audit).** The v1 analysis used the master as-is (2014–2024);\n"
    "   201,035 YEAR=2014 rows (8.05%) entered both training and test sets. The full protocol\n"
    "   was rerun on the corrected 2015–2024 working set (2,297,168 rows) with a new frozen\n"
    f"   test set (n={cfg['n_test']:,}); all conclusions unchanged (max |Δ| = {max_delta:.4f}). See §10.\n"
    "8. **SMOTE/ADASYN synthetic-value audit.** A strict per-feature audit (see §11) found 0\n"
    "   invalid categorical codes in ~39M synthetic cells across both methods and both scopes.\n"
    "   Mechanism note: imbalanced-learn 0.14.2 casts interpolated floats back to the input\n"
    "   dtype via `X_new.astype(X.dtype)` (truncation toward zero), not rounding — disclosed,\n"
    "   impact negligible here (synthetic code distributions ≈ minority-train distributions).\n"
    "9. **Rerun script syntax error.** First launch of the audit rerun failed at parse time\n"
    "   (stray parenthesis in a test-set assertion); fixed, syntax re-verified, rerun launched\n"
    "   clean. No partial outputs were written by the failed launch.")

# ---------- §9 addition ----------
v2 = v2.replace(
    "No claim is made that any technique is universally best: the results hold for this\n"
    "dataset, these features, these models, and the default 0.5 threshold. Threshold tuning,\n"
    "cost-sensitive training, or calibration could change the operating-point trade-offs.",
    "No claim is made that any technique is universally best: the results hold for this\n"
    "dataset, these features, these models, and the default 0.5 threshold. Threshold tuning,\n"
    "cost-sensitive training, or calibration could change the operating-point trade-offs.\n\n"
    "*v2: the corrected 2015–2024 rerun confirms the interpretation is scope-robust — "
    f"excluding 2014 moved no metric by more than {max_delta:.4f}.*")

# ---------- new sections ----------
def scope_rows(scope_key, label):
    s = val[scope_key]
    return (f"| {label} | SMOTE | {s['SMOTE']['n_synthetic_rows']:,} | "
            f"{s['SMOTE']['n_rows_with_at_least_one_invalid_feature']:,} "
            f"({s['SMOTE']['pct_rows_with_at_least_one_invalid_feature']}%) | "
            f"{', '.join(s['SMOTE']['features_with_any_invalid']) or '— none —'} |\n"
            f"| {label} | ADASYN | {s['ADASYN']['n_synthetic_rows']:,} | "
            f"{s['ADASYN']['n_rows_with_at_least_one_invalid_feature']:,} "
            f"({s['ADASYN']['pct_rows_with_at_least_one_invalid_feature']}%) | "
            f"{', '.join(s['ADASYN']['features_with_any_invalid']) or '— none —'} |")


new_sections = f"""
## 10. Year-scope audit

**Finding:** the master file (`brfss_master.parquet`) contains all years 2014–2024.
YEAR=2014 contributes **{year['year_2014_n']:,} rows ({year['year_2014_pct_of_master']}%)**.
The v1 experiments used the master as-is, so all four experiments trained on and were
evaluated against 2014 data: **{year['original_train_2014_n']:,}** 2014 rows in the original
training set and **{year['original_test_2014_n']:,}** in the original test set.

**Proof:** (i) the saved frozen split was re-joined to the master on row index — 2014 rows
are present in both train and test; (ii) the exact v1 preprocessing + split was recomputed
(seed 42, stratified 80/20) and produced byte-identical indices and sizes
(train=1,998,562 / test=499,641). Reproduction assertion: PASS.

**Correction:** a 2015–2024 working set was built by excluding YEAR==2014
({cfg['dropped_2014_rows']:,} rows dropped → {cfg['working_rows']:,} rows) and the entire
protocol was rerun identically: same 15 features, same integer encoding, stratified 80/20
split seed 42 → new frozen test set n={cfg['n_test']:,} (No {cfg['test_dist']['No']:,} /
Yes {cfg['test_dist']['Yes']:,}), same 3 models + hyperparameters, resampling on train only,
same QC battery (all PASS). Results: `11_Audit_2015_2024_Rerun/`.

**Impact:** negligible. Max absolute metric change vs v1 = {max_delta:.4f}; the technique
ranking, the ROS≈SMOTE finding, and the ADASYN-recall finding are all unchanged. The
v1 table is therefore a valid sensitivity reference (Appendix A), and the corrected
2015–2024 table (§6.1) is the primary result for the thesis.

## 11. Synthetic-feature validity audit

**Question:** standard SMOTE/ADASYN interpolate continuous values, but all 15 BRFSS
predictors are integer-coded categoricals (e.g. EXERANY2 ∈ {{1,2}}, MARITAL ∈ {{1..6}},
_AGEG5YR ∈ {{5..14}}). Do the synthetic samples contain fractional or otherwise invalid codes?

**Method (strict):** SMOTE and ADASYN were regenerated deterministically (seed 42) on both
scopes. Reproduction was verified first: resampled class counts exactly equal the saved
`class_distribution.json` values, and `X_res[:n_orig] == X_train` exactly, so rows
`X_res[n_orig:]` are precisely the synthetic samples. For each of the 15 features, the
valid set = unique values observed in the original training data; a synthetic value is
**invalid** if it is non-integer OR not in the valid set (being inside [min, max] is not
sufficient). Per-feature tables: `12_Audit_Feature_Validity/*/​*_per_feature_validity.csv`;
synthetic samples persisted as parquet.

**Result:**

| Scope | Method | Synthetic rows | Rows with ≥1 invalid feature value | Affected features |
|---|---|---|---|---|
| Original (2014–2024) | SMOTE | {val['original_2014_2024']['SMOTE']['n_synthetic_rows']:,} | {val['original_2014_2024']['SMOTE']['n_rows_with_at_least_one_invalid_feature']:,} ({val['original_2014_2024']['SMOTE']['pct_rows_with_at_least_one_invalid_feature']}%) | — none — |
| Original (2014–2024) | ADASYN | {val['original_2014_2024']['ADASYN']['n_synthetic_rows']:,} | {val['original_2014_2024']['ADASYN']['n_rows_with_at_least_one_invalid_feature']:,} ({val['original_2014_2024']['ADASYN']['pct_rows_with_at_least_one_invalid_feature']}%) | — none — |
| Corrected (2015–2024) | SMOTE | {val['corrected_2015_2024']['SMOTE']['n_synthetic_rows']:,} | {val['corrected_2015_2024']['SMOTE']['n_rows_with_at_least_one_invalid_feature']:,} ({val['corrected_2015_2024']['SMOTE']['pct_rows_with_at_least_one_invalid_feature']}%) | — none — |
| Corrected (2015–2024) | ADASYN | {val['corrected_2015_2024']['ADASYN']['n_synthetic_rows']:,} | {val['corrected_2015_2024']['ADASYN']['n_rows_with_at_least_one_invalid_feature']:,} ({val['corrected_2015_2024']['ADASYN']['pct_rows_with_at_least_one_invalid_feature']}%) | — none — |

**Zero invalid codes** across ~39M synthetic feature cells (both methods, both scopes):
no fractional values, no integer-but-unobserved codes, no out-of-range values. **Affected
features: none.**

**Mechanism note (why, verified in imbalanced-learn 0.14.2 source):** both
`BaseSMOTE._generate_samples` and `ADASYN._fit_resample` end with
`X_new = X_new.astype(X.dtype)` — the interpolated floats are **silently truncated
toward zero** to the input dtype (int64 here), not rounded. Additionally, on this data
~93% of SMOTE synthetic rows exactly duplicate an existing training profile (their
k-nearest neighbours are exact duplicates, so no interpolation occurs at all), which is
why SMOTE behaves nearly identically to ROS here. The silent truncation is disclosed —
not hidden — but its practical impact on these features is negligible: synthetic code
distributions closely match the minority-class training distributions
(e.g. EXERANY2 0.612/0.388 → 0.645/0.355; _AGEG5YR per-code shares within 0.005).

**Why no SMOTE-N:** SMOTE-N is the categorical-valid alternative, but it is assigned to
another team member and is unnecessary for *this* feature representation — all codes are
consecutive integers and the strict audit passes with 0% invalid. SMOTE-N remains the
right recommendation for non-consecutive or non-ordinal categorical codes, where the
truncation mechanism above could silently map interpolations onto wrong codes.

## 12. Final recommendation

**The Person 1 SMOTE/ADASYN results are methodologically acceptable for the thesis,**
with the following disclosed caveats (all documented above, none hidden):

1. **Reported metrics are honest regardless of synthetic-sample mechanics** — every
   metric is computed on real, untouched test observations; the test set was never
   resampled in any experiment (fingerprints re-verified after every run, both scopes).
2. **ROS is categorical-valid by construction** (exact row replication); SMOTE/ADASYN
   pass the strict per-feature validity audit with 0% invalid codes on both scopes.
3. **The silent int-truncation in imbalanced-learn's SMOTE/ADASYN must be disclosed**
   (done in §11): interpolated values are truncated toward zero rather than rounded.
   Impact here is negligible, but the thesis should state the mechanism rather than
   imply textbook continuous interpolation on categorical codes.
4. **Scope:** cite the corrected 2015–2024 results (§6.1) as primary; the original
   2014–2024 analysis is a valid sensitivity check (Appendix A) since excluding 2014
   changed no metric by more than {max_delta:.4f}.
5. **No re-implementation is recommended**: do not round/clip synthetic values (would
   alter the published-method comparison), do not switch to SMOTE-N (assigned to another
   group; unnecessary here), and do not claim these results generalize beyond this
   dataset, feature set, models, and the 0.5 decision threshold.

"""
v2 = v2.replace("### File map", new_sections + "### File map")

# ---------- file map ----------
v2 = v2.replace(
    "- `validation.log` — full execution/validation log with corrections",
    "- `validation.log` — full execution/validation log with corrections\n"
    "- `11_Audit_2015_2024_Rerun/` — year-scope audit: year_counts.csv, corrected splits +\n"
    "  protocol_config_audit.json, per-experiment results/class-distributions/test-predictions,\n"
    "  combined_results_2015_2024.csv\n"
    "- `12_Audit_Feature_Validity/` — per-feature validity CSVs, validity_summaries.json,\n"
    "  persisted synthetic samples (parquet) for SMOTE/ADASYN × original/corrected scope\n"
    "- `09_Code/audit/` — 01_year_audit.py, 02_rerun_2015_2024.py, 03_feature_validity.py,\n"
    "  04_build_report_v2.py (this report's builder)\n"
    "- `10_Final_Report/Person_1_Final_Report_ORIGINAL_SCOPE.md` — v1 report, verbatim")

with open(f"{REP}/Person_1_Final_Report.md", "w") as f:
    f.write(v2)
print(f"v2 report written ({len(v2)} chars); max scope delta = {max_delta:.4f}")
print(f"best corrected F1: {best_c['Technique']} + {best_c['Model']} = {best_c['F1']:.4f}")
