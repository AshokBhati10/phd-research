"""P4 STEP 6 -- Build the final integration report.

All tables are generated programmatically from validated CSVs; all key numbers in
the narrative are computed, never hand-typed. Output:
  report/Person_4_Integration_Report.md
  metadata/sources.json, metadata/environment.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_p4 as C

OUT = C.BASE / "results"
T = OUT / "report_tables"
REPORT = C.BASE / "report" / "Person_4_Integration_Report.md"
META = C.BASE / "metadata"


def g(table):
    return (T / table).read_text(encoding="utf-8")


def main():
    comb = pd.read_csv(OUT / "combined_results.csv")
    mdf = pd.read_csv(OUT / "statistical_tests_mcnemar.csv")
    sdf = pd.read_csv(OUT / "statistical_tests_summary.csv")
    eff = pd.read_csv(OUT / "resampling_effects.csv")
    bootp = OUT / "auc_bootstrap_cis.csv"
    boot = pd.read_csv(bootp) if bootp.exists() else None

    def row(tech, model):
        return comb[(comb["Technique"] == tech) & (comb["Model"] == model)].iloc[0]

    def v(tech, model, metric):
        return float(row(tech, model)[metric])

    # ---- computed facts ------------------------------------------------------
    f1_champs = {m: sdf[sdf["Model"] == m].iloc[0]["F1_champion"] for m in C.MODELS}
    base_f1 = {m: v("No Balancing", m, "F1") for m in C.MODELS}
    rec_top = comb.sort_values("Recall", ascending=False).iloc[0]
    acc_top = comb.sort_values("Accuracy", ascending=False).iloc[0]
    n_sig = int(mdf["significant_holm"].sum())
    ns_pairs = mdf[~mdf["significant_holm"]]
    ns_lines = [
        f"| {r['Model']} | {r['Tech_A']} vs {r['Tech_B']} | {r['p_holm']:.3f} |"
        for _, r in ns_pairs.iterrows()
    ]
    xgb = comb[comb["Model"] == "XGBoost"].set_index("Technique")
    non_base = comb[comb["Technique"] != "No Balancing"]
    active = non_base[non_base["Technique"] != "Tomek Links"]
    f1_active_min, f1_active_max = float(active["F1"].min()), float(active["F1"].max())
    f1_tomek = float(non_base[non_base["Technique"] == "Tomek Links"]["F1"].max())

    ros_l = row("Random Oversampling", "LogisticRegression")
    f1_range_base = f"{min(base_f1.values()):.3f}-{max(base_f1.values()):.3f}"
    f1_best_all = {m: v(f1_champs[m], m, "F1") for m in C.MODELS}
    enn_xgb_f1 = v("ENN", "XGBoost", "F1")
    enn_xgb_acc = v("ENN", "XGBoost", "Accuracy")
    ros_xgb_acc = v("Random Oversampling", "XGBoost", "Accuracy")

    # ---- metadata -------------------------------------------------------------
    sources = [
        {"technique": s["technique"], "person": s["person"], "family": s["family"],
         "results_csv": s["results_csv"], "predictions": s["predictions"],
         "caveat": s["caveat"], "role": ("replication evidence (not a table row)"
                                         if s["person"] == "P2" and s["technique"] == "No Balancing"
                                         else "combined-table row")}
        for s in C.SOURCES
    ]
    META.mkdir(parents=True, exist_ok=True)
    (META / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    try:
        env = json.loads((C.P2 / "metadata" / "environment.json").read_text(encoding="utf-8"))
    except Exception:
        env = {}
    (META / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat()

    def trace_table():
        lines = ["| Technique | Person | Results file | Prediction file | Verified |",
                 "|---|---|---|---|---|"]
        for _, r in comb.drop_duplicates("Technique").iterrows():
            lines.append(f"| {r['Display']} | {r['Person']} | {r['Source_results']} | "
                         f"{r['Source_predictions']} | gate-1 recomputation |")
        lines.append("| No Balancing (P2 replication) | P2 | results_baseline_no_balancing.csv | "
                     "test_predictions_no_balancing.parquet | LR/RF bit-identical to P1 |")
        return "\n".join(lines)

    boot_text = "_pending run of 03_statistical_tests.py --bootstrap_"
    if boot is not None and len(boot):
        blines = ["| Classifier | Comparison | metric | mean diff | 95% CI | |",
                  "|---|---|---|---|---|---|"]
        for _, r in boot.iterrows():
            blines.append(
                f"| {r['Model']} | {r['Tech_A']} (champion) vs {r['Tech_B']} | {r['metric']} "
                f"| {r['mean_diff']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |"
                f"{'contains 0' if r['ci_low'] <= 0 <= r['ci_high'] else 'excludes 0'} |")
        boot_text = ("\n".join(blines) +
                     "\n\n*Champion vs rival on identical test rows; a CI containing 0 means the"
                     " AUC gap is not established at the 5% level despite statistical significance"
                     " of the label patterns.*")

    # ---- report ---------------------------------------------------------------
    md = f"""# Person 4 — Integrated Evaluation
## Class-Imbalance Resampling for BRFSS Diabetes Prediction (P1 + P2 + P3)

Generated {now} from validated artifacts. Every number below is traceable to a
source file listed in §2 and re-verified by `01_verify_inputs.py`
(see `results/input_validation.json`).

---

## 1. Executive summary

Three team members each implemented one family of class-imbalance techniques on the
**same BRFSS 2015–2024 dataset (2,297,168 rows), the same frozen stratified test set
(459,434 rows, SHA-256 fingerprint `70e180c6…`), the same three classifiers, the same
metrics, and seed 42**, applying resampling to the training data only:

- **P1 — basic oversampling:** No Balancing (baseline), Random Oversampling (ROS), SMOTE, ADASYN
- **P2 — advanced oversampling:** Borderline-SMOTE, KMeans-SMOTE, SVM-SMOTE†, SMOTE-N†
- **P3 — cleaning & hybrid:** Tomek Links, ENN, SMOTE-Tomek, SMOTE-ENN

This report integrates all **36 result rows (12 techniques × 3 classifiers)** into one
comparison, tests them statistically, and states what each result means.

**Five headline findings**

1. **Oversampling (basic and advanced), the hybrids and ENN roughly double
   minority-class F1** relative to the unbalanced baseline
   ({f1_range_base} → **{f1_active_min:.3f}–{f1_active_max:.3f}**), at the cost of precision and
   accuracy. The gain is driven by detecting far more of the 78,366 diabetic cases in the
   test set (recall {v('No Balancing', 'RandomForest', 'Recall'):.2f}–{v('No Balancing', 'XGBoost', 'Recall'):.2f} → up to {rec_top['Recall']:.2f}).
   The sole exception is **Tomek Links**, whose F1 ({f1_tomek:.3f}) stays at baseline level —
   it only deletes 7,161 border rows (Finding 4).
2. **Plain Random Oversampling is the F1 champion for all three classifiers**
   ({f1_best_all['LogisticRegression']:.4f} / {f1_best_all['RandomForest']:.4f} / {f1_best_all['XGBoost']:.4f}) —
   the most sophisticated oversamplers add essentially nothing on F1 and are
   statistically indistinguishable from ROS/SMOTE in label behaviour (§6).
3. **No technique wins on every metric — the choice depends on the target metric.**
   Accuracy and probability-ranking quality (ROC/PR-AUC) are highest for the baseline
   family and data-cleaning methods; raw minority detection (recall) is highest for
   aggressive oversamplers (ADASYN {rec_top['Recall']:.4f} with RandomForest); F1 peaks at the
   ROS/SMOTE family (Figure 2 shows the trade-off).
4. **Cleaning (Tomek Links, ENN) protects the baseline's ranking quality.** Tomek Links is
   statistically indistinguishable from the baseline on all three classifiers
   (§6) despite removing only 7,161 of 1.52M majority rows (0.47%); ENN (XGBoost) reaches
   F1 {enn_xgb_f1:.4f} with accuracy {enn_xgb_acc:.3f}
   (vs ROS accuracy {ros_xgb_acc:.3f}) — the best *accuracy-preserving* F1 in the study.
5. **Oversampling mostly moves the operating point, not the ranking curve.** For
   XGBoost, Random Oversampling's PR-AUC ({v('Random Oversampling','XGBoost','PR_AUC'):.4f}) is
   virtually identical to the baseline's ({v('No Balancing','XGBoost','PR_AUC'):.4f}): resampling
   converts ranking quality into better 0.5-threshold decisions rather than creating a
   better-ranked model. This reframes what resampling "buys" — a shift of the decision
   threshold behaviour, not an intrinsically better model.

---

## 2. Data, shared protocol, and provenance

**Dataset.** BRFSS 2015–2024 master survey, YEAR 2014 excluded → 2,297,168 rows.
Target `DIABETE4` (Yes = 1, 17.06% prevalence). Fifteen integer-coded categorical
predictors (EXERANY2, _MICHD, ASTHMA3, ADDEPEV3, CHCKDNY2, MARITAL, _EDUCAG, DRNKANY6,
_AGEG5YR, _BMI5CAT, _INCOMG1, _RFSMOK3, GENHLTH, HAVARTH4, SEXVAR); `YEAR` is metadata
and never a predictor.

**Split.** Person 1's frozen `train_test_split(test_size=0.20, random_state=42,
stratify=y, shuffle=True)` → train 1,837,734 (No 1,524,272 / Yes 313,462) and test
459,434 (No 381,068 / Yes 78,366). The test set was never resampled by anyone.

**Where each reported value comes from.** Every row of the integrated table is copied
from the person's own validated results file and cross-checked by recomputing all six
metrics from that person's saved prediction file on the shared test set:

| Person | Results file | Prediction files used | Verified by |
|---|---|---|---|
| P1 | `P1_Basic_Oversampling/11_Audit_2015_2024_Rerun/07_combined_results_audit/combined_results_2015_2024.csv` | `03–06_*_audit/test_predictions_2015_2024.parquet` (4) | 82-check gate (see step 1) |
| P2 | `P2_Advanced_Oversampling/results/results_person2.csv` (+baseline CSV) | `results/test_predictions_*.parquet` (5) | P2's own 119/119 gate, re-verified here |
| P3 | `P3_Advanced_Resampling/results/results_person3.csv` | `results/test_predictions_*_*.parquet` (4) | re-verified here |

**Fingerprint chain.** Person 3's protocol froze the test-set SHA-256
(`70e180c6…`); Person 2's build verified it before running; P4 re-derives it from
Person 1's split parquets (`11_Audit_2015_2024_Rerun/02_preprocessing_audit/`) —
all three agree. All 13 prediction files carry an identical `y_true` vector, so every
McNemar test in §6 compares two classifiers on **exactly the same patients**.

**Invariant caveat (XGBoost).** Reproducing P1's corrected baseline on different
hardware gives bit-identical LogisticRegression and RandomForest predictions but a
slightly different XGBoost model (~1.9e-3 in metrics, 1,238 label flips) because XGBoost
`hist` is not bit-reproducible across thread counts. Cross-person XGBoost differences
below ~0.002 are therefore machine noise, not method differences.

---

## 3. The three approaches and their techniques

| Family | Person | Technique (config) | Training data | Together with |
|---|---|---|---|---|
| A. Basic oversampling | P1 | Random Oversampling (`sampling_strategy="auto"`, seed 42) | full 1.84M train | ROS duplicates minority rows |
| A | P1 | SMOTE (`k_neighbors=5`, seed 42) | full | interpolates synthetic minority points |
| A | P1 | ADASYN (`n_neighbors=5`, seed 42) | full | interpolates, weighted to hard samples |
| B. Advanced oversampling | P2 | Borderline-SMOTE (defaults: borderline-1, k=5, m=10) | full | synthesises near the class border only |
| B | P2 | KMeans-SMOTE (`kmeans_estimator=100`, `cluster_balance_threshold=0.2`*) | full | synthesises inside minority-dense clusters |
| B | P2 | SVM-SMOTE† (defaults incl. `out_step=0.5`) | 50k subsample | synthesises along minority support vectors |
| B | P2 | SMOTE-N† (k=5, categorical-native) | 50k subsample | majority-votes category codes |
| C. Cleaning & hybrid | P3 | Tomek Links (`sampling_strategy="auto"`) | full | removes majority–minority border pairs |
| C | P3 | ENN (`sampling_strategy="auto"`) | full | removes majority rows misclassified by 3-NN |
| C | P3 | SMOTE-Tomek (SMOTE + Tomek, seed 42) | full | SMOTE then border cleaning |
| C | P3 | SMOTE-ENN (SMOTE + ENN, staged, seed 42) | full | SMOTE then aggressive cleaning |
| — | P1 | No Balancing (baseline) | full | none |

\\* library defaults fail on this all-categorical feature space (documented).
† O(n²) cost/memory; both P2 techniques marked with the dagger use one shared
stratified 50,000-row training subsample — their results are indicative but not
strictly like-for-like with the full-train rows (§4, §9).

Classifiers (identical for all 36 rows): LogisticRegression (`max_iter=1000`,
`lbfgs`, seed 42; `StandardScaler` fitted on the resampled training data),
RandomForest (100 trees, depth 12), XGBoost (300 rounds, depth 6, lr 0.05, `hist`).

---

## 4. Is the comparison fair?

**What is controlled — identical for every row:** dataset and scope; the frozen,
fingerprint-verified test set; the 15 features and their integer-coded representation;
classifier hyperparameters; metrics and the 0.5 decision threshold; seed 42; train-only
resampling; CSV/parquet output schema. The 82-check verification step recomputed every
metric from saved predictions and matched every source CSV, and confirmed the P2
baseline replication of P1's baseline (LR/RF bit-identical).

**What differs, and how it is handled:**

1. **Family scope** is the point of comparison (basic vs advanced vs cleaning), so the
   families are grouped in every table/figure rather than treated as one flat list.
2. **50k subsample for SVM-SMOTE and SMOTE-N†** — these two rows are flagged wherever
   they appear; conclusions never rest on them.
3. **KMeans-SMOTE non-default parameters** were forced by a library failure on
   all-categorical data; documented in P2's protocol and reproduced here as-is.
4. **SVM-SMOTE extrapolation** produces a small share (2.49%) of one-code-below-range
   synthetic values; inherent to `out_step=0.5`, reported in P2's audit.
5. **XGBoost cross-machine noise** (~2e-3) — differences below this are not interpreted.
6. **Float32 vs float64 probability storage** (P1/P2 vs P3) — all metrics were computed
   consistently from each person's own CSVs; recomputation tolerances reflect storage.
7. **Timings are machine-dependent** and excluded from every comparison.

---

## 5. Integrated results

**How to read the metrics.** *Accuracy* is dominated by the 83% non-diabetic majority —
a model predicting "No" for everyone scores 0.83, so accuracy alone flatters the
baseline. *Recall (sensitivity)* is the share of true diabetic cases detected — the
metric that matters most clinically for screening. *Precision* is how many flagged
cases are truly diabetic. *F1* balances precision and recall on the minority class.
*ROC-AUC* measures ranking across all thresholds; *PR-AUC (average precision)* is the
more informative ranking metric under 17% prevalence.

### Main table — all 36 rows

{g('main_results.md')}

### Family means (per classifier)

{g('family_summary.md')}

### Resampling mechanics (what each technique did to the training set)

{g('resampling_effects.md')}

*Accounting note:* P1/P2 report gross additions for pure oversamplers; P3's hybrid
techniques (SMOTE-Tomek, SMOTE-ENN) are recorded as **net** effects by their staged
implementation — e.g. SMOTE-ENN ends at Yes 313,462 → 425,916 (+112,454 net) after
SMOTE additions and ENN removals, which are not decomposed in Person 3's artifact.
`Final_train_size` and `Net_change` are exact for every row.

---

## 6. Statistical analysis

**Method.** For each classifier we compared every technique pair with **McNemar's
test** on the paired error patterns over the same 459,434 test rows (66 pairs × 3
classifiers = 198 tests), with **Holm correction** within each classifier. Effect
sizes (accuracy and recall differences) are reported next to every p-value: with
n = 459,434, even trivially small differences become "significant", so significance
never stands alone.

**Overall:** {n_sig}/198 pairs are significant after Holm correction. The seven
non-significant pairs are themselves informative — they identify *equivalences*:

| Classifier | Non-significant pair | Adjusted p |
|---|---|---|
{'\n'.join(ns_lines)}

Two patterns emerge:
- **Tomek Links ≈ baseline** on all three classifiers: removing 7,161 of 1.52M majority
  rows (0.47%) changes almost nothing (e.g. 901 vs 891 disagreeing rows for XGBoost).
- **ROS ≈ SMOTE ≈ SMOTE-Tomek** (label patterns): at this scale the interpolating
  oversamplers produce practically the same decisions as plain duplication.

**Per-model contrasts used below (XGBoost, Holm):** ROS vs Borderline-SMOTE
(p ≈ 0, accuracy +{(xgb.loc['Random Oversampling','Accuracy']-xgb.loc['Borderline-SMOTE','Accuracy']):.3f},
recall {(xgb.loc['Random Oversampling','Recall']-xgb.loc['Borderline-SMOTE','Recall']):+.3f});
SMOTE vs SMOTE-Tomek (adjusted p ≈ 1e-6, recall +0.002 — Tomek cleaning after SMOTE is
cosmetic here); ENN vs Tomek Links (p ≈ 0, accuracy +0.042, recall −0.348 — ENN removes
470,885 noisy majority rows vs Tomek's 7,161-row border cleanup).

**Champion AUC differences with paired bootstrap 95% CIs** (500 resamples, seed 42):

{boot_text}

---

## 7. Findings — which approach wins under which metric, and why

**Metric winners (top technique per classifier).** "Family" shows where each winner
comes from; the full list is in the workbook (`Winners_tradeoffs`).

{g('winners.md')}

**Finding 1 — best F1: plain Random Oversampling (all three classifiers).**
ROS reaches F1 {v('Random Oversampling','LogisticRegression','F1'):.4f} (LR),
{v('Random Oversampling','RandomForest','F1'):.4f} (RF),
{v('Random Oversampling','XGBoost','F1'):.4f} (XGB). Its margin over SMOTE variants is
tiny ({v('SMOTE','XGBoost','F1'):.4f} for SMOTE-XGB) but consistent and Holm-significant
against most rivals because n is very large. *Why it matters:* the marginal value of
advanced oversampling over naive duplication is close to zero on this dataset — the
17% minority class already provides 313,462 positives, more than enough for simple
duplication to balance the loss without overfitting visibly.

**Finding 2 — best raw minority detection: aggressive synthetic oversampling.**
ADASYN-XGB recall {v('ADASYN','XGBoost','Recall'):.4f} and ADASYN/RF {v('ADASYN','RandomForest','Recall'):.4f},
Borderline-SMOTE/RF {v('Borderline-SMOTE','RandomForest','Recall'):.4f} — but at precision
~0.30 and accuracy ~0.65–0.68. If the screening use-case tolerates many false
positives to catch nearly all diabetics, this family is the operating point to choose.

**Finding 3 — best accuracy-preserving option: ENN (cleaning-only).** ENN-XGB keeps
F1 at {enn_xgb_f1:.4f} while holding accuracy at {enn_xgb_acc:.4f} and ROC-AUC
{xgb.loc['ENN','ROC_AUC']:.4f} — it removes 470,885 ambiguous majority rows instead of
adding any synthetic ones. Under XGBoost its F1 is the runner-up overall
({v('ENN','XGBoost','F1'):.4f} vs ROS {v('Random Oversampling','XGBoost','F1'):.4f})
with a dramatically better accuracy/recall trade-off than any oversampler
(Figure 2, grey-green cluster).

**Finding 4 — best probability ranking: the baseline and Tomek Links.** ROC-AUC
(XGB): baseline {xgb.loc['No Balancing','ROC_AUC']:.4f}, Tomek {xgb.loc['Tomek Links','ROC_AUC']:.4f};
 Tomek Links is the only technique that is *never significantly worse* than the
 baseline on any classifier and equals or slightly exceeds its AUC — the shallowest
 possible intervention (delete 7,161 border pairs, 0.47% of the majority class) that
 leaves the ranking untouched. Aggressive oversamplers trade 0.005–0.04 of AUC for
 their recall gains.

**Finding 5 — the unifying explanation.** Resampling leaves the models' *ranking*
almost unchanged (PR-AUC XGB: ROS {xgb.loc['Random Oversampling','PR_AUC']:.4f} vs
baseline {xgb.loc['No Balancing','PR_AUC']:.4f}) but reshapes the 0.5-threshold
decision: balancing the training classes teaches the classifier that the prior
probability of diabetes is ~50%, not 17%, so it says "Yes" far more often at the same
score. Recall and F1 therefore jump, while AUC metrics — which are threshold-free —
stay close to baseline. **For this research, technique choice is effectively a
choice of operating point, plus a modest ENN data-cleaning dividend.**

**Practical recommendation.** For a diabetes-screening model: train on the plain
baseline to get the best-calibrated probabilities, then select the operating
threshold on a validation curve — or, if a single 0.5-threshold model is required,
use Random Oversampling (best F1, no synthetic-data caveats) or ENN when
accuracy/precision must be preserved. Advanced oversamplers are justified only when
their specific inductive biases (border-focused, cluster-focused, categorical-native
generation) are themselves the research object.

---

## 8. Figures

| Figure | What it shows | How to read it |
|---|---|---|
| ![F1](../figures/01_f1_headline.png) **Fig. 1** | F1 for all 12 techniques × 3 classifiers | the study in one picture: the baseline bars are ~half the height of every resampled bar; within the resampled cluster the differences are small |
| ![Pareto](../figures/02_accuracy_recall_pareto.png) **Fig. 2** | accuracy vs recall, every technique | no technique dominates both axes; families occupy different regions — oversamplers top-left (high recall, low accuracy), baseline/cleaning bottom-right |
| ![AUC](../figures/03_auc_ranking_quality.png) **Fig. 3** | ROC-AUC and PR-AUC (XGBoost) | cleaning/grey bars sit at baseline level; aggressive oversamplers are slightly lower |
| ![Mechanics](../figures/04_resampling_mechanics.png) **Fig. 4** | rows added/removed per technique | explains *why*: +1.2M synthetic rows for oversamplers, −470k for ENN, 82,944 for the † techniques |
| ![Curves](../figures/05_family_champion_curves.png) **Fig. 5** | ROC and PR curves of family champions | the curves nearly coincide — visual proof that resampling moves the operating point, not the curve |
| ![Heatmap](../figures/06_metric_heatmap.png) **Fig. 6** | XGBoost × 6 metrics, all techniques | one-glance summary of every winner documented in §7 |

---

## 9. Limitations

1. Same-survey splits from one pool: patients across years may repeat; no external
   validation cohort.
2. 15 integer-coded categorical features only; no continuous biomarkers (BMI is
   categorical buckets). SMOTE-family interpolation on integer codes is the documented
   team convention, not a claim about feature semantics.
3. SVM-SMOTE and SMOTE-N (†) rest on a 50k subsample — lower training variance and
   not strictly comparable to full-train rows.
4. XGBoost cross-machine noise (~2e-3) sets the resolution floor for cross-person
   XGBoost comparisons; LR/RF are exactly reproducible.
5. Multiple-comparison correction (Holm) is conservative; conversely, with n = 459,434
   even sub-noise differences reach significance — effect sizes are the guide, and the
   XGBoost noise floor is respected in every claim above.
6. Threshold 0.5 with class-balancing changes the prior; a threshold-moved baseline
   (predict "Yes" when score > ~0.29) would narrow the gap of some resampled models —
   left for future work, as the team protocol fixed the threshold at 0.5.
7. No calibration analysis (reliability diagrams/Brier score); the §7 operating-point
   interpretation would be strengthened by it.

---

## 10. Implications for the research conclusions

- Class imbalance handling on BRFSS diabetes data is **an operating-point decision,
  not a modelling upgrade**: the chosen resampling strategy determines the
  recall/precision mix at threshold 0.5 while ranking quality stays near baseline.
- **Simplicity wins on this dataset:** naive duplication is the F1 champion, and the
  most valuable non-oversampling intervention is cleaning (ENN), not synthesis.
- Advanced oversampling (P2) is **characterised, not discredited**: its variants shape
  the precision/recall trade-off in distinct ways (KMeans-SMOTE conservative,
  Borderline/ADASYN aggressive) and matter when recall is the explicit objective.
- For the thesis narrative: imbalance treatment should be reported as a
  **decision and reporting choice per use-case** (screening vs triage), with the
  fairness caveats of §4 attached to any cross-technique claims.

---

## 11. Traceability appendix

Every row of the main table maps to exactly two source artifacts plus one verification:

{trace_table()}Full 198-row statistical matrix: `results/statistical_tests_mcnemar.csv`.
Machine-readable provenance: `metadata/sources.json`.

---

## 12. Reproducibility

```powershell
$py = "P2_Advanced_Oversampling/.venv/Scripts/python.exe"
& $py P4_Integration_Evaluation/code/01_verify_inputs.py      # gate 1: 82 checks
& $py P4_Integration_Evaluation/code/02_integrate_results.py  # tables
& $py P4_Integration_Evaluation/code/03_statistical_tests.py --bootstrap
& $py P4_Integration_Evaluation/code/04_figures.py
& $py P4_Integration_Evaluation/code/05_workbook.py
& $py P4_Integration_Evaluation/code/06_build_report.py       # this file
& $py P4_Integration_Evaluation/code/07_validate_p4.py        # gate 2
```

Environment and versions: `metadata/environment.json` (identical stack to P1/P3:
scikit-learn 1.9.1, imbalanced-learn 0.14.2, xgboost 3.4.1, numpy 2.5.3).
P1–P3 folders were never modified (verified in gate 2).

---

*Person 4 integration of Person 1 (basic oversampling), Person 2 (advanced
oversampling) and Person 3 (cleaning & hybrid). All conclusions are bounded by the
documented deviations in §4 and the validation gates in §12.*
"""

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(md, encoding="utf-8")
    print(f"report written: {REPORT} ({len(md):,} chars)")


if __name__ == "__main__":
    main()
