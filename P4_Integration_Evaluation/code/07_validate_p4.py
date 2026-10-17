"""P4 STEP 7 -- Final validation gate (gate 2).

Verifies every P4 deliverable is consistent with its sources and that
P1/P2/P3 folders were never modified:
  1. gate-1 input validation still all-pass,
  2. combined table = 36 rows from all 12 canonical techniques,
  3. statistical matrix = 198 rows, Holm adjustment sane,
  4. six figures x (png, pdf) exist,
  5. workbook exists with expected sheets,
  6. report exists and embeds the generated tables (no hand transcription),
  7. P1/P2/P3 folders untouched in git.
Writes results/validation_report.json; exits nonzero on failure.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_p4 as C

OUT = C.BASE / "results"
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def main():
    # 1. gate 1
    iv = json.loads((OUT / "input_validation.json").read_text(encoding="utf-8"))
    check("gate 1 (input verification) all pass", iv.get("all_pass") is True,
          f"{sum(c['pass'] for c in iv['checks'])}/{len(iv['checks'])}")

    # 2. combined table
    comb = pd.read_csv(OUT / "combined_results.csv")
    check("combined table has 36 rows", len(comb) == 36, f"{len(comb)}")
    check("12 canonical techniques present",
          set(comb["Technique"]) == set(C.CANONICAL_ORDER) - {"No Balancing"} | {"No Balancing"},
          f"{comb['Technique'].nunique()} techniques")
    check("3 rows per technique", bool((comb.groupby('Technique')['Model'].count() == 3).all()))
    check("source columns populated", bool(comb[["Source_results", "Source_predictions"]].notna().all().all()))
    check("caveats recorded for dagger techniques",
          bool(comb[comb["Technique"].isin(["SVM-SMOTE", "SMOTE-N"])]["Caveat"].str.contains("50k").all()))

    # 3. statistics
    mdf = pd.read_csv(OUT / "statistical_tests_mcnemar.csv")
    check("McNemar matrix has 198 rows", len(mdf) == 198, f"{len(mdf)}")
    check("66 pairs per classifier", bool(mdf.groupby("Model")["Tech_A"].count().eq(66).all()))
    check("Holm p-values monotone with raw p",
          bool((mdf["p_holm"] >= mdf["p_value"] - 1e-12).all()))
    check("no p_holm reported as tiny when raw is 1",
          bool(~((mdf["p_value"] >= 0.99) & (mdf["p_holm"] < 0.05)).any()))

    # 4. figures
    stems = ["01_f1_headline", "02_accuracy_recall_pareto", "03_auc_ranking_quality",
             "04_resampling_mechanics", "05_family_champion_curves", "06_metric_heatmap"]
    for s in stems:
        ok = (C.BASE / "figures" / f"{s}.png").exists() and (C.BASE / "figures" / f"{s}.pdf").exists()
        check(f"figure {s} (png+pdf)", ok)

    # 5. workbook
    from openpyxl import load_workbook
    wbp = C.BASE / "workbook" / "Person_4_Integration_Workbook.xlsx"
    check("workbook exists", wbp.exists())
    wb = load_workbook(wbp, read_only=True)
    needed = {"Overview", "Combined_results", "Family_summary", "Winners_tradeoffs",
              "Mean_ranks", "Stat_tests_summary", "Stat_tests_full", "AUC_bootstrap_CIs",
              "Resampling_effects", "Fairness_register", "Traceability"}
    check("workbook sheets complete", needed <= set(wb.sheetnames), str(sorted(set(wb.sheetnames))))
    ws = wb["Combined_results"]
    check("workbook Combined_results has 37 rows (header + 36)", ws.max_row == 37, str(ws.max_row))

    # 6. report
    rep = C.BASE / "report" / "Person_4_Integration_Report.md"
    check("report exists", rep.exists())
    if rep.exists():
        text = rep.read_text(encoding="utf-8")
        for stem, table in [("main results", "results/report_tables/main_results.md"),
                            ("family summary", "results/report_tables/family_summary.md"),
                            ("resampling effects", "results/report_tables/resampling_effects.md")]:
            p = C.BASE / table
            body = p.read_text(encoding="utf-8")
            probe = body.splitlines()[0]
            check(f"report embeds generated {stem} table", probe in text)
            # spot-check a data line too
            data_probe = body.splitlines()[2]
            check(f"report {stem} data rows match generation", data_probe in text)
        for s in stems:
            check(f"report references figure {s}", s in text)
        check("report states the 36-row provenance", "36 result rows" in text or "36 rows" in text)
        check("report documents the test fingerprint", "70e180c6" in text)
        check("report includes limitations section", "## 9. Limitations" in text)

    # 7. P1/P2/P3 untouched
    try:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=C.REPO,
                                capture_output=True, text=True, check=True).stdout
        dirty = [ln for ln in status.splitlines()
                 if ln and ln.split()[1].startswith(("P1_Basic_Oversampling",
                                                     "P2_Advanced_Oversampling",
                                                     "P3_Advanced_Resampling"))]
        check("P1/P2/P3 folders untouched in git", not dirty, "; ".join(dirty[:5]))
    except Exception as exc:  # pragma: no cover
        check("git available for untouched check", False, str(exc))

    passed = sum(c["pass"] for c in checks)
    report = {"generated_at_utc": pd.Timestamp.utcnow().isoformat(),
              "passed": passed, "failed": len(checks) - passed,
              "all_pass": passed == len(checks), "checks": checks}
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{passed}/{len(checks)} checks passed")
    print(f"report: {OUT / 'validation_report.json'}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
