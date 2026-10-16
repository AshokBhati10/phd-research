"""
AUDIT 1a - Year-scope audit: YEAR tabulation + proof the original experiments used 2014 rows.
- Tabulates YEAR value counts from the master (read-only).
- Joins the ORIGINAL saved X_train/X_test (frozen split) with master YEAR via index
  to count YEAR==2014 rows that actually entered the original experiments.
- Recomputes the split from the master and asserts sizes match the original
  (train=1,998,562 / test=499,641) to prove reproducibility of the inclusion.
Does NOT modify the master dataset or any original outputs.
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MASTER = "/home/hatch/workspace/brfss_data/brfss_master.parquet"
BASE = "/home/hatch/workspace/Person_1_Class_Imbalance"
OUT = f"{BASE}/11_Audit_2015_2024_Rerun/01_year_audit"
os.makedirs(OUT, exist_ok=True)

FEATURES = ["EXERANY2", "_MICHD", "ASTHMA3", "ADDEPEV3", "CHCKDNY2",
            "MARITAL", "_EDUCAG", "DRNKANY6", "_AGEG5YR", "_BMI5CAT",
            "_INCOMG1", "_RFSMOK3", "GENHLTH", "HAVARTH4", "SEXVAR"]
TARGET = "DIABETE4"

print("Loading master (read-only)...", flush=True)
df = pd.read_parquet(MASTER)
n_total = len(df)
assert n_total == 2498203, f"master row count changed: {n_total}"

# --- 1. YEAR value counts ---
yc = df["YEAR"].value_counts().sort_index()
year_table = pd.DataFrame({"year": yc.index.astype(int), "n": yc.values.astype(int)})
year_table["pct_of_master"] = (year_table["n"] / n_total * 100).round(4)
year_table.to_csv(f"{OUT}/year_counts.csv", index=False)
years_present = sorted(year_table["year"].tolist())
n2014 = int(yc.get(2014, 0))
pct2014 = round(n2014 / n_total * 100, 4)
print(year_table.to_string(index=False), flush=True)
print(f"YEARS PRESENT: {years_present}", flush=True)
print(f"YEAR=2014: n={n2014:,} ({pct2014}% of master)", flush=True)

# --- 2. Prove the original experiments included 2014 rows ---
# The frozen original split preserved df indices; join YEAR back via index.
Xtr_orig = pd.read_parquet(f"{BASE}/02_Preprocessing/X_train.parquet")
Xte_orig = pd.read_parquet(f"{BASE}/02_Preprocessing/X_test.parquet")
ytr_orig = pd.read_parquet(f"{BASE}/02_Preprocessing/y_train.parquet")["y"]
yte_orig = pd.read_parquet(f"{BASE}/02_Preprocessing/y_test.parquet")["y"]

yr_train = df.loc[Xtr_orig.index, "YEAR"]
yr_test = df.loc[Xte_orig.index, "YEAR"]
n2014_train = int((yr_train == 2014).sum())
n2014_test = int((yr_test == 2014).sum())
print(f"Original experiment TRAIN set: {len(Xtr_orig):,} rows, of which YEAR=2014: {n2014_train:,}", flush=True)
print(f"Original experiment TEST set:  {len(Xte_orig):,} rows, of which YEAR=2014: {n2014_test:,}", flush=True)
assert n2014_train + n2014_test == n2014, "2014 rows not fully accounted for in frozen split"
assert n2014_train > 0 and n2014_test > 0, "unexpected: no 2014 rows in original split"

# --- 3. Recompute the split from the master; assert identical sizes ---
df2 = df.loc[df["_AGEG5YR"] >= 5].copy()
X = df2[FEATURES].astype(int)
y = (df2[TARGET] == "Yes").astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42,
                                      stratify=y, shuffle=True)
assert len(Xtr) == 1998562 and len(Xte) == 499641, "recomputed split sizes differ from original"
assert set(Xtr.index) == set(Xtr_orig.index), "recomputed train indices differ from original"
assert set(Xte.index) == set(Xte_orig.index), "recomputed test indices differ from original"
print("REPRODUCTION ASSERTION PASS: recomputed split is identical to the original "
      "(train=1,998,562 / test=499,641); 2014 rows were included in ALL original experiments.", flush=True)

summary = {
    "master_rows": n_total,
    "years_present": years_present,
    "year_2014_n": n2014,
    "year_2014_pct_of_master": pct2014,
    "original_train_2014_n": n2014_train,
    "original_test_2014_n": n2014_test,
    "original_train_total": 1998562,
    "original_test_total": 499641,
    "reproduction_of_original_split": "PASS (identical indices and sizes)",
    "conclusion": "All four original Person 1 experiments trained on and were evaluated "
                  "against data including 2014 observations (scope 2014-2024), while the "
                  "intended research scope is 2015-2024.",
}
with open(f"{OUT}/year_audit_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2), flush=True)
print("YEAR-SCOPE AUDIT (1a) COMPLETE.", flush=True)
