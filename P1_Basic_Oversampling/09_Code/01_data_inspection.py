"""
STEP 1 - Data Inspection of BRFSS 2015-2024 Master Dataset
Person 1 | Diabetes prediction class-imbalance study
Outputs: 01_Data_Inspection/
"""
import json
import numpy as np
import pandas as pd

OUT = "/home/hatch/workspace/Person_1_Class_Imbalance/01_Data_Inspection"
SRC = "/home/hatch/workspace/brfss_data/brfss_master.parquet"

log = []
def v(msg):
    log.append(msg)
    print(msg, flush=True)

df = pd.read_parquet(SRC)
v(f"file name: brfss_master.parquet")
v(f"rows: {df.shape[0]:,}")
v(f"columns: {df.shape[1]}")
v(f"column names: {list(df.columns)}")
v("\ndtypes:")
v(str(df.dtypes))
v(f"\nmemory usage (MB): {df.memory_usage(deep=True).sum()/1e6:.1f}")

# year coverage
year_cols = [c for c in df.columns if 'year' in c.lower() or 'iyear' in c.lower() or c.upper().startswith('IYEAR')]
v(f"\ncandidate year columns: {year_cols}")
for c in year_cols:
    v(f"value counts of {c}:\n{df[c].value_counts(dropna=False).sort_index()}")

# target candidate columns
for cand in ["DIABETE4", "DIABETE3", "DIABETES", "diabetes"]:
    if cand in df.columns:
        v(f"\n{cand} value counts (incl. NaN):\n{df[cand].value_counts(dropna=False)}")

# missing values
miss = df.isna().sum()
v(f"\ntotal missing cells: {int(miss.sum())}")
if miss.sum() > 0:
    v(str(miss[miss > 0]))
else:
    v("NO missing values in any column")

# duplicates
dup = int(df.duplicated().sum())
v(f"\nduplicate rows: {dup}")

# near-constant columns
v("\nvalue-counts per column (top 5):")
uniq = {}
for c in df.columns:
    vc = df[c].value_counts(dropna=False)
    uniq[c] = len(vc)
    top = vc.head(5).to_dict()
    dom = vc.iloc[0] / len(df)
    v(f"  {c}: n_unique={uniq[c]}, top={top}, dominant_share={dom:.4f}")

v("\nconstant columns (1 unique): " + str([c for c, u in uniq.items() if u == 1]))
v("near-constant columns (>0.99 share): " + str([c for c in df.columns
      if df[c].value_counts(dropna=False).iloc[0]/len(df) > 0.99]))

# suspicious values: negatives, extreme
v("\ndescribe (numeric):")
v(str(df.describe(include=[np.number]).T))

# save column inventory
inv = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(t) for t in df.dtypes],
    "n_unique": [uniq[c] for c in df.columns],
    "n_missing": [int(miss[c]) for c in df.columns],
})
inv.to_csv(f"{OUT}/column_inventory.csv", index=False)

with open(f"{OUT}/inspection_log.txt", "w") as f:
    f.write("\n".join(log))
v("\nsaved column_inventory.csv and inspection_log.txt")
