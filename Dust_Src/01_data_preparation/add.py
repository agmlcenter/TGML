# add.py
import pandas as pd

files = [
    "DustFeatures_10km_5h_SW_IranVISI2_2018-01-01_2022-01-01.csv",
    "DustFeatures_10km_5h_SW_IranVISI2_2021-01-01_2023-01-01.csv",
    "DustFeatures_10km_5h_SW_IranVISI2_2015-01-01_2018-01-01.csv",
    "DustFeatures_10km_5h_SW_IranVISI2_2023-01-01_2025-01-01.csv"
]

out_file = "DustFeatures_10km_5h_SW_IranVISI2_2015-01-01_2025-01-01_merged_no2018dupe.csv"

dfs = []
for f in files:
    df = pd.read_csv(f)

    # Drop rows from year 2018 in the 2018–2022 file only
    if "2018-01-01_2022-01-01" in f:
        # system:index format like 20210101T05_15_0 -> year = first 4 chars
        year = df["system:index"].astype(str).str.slice(0, 4)
        df = df[year != "2018"].copy()

    dfs.append(df)

# Merge
merged = pd.concat(dfs, ignore_index=True)

# Optional: remove any overlaps by unique timestamp key
if "system:index" in merged.columns:
    merged = merged.drop_duplicates(subset=["system:index"])

# Sort chronologically (works lexicographically for YYYYMMDD...)
merged = merged.sort_values(by="system:index")

# Save
merged.to_csv(out_file, index=False)
print(f"✅ Saved: {out_file}  | rows={len(merged)}")
