import pandas as pd
import os

# === PATHS ===
illness_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/final_illnesses"
output_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/final_illnesses/combined_illness.csv"

# === COMBINE ILLNESS FILES ===
combined_illness = pd.DataFrame()

for year in [2019, 2020, 2021, 2022, 2023]:
    path = os.path.join(illness_dir, f"relevant_illness_{year}.csv")
    df = pd.read_csv(path, parse_dates=["ParsedDateTime"])
    df["Year"] = year
    combined_illness = pd.concat([combined_illness, df], ignore_index=True)
    print(f"✅ Loaded {year} — Rows: {len(df)}")

# === SAVE COMBINED OUTPUT ===
combined_illness.to_csv(output_path, index=False)
print(f"\n✅ Combined illness file saved to: {output_path} — Total rows: {len(combined_illness)}")
