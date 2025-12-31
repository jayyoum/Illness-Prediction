import pandas as pd
import os

# === PATHS ===
env_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data"
output_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data/merged_env.csv"

# === COMBINE ENVIRONMENTAL FILES ===
combined_env = pd.DataFrame()

for year in [2019, 2020, 2021, 2022, 2023]:
    path = os.path.join(env_dir, f"{year}_merged_with_lag.csv")
    df = pd.read_csv(path, parse_dates=["DateTime"])
    df["Year"] = year
    combined_env = pd.concat([combined_env, df], ignore_index=True)
    print(f"✅ Loaded {year} — Rows: {len(df)}")

# === SAVE COMBINED OUTPUT ===
combined_env.to_csv(output_path, index=False)
print(f"\n✅ Combined environmental file saved to: {output_path} — Total rows: {len(combined_env)}")