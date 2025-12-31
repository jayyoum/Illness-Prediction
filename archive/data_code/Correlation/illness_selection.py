import os
import pandas as pd

# === USER CONFIGURATION ===
base_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Correlation Results"
years = [2019, 2020, 2021, 2022, 2023]
illness_targets = ["J310", "J060", "K297", "N_"]
output_dir = os.path.join(base_dir, "selected_illness")
os.makedirs(output_dir, exist_ok=True)

# === PROCESS EACH YEAR ===
for year in years:
    print(f"\n📅 Processing year {year}...")
    year_dir = os.path.join(base_dir, f"filtered_correlation_{year}")
    if not os.path.isdir(year_dir):
        print(f"❌ Folder not found: {year_dir}")
        continue

    all_records = []

    for lag_folder in os.listdir(year_dir):
        lag_path = os.path.join(year_dir, lag_folder)
        if not os.path.isdir(lag_path) or not lag_folder.startswith("lag_"):
            continue

        lag_num = int(lag_folder.split("_")[1])  # Extract lag number

        for file in os.listdir(lag_path):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(lag_path, file)
            df = pd.read_csv(file_path)

            # Filter for exact illness matches
            matches = df[df["IllnessCode"].astype(str).isin(illness_targets)]

            if not matches.empty:
                matches["Lag"] = lag_num
                matches["GroupFile"] = file
                all_records.append(matches)

    if all_records:
        result_df = pd.concat(all_records, ignore_index=True)
        result_df.sort_values(by=["IllnessCode", "Lag"], inplace=True)
        result_path = os.path.join(output_dir, f"selected_illness_correlations_{year}.csv")
        result_df.to_csv(result_path, index=False)
        print(f"✅ Saved: {result_path}")
    else:
        print("⚠️ No matching illness records found.")