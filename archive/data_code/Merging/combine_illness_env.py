import pandas as pd
import os

# === CONFIGURATION ===
illness_name = "Chronic kidney disease, stage 5"
lag_days = 4  # Shift illness date forward by this many days

# === FILE PATHS ===
illness_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/final_illnesses/combined_illness.csv"
env_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data/merged_env.csv"
output_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental/merged_data_{illness_name.replace(', ', '_').replace(' ', '_')}_lag{lag_days}.csv"

# === LOAD DATA ===
illness_df = pd.read_csv(illness_path, parse_dates=["ParsedDateTime"])
env_df = pd.read_csv(env_path, parse_dates=["DateTime"])

# === FILTER ILLNESS DATA AND APPLY LAG ===
illness_df = illness_df[illness_df["IllnessName"] == illness_name].copy()
illness_df["ParsedDateTime"] = illness_df["ParsedDateTime"] + pd.Timedelta(days=lag_days)

# === MERGE ON DATE + REGION ===
merged = pd.merge(
    illness_df,
    env_df,
    left_on=["ParsedDateTime", "RegionCode"],
    right_on=["DateTime", "RegionCode"],
    how="inner"
)

# === OPTIONAL: Drop NaNs ===
# merged.dropna(inplace=True)

# === SAVE MERGED DATA ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged.to_csv(output_path, index=False)

print(f"✅ Merged data saved to:\n{output_path}")
print(f"🔢 Rows: {len(merged)} | Columns: {len(merged.columns)}")
