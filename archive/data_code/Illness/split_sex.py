import pandas as pd
import os

# === USER INPUT ===
year = 2019
input_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Aggregated_no_age/aggregated_illness_{year}.csv"
output_dir = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Sex_Split_{year}"
os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(input_path)

# === Split by SexCode ===
sex_map = {
    1: "Male",
    2: "Female"
}

for sex_code, label in sex_map.items():
    sex_df = df[df["SexCode"] == sex_code]
    output_path = os.path.join(output_dir, f"{label}_{year}.csv")
    sex_df.to_csv(output_path, index=False)
    print(f"✅ Saved {label} data: {output_path} — Rows: {len(sex_df)}")