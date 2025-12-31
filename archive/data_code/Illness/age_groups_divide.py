import pandas as pd
import os

# === USER INPUT ===
year = 2023
input_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/aggregated_illness_{year}.csv"
output_dir = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Age_Group_Split_{year}"
os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(input_path)

# === Define age groups ===
age_groups = {
    "AgeGroup_1_3": list(range(1, 4)),
    "AgeGroup_4_8": list(range(4, 9)),
    "AgeGroup_9_13": list(range(9, 14)),
    "AgeGroup_14_18": list(range(14, 19)),
}

# === Split and save ===
for group_name, age_range in age_groups.items():
    group_df = df[df["AgeCode"].isin(age_range)]
    output_path = os.path.join(output_dir, f"{group_name}_{year}.csv")
    group_df.to_csv(output_path, index=False)
    print(f"✅ Saved {group_name} to {output_path}")