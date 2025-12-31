import pandas as pd
import os

# === USER INPUT ===
year = 2019
input_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Aggregated_no_sex/aggregated_illness_{year}.csv"
output_dir = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/New_Age_Group_Split_{year}"
os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(input_path)

# === Define age groups ===
age_groups = {
    "AgeGroup_1_2": list(range(1, 3)),
    "AgeGroup_3_4": list(range(3, 5)),
    "AgeGroup_5_9": list(range(5, 10)),
    "AgeGroup_10_13": list(range(10, 14)),
    "AgeGroup_14_18": list(range(14, 19)),
}

# === Split and save ===
for group_name, age_range in age_groups.items():
    group_df = df[df["AgeCode"].isin(age_range)]
    output_path = os.path.join(output_dir, f"{group_name}_{year}.csv")
    group_df.to_csv(output_path, index=False)
    print(f"✅ Saved {group_name} to {output_path}")