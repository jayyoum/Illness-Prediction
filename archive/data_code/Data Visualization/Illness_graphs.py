import os
import pandas as pd
import matplotlib.pyplot as plt

# === USER SETTINGS ===
year = 2019
illness_group = "F"  # Set to A, B, C, ..., Z as needed

# === FILE PATHS ===
agg_data_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Aggregated_no_age_sex/aggregated_illness_{year}.csv"
output_base_dir = f"/Users/jay/Desktop/Illness Prediction/Plots/Illness_Graphs_{year}"

# === LOAD DATA ===
df = pd.read_csv(agg_data_path, parse_dates=["ParsedDateTime"])
df = df[df["IllnessCode"].str.startswith(illness_group)]

# === Create output directory for group ===
output_group_dir = os.path.join(output_base_dir, "Group_" + illness_group)
os.makedirs(output_group_dir, exist_ok=True)

# === Process each illness code ===
illness_codes = df["IllnessCode"].unique()
for illness_code in illness_codes:
    illness_df = df[df["IllnessCode"] == illness_code]
    
    for region in illness_df["RegionCode"].unique():
        region_df = illness_df[illness_df["RegionCode"] == region]
        region_df = region_df.groupby("ParsedDateTime")["CaseCount"].sum().sort_index().reset_index()

        if region_df.empty:
            continue

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(region_df["ParsedDateTime"], region_df["CaseCount"], marker='o', linestyle='-')
        ax.set_title(f"{illness_code} — Region {region} ({year})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Case Count")
        ax.grid(True)

        # Save
        filename = f"{illness_code}_Region_{region}.png"
        filepath = os.path.join(output_group_dir, filename)
        fig.tight_layout()
        fig.savefig(filepath)
        plt.clf()
        plt.close('all')

        print(f"✅ Saved: {filepath}")
