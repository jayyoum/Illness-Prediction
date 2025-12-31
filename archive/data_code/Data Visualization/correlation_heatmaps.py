import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_illness_heatmaps_for_year(base_dir: str):
    # Identify all lag folders
    lag_folders = [f for f in sorted(os.listdir(base_dir)) if f.startswith("lag_")]
    if not lag_folders:
        print(f"⚠️ No lag folders found in {base_dir}")
        return

    # Collect all group letters automatically
    group_letters = set()
    for folder in lag_folders:
        lag_path = os.path.join(base_dir, folder)
        for file in os.listdir(lag_path):
            if file.startswith("Group_") and file.endswith("_PearsonR.csv"):
                group_letter = file.split("_")[1]
                group_letters.add(group_letter)

    print(f"📂 Found groups: {sorted(group_letters)}")

    for group_letter in sorted(group_letters):
        print(f"\n📊 Generating heatmaps for Group {group_letter}...")

        output_dir = os.path.join(base_dir, "Illness_Heatmaps", f"Group_{group_letter}")
        os.makedirs(output_dir, exist_ok=True)

        # Collect all lag data for this group
        lag_data = {}
        for folder in lag_folders:
            lag_num = int(folder.split("_")[1])
            group_path = os.path.join(base_dir, folder, f"Group_{group_letter}_PearsonR.csv")
            if os.path.exists(group_path):
                df = pd.read_csv(group_path)
                df['lag'] = lag_num
                lag_data[lag_num] = df

        if not lag_data:
            continue

        combined_df = pd.concat(lag_data.values(), ignore_index=True)
        illness_codes = combined_df['IllnessCode'].unique()

        for illness in illness_codes:
            illness_df = combined_df[combined_df['IllnessCode'] == illness]
            pivot = illness_df.pivot(index='lag', columns='Feature', values='PearsonR')

            plt.figure(figsize=(23, 8))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                annot_kws={"size": 9}
            )
            plt.title(f"Illness {illness} - Pearson Correlations by Lag and Feature", fontsize=16)
            plt.ylabel("Lag (Days)", fontsize=12)
            plt.xlabel("Environmental Feature", fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()

            filename = f"Illness_{illness}_heatmap.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            print(f"✅ Saved heatmap: {filepath}")

# 🟢 Run this on multiple years
if __name__ == "__main__":
    base_root = "/Users/jay/Desktop/Illness Prediction/Processed Data/Correlation Results"
    years = [2019, 2020, 2021, 2022, 2023]  # ← Edit this list if needed

    for year in years:
        print(f"\n================= 📆 YEAR {year} =================")
        year_dir = os.path.join(base_root, f"cleaned_Lags_Pearson_R_{year}")
        if os.path.exists(year_dir):
            generate_illness_heatmaps_for_year(year_dir)
        else:
            print(f"❌ Directory not found: {year_dir}")
