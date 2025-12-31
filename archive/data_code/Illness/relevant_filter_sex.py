import pandas as pd
import os

def filter_by_top_region_strict(agg_path, output_path, min_nonzero_days=115):
    df = pd.read_csv(agg_path, parse_dates=["ParsedDateTime"])

    # Step 1: Group by Illness + Region and calculate counts
    region_stats = df.groupby(["IllnessCode", "RegionCode"])["CaseCount"].agg(
        NonzeroDays=lambda x: (x > 0).sum(),
        DaysOver2=lambda x: (x > 2).sum()
    ).reset_index()

    # Step 2: Pick the region with most nonzero days per illness
    top_regions = region_stats.sort_values("NonzeroDays", ascending=False) \
                              .drop_duplicates("IllnessCode", keep="first")

    # Step 3: Filter illnesses where the top region meets BOTH criteria
    passing = top_regions[
        (top_regions["NonzeroDays"] >= min_nonzero_days) & 
        (top_regions["DaysOver2"] > 0)
    ]["IllnessCode"]

    # Step 4: Filter original dataframe
    filtered_df = df[df["IllnessCode"].isin(passing)]
    filtered_df.to_csv(output_path, index=False)

    print(f"✅ Filtered data saved to: {output_path}")
    print(f"🧼 Illnesses remaining: {len(passing)} of {df['IllnessCode'].nunique()}")


# === Run for both Male and Female files ===
year = 2019
input_base = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Sex_Split_2019"
output_base = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/relevant illnesses sex"

os.makedirs(output_base, exist_ok=True)

for label in ["Male", "Female"]:
    input_path = os.path.join(input_base, f"{label}_{year}.csv")
    output_path = os.path.join(output_base, f"relevant_illness_{label}_{year}.csv")
    print(f"\n🔍 Filtering for: {label}")
    filter_by_top_region_strict(input_path, output_path, min_nonzero_days=115)
