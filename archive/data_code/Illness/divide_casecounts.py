import pandas as pd
import os

def save_grouped_csvs(df, group_by_cols, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    group_cols = ['ParsedDateTime', 'RegionCode'] + group_by_cols
    grouped = df.groupby(group_cols)['CaseCount'].sum().reset_index()

    for region_code, region_df in grouped.groupby('RegionCode'):
        filename = f"region_{region_code}.csv"
        filepath = os.path.join(output_dir, filename)
        region_df.to_csv(filepath, index=False)
        print(f"✅ Saved: {filepath}")


def process_illness_csv(input_csv, output_base):
    print(f"\n📂 Processing: {input_csv}")
    df = pd.read_csv(input_csv, parse_dates=['ParsedDateTime'])

    # Option 1: Combined across AgeCode and SexCode
    combined_output_dir = os.path.join(output_base, "combined_all")
    save_grouped_csvs(df, group_by_cols=['IllnessCode'], output_dir=combined_output_dir)

    # Option 2: Grouped by AgeCode (combine sexes)
    grouped_by_age_dir = os.path.join(output_base, "grouped_by_age")
    save_grouped_csvs(df, group_by_cols=['IllnessCode', 'AgeCode'], output_dir=grouped_by_age_dir)

    # Option 3: Grouped by SexCode (combine ages)
    grouped_by_sex_dir = os.path.join(output_base, "grouped_by_sex")
    save_grouped_csvs(df, group_by_cols=['IllnessCode', 'SexCode'], output_dir=grouped_by_sex_dir)


if __name__ == "__main__":
    # Change these paths as needed
    input_csv_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/aggregated_illness_2019.csv"
    output_base_folder = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Split_By_Region_2019"

    process_illness_csv(input_csv_path, output_base_folder)
