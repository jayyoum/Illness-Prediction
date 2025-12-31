import pandas as pd
import os
from itertools import product

def aggregate_by_date_region_sex_fill_zero(
    input_dir: str,
    output_dir: str,
    years: list
):
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        input_file = os.path.join(input_dir, f"illness_{year}_cleaned.csv")
        output_file = os.path.join(output_dir, f"aggregated_illness_{year}.csv")

        print(f"\n📅 Processing {year}...")

        try:
            df = pd.read_csv(input_file, parse_dates=['ParsedDateTime'])

            # Group by date, region, illness, sex
            grouped = df.groupby(
                ['ParsedDateTime', 'RegionCode', 'IllnessCode', 'SexCode']
            ).size().reset_index(name='CaseCount')

            # Get unique values
            all_dates = df['ParsedDateTime'].dropna().unique()
            all_regions = df['RegionCode'].dropna().unique()
            all_illnesses = df['IllnessCode'].dropna().unique()
            all_sexes = df['SexCode'].dropna().unique()

            # Create all combinations
            all_combinations = pd.DataFrame(
                product(all_dates, all_regions, all_illnesses, all_sexes),
                columns=['ParsedDateTime', 'RegionCode', 'IllnessCode', 'SexCode']
            )

            # Merge and fill missing counts
            full_df = pd.merge(
                all_combinations,
                grouped,
                on=['ParsedDateTime', 'RegionCode', 'IllnessCode', 'SexCode'],
                how='left'
            ).fillna({'CaseCount': 0})

            full_df['CaseCount'] = full_df['CaseCount'].astype(int)

            # Save output
            full_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"✅ Saved aggregated file to: {output_file}")

        except Exception as e:
            print(f"❌ Failed to process {year}: {e}")

# === Example usage
if __name__ == "__main__":
    input_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data"
    output_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Aggregated_no_age"
    years = [2019, 2020, 2021, 2022, 2023]

    aggregate_by_date_region_sex_fill_zero(input_dir, output_dir, years)
