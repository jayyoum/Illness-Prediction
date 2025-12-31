import pandas as pd
import os
import itertools

def aggregate_illness_data(
    input_dir: str,
    output_dir: str,
    years: list
):
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        input_file = os.path.join(input_dir, f"illness_{year}_cleaned.csv")
        output_file = os.path.join(output_dir, f"aggregated_illness_{year}.csv")
        
        print(f"\nProcessing {year}...")

        try:
            df = pd.read_csv(input_file, parse_dates=["ParsedDateTime"])

            # Aggregate the actual data
            actual = df.groupby(
                ['ParsedDateTime', 'RegionCode', 'IllnessCode', 'AgeCode', 'SexCode']
            ).size().reset_index(name='CaseCount')

            # Get all unique values for key dimensions
            all_dates = df['ParsedDateTime'].dropna().unique()
            all_regions = df['RegionCode'].dropna().unique()
            all_illnesses = df['IllnessCode'].dropna().unique()
            all_ages = df['AgeCode'].dropna().unique()
            all_sexes = df['SexCode'].dropna().unique()

            # Create all possible combinations
            full_index = pd.DataFrame(
                itertools.product(all_dates, all_regions, all_illnesses, all_ages, all_sexes),
                columns=['ParsedDateTime', 'RegionCode', 'IllnessCode', 'AgeCode', 'SexCode']
            )

            # Merge and fill missing with 0
            merged = pd.merge(full_index, actual, on=['ParsedDateTime', 'RegionCode', 'IllnessCode', 'AgeCode', 'SexCode'], how='left')
            merged['CaseCount'] = merged['CaseCount'].fillna(0).astype(int)

            merged.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"✅ Saved aggregated file to: {output_file}")

        except Exception as e:
            print(f"❌ Failed to process {year}: {e}")

# Example usage
if __name__ == "__main__":
    input_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data"
    output_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Aggregated"
    years = [2019, 2020, 2021, 2022, 2023]

    aggregate_illness_data(input_dir, output_dir, years)