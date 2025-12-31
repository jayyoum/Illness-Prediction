import pandas as pd
import os
from itertools import product

def aggregate_by_year_and_agecodes(
    input_file: str,
    output_file: str,
    age_codes: list
):
    try:
        print(f"\n📅 Aggregating for: {os.path.basename(output_file)} | AgeCodes: {age_codes}")
        
        df = pd.read_csv(input_file, parse_dates=['ParsedDateTime'])

        # Filter only selected AgeCodes
        df = df[df["AgeCode"].isin(age_codes)]

        # Group by Date, Region, Illness, AgeCode
        grouped = df.groupby(
            ['ParsedDateTime', 'RegionCode', 'IllnessCode', 'AgeCode']
        ).size().reset_index(name='CaseCount')

        # Get unique combinations only from filtered data
        all_dates = df['ParsedDateTime'].dropna().unique()
        all_regions = df['RegionCode'].dropna().unique()
        all_illnesses = df['IllnessCode'].dropna().unique()
        all_agecodes = df['AgeCode'].dropna().unique()

        all_combinations = pd.DataFrame(
            product(all_dates, all_regions, all_illnesses, all_agecodes),
            columns=['ParsedDateTime', 'RegionCode', 'IllnessCode', 'AgeCode']
        )

        # Merge and fill missing counts with 0
        full_df = pd.merge(
            all_combinations,
            grouped,
            on=['ParsedDateTime', 'RegionCode', 'IllnessCode', 'AgeCode'],
            how='left'
        ).fillna({'CaseCount': 0})

        full_df['CaseCount'] = full_df['CaseCount'].astype(int)

        # Save output
        full_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ Saved: {output_file} — Rows: {len(full_df)}")

    except Exception as e:
        print(f"❌ Error: {e}")


# === Example usage ===
if __name__ == "__main__":
    year = 2019
    input_file = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/illness_{year}_cleaned.csv"
    output_file = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Aggregated_no_sex/aggregated_illness_{year}_age_16_18.csv"
    age_codes = [16, 17, 18]

    aggregate_by_year_and_agecodes(input_file, output_file, age_codes)