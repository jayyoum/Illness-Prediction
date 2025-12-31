import os
import pandas as pd
from datetime import timedelta

def merge_env_data_for_multiple_years(years, base_dir):
    for year in years:
        try:
            current_file = os.path.join(base_dir, f"{year}_merged.csv")
            prev_file = os.path.join(base_dir, f"{year - 1}_merged.csv")
            output_file = os.path.join(base_dir, f"{year}_merged_with_lag.csv")

            if not os.path.exists(current_file):
                print(f"⚠️ Skipped {year}: missing {current_file}")
                continue
            if not os.path.exists(prev_file):
                print(f"⚠️ Skipped {year}: missing previous year {prev_file}")
                continue

            df_current = pd.read_csv(current_file, parse_dates=['DateTime'])
            df_prev = pd.read_csv(prev_file, parse_dates=['DateTime'])

            first_date_current = df_current['DateTime'].min()
            df_prev_lag = df_prev[df_prev['DateTime'] > (first_date_current - timedelta(days=8))]

            combined = pd.concat([df_prev_lag, df_current], ignore_index=True)
            combined = combined.sort_values(by='DateTime')

            combined.to_csv(output_file, index=False)
            print(f"✅ Merged {year} with previous 7 days. Saved to: {output_file}")

        except Exception as e:
            print(f"❌ Error processing {year}: {e}")

if __name__ == "__main__":
    base_directory = "/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data"
    years_to_merge = [2020, 2021, 2022, 2023]  # Add more years if needed
    merge_env_data_for_multiple_years(years_to_merge, base_directory)