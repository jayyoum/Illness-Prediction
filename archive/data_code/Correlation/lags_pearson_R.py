import os
import pandas as pd
import numpy as np
from collections import defaultdict

def run_lagged_correlation(
    env_data_template: str,
    illness_data_template: str,
    output_base_dir: str,
    year: int,
    max_lag: int = 7
):
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"\n📅 Processing Year: {year}")
    env_df = pd.read_csv(env_data_template.format(year=year), parse_dates=['DateTime'])

    illness_df = pd.read_csv(illness_data_template.format(year=year), parse_dates=['ParsedDateTime'])

    # Do not aggregate (assumes illness is already aggregated by region)
    for lag in range(1, max_lag + 1):
        print(f"⏱️ Running lag {lag}...")

        lagged_df = illness_df.copy()
        lagged_df['ParsedDateTime'] = lagged_df['ParsedDateTime'] + pd.Timedelta(days=lag)

        merged = pd.merge(
            lagged_df,
            env_df,
            left_on=['ParsedDateTime', 'RegionCode'],
            right_on=['DateTime', 'RegionCode'],
            how='inner'
        )

        env_features = env_df.select_dtypes(include='number').columns.tolist()
        env_features = [col for col in env_features if col != 'RegionCode']

        grouped_results = defaultdict(list)
        grouped = merged.groupby('IllnessCode')

        for illness_code, group in grouped:
            group_name = illness_code[0]  # A, B, C, etc.

            for feature in env_features:
                x = group[feature].values
                y = group['CaseCount'].values

                if len(x) < 2 or np.isnan(x).any() or np.isnan(y).any():
                    continue

                r = np.corrcoef(x, y)[0, 1]
                if pd.notnull(r):
                    grouped_results[group_name].append({
                        'IllnessCode': illness_code,
                        'Feature': feature,
                        'PearsonR': round(r, 4),
                        'Year': year
                    })

        # Save results
        lag_output_dir = os.path.join(output_base_dir, f"lag_{lag}")
        os.makedirs(lag_output_dir, exist_ok=True)

        for group_letter, records in grouped_results.items():
            df = pd.DataFrame(records)
            df['AbsR'] = df['PearsonR'].abs()
            df.sort_values(by='AbsR', ascending=False, inplace=True)
            df.drop(columns='AbsR', inplace=True)
            out_path = os.path.join(lag_output_dir, f"Group_{group_letter}_PearsonR.csv")
            df.to_csv(out_path, index=False)
            print(f"✅ Lag {lag} saved: {out_path}")

# 🟢 Example usage
if __name__ == "__main__":
    run_lagged_correlation(
        env_data_template="/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data/{year}_merged_with_lag.csv",
        illness_data_template="/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/aggregated_illness_{year}.csv",
        output_base_dir="/Users/jay/Desktop/Illness Prediction/Processed Data/Correlation Results/Lags_Pearson_R_2023",
        year=2023,
        max_lag=7
    )
