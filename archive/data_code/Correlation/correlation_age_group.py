import os
import pandas as pd
import numpy as np
from collections import defaultdict

def run_lagged_correlation(
    env_data_template: str,
    illness_file_path: str,
    output_base_dir: str,
    year: int,
    age_group_label: str,
    max_lag: int = 7
):
    output_base_dir = os.path.join(output_base_dir, age_group_label)
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"\n📅 Processing Year: {year} — Age Group: {age_group_label}")
    env_df = pd.read_csv(env_data_template.format(year=year), parse_dates=['DateTime'])
    illness_df = pd.read_csv(illness_file_path, parse_dates=['ParsedDateTime'])

    for lag in range(0, max_lag + 1):
        print(f"\n⏱️ Running lag {lag}...")

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
            group_name = illness_code[0]
            group = group[group['CaseCount'] > 1]
            y = group['CaseCount'].values

            if len(y) < 2:
                print(f"⛔ Skipping {illness_code} — too few nonzero values")
                continue

            mean_y = np.mean(y)
            std_y = np.std(y)
            range_y = np.ptp(y)

            if range_y < 3 or (mean_y > 0 and std_y / mean_y < 0.05):
                print(f"⛔ Skipping {illness_code} — insufficient variation (std/mean={std_y/mean_y:.4f})")
                continue

            for feature in env_features:
                x = group[feature].values

                if len(x) != len(y) or len(x) < 2:
                    continue
                if np.isnan(x).any() or np.isnan(y).any():
                    continue

                r = np.corrcoef(x, y)[0, 1]
                if pd.notnull(r):
                    grouped_results[group_name].append({
                        'IllnessCode': illness_code,
                        'Feature': feature,
                        'PearsonR': round(r, 4),
                        'Year': year
                    })

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


# 🔁 Loop through all age groups
if __name__ == "__main__":
    year = 2019
    base_env_template = "/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data/{year}_merged_with_lag.csv"
    base_output_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Correlation Results/9Age_Split_Correlation_2019"
    illness_base_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Age_Group_Split_2019"

    age_groups = {
        "AgeGroup_1_3": "AgeGroup_1_3_2019.csv",
        "AgeGroup_4_8": "AgeGroup_4_8_2019.csv",
        "AgeGroup_9_13": "AgeGroup_9_13_2019.csv",
        "AgeGroup_14_18": "AgeGroup_14_18_2019.csv"
    }

    for label, file_name in age_groups.items():
        illness_path = os.path.join(illness_base_dir, file_name)
        run_lagged_correlation(
            env_data_template=base_env_template,
            illness_file_path=illness_path,
            output_base_dir=base_output_dir,
            year=year,
            age_group_label=label,
            max_lag=7
        )
