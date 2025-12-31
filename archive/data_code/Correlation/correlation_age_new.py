import os
import pandas as pd
import numpy as np
from collections import defaultdict

def run_correlation_for_all_age_groups(
    env_data_template: str,
    illness_folder: str,
    output_base_dir: str,
    year: int,
    max_lag: int = 7
):
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"\n📅 Processing Year: {year}")
    env_df = pd.read_csv(env_data_template.format(year=year), parse_dates=['DateTime'])

    # Find all age group illness files for the given year
    agegroup_files = [f for f in os.listdir(illness_folder)
                      if f.endswith(f"{year}.csv") and f.startswith("AgeGroup")]
    
    if not agegroup_files:
        print("❌ No age group illness files found.")
        return

    for age_file in agegroup_files:
        age_group_name = age_file.replace(f"_{year}.csv", "")
        print(f"\n👶 Processing age group: {age_group_name}")

        illness_df = pd.read_csv(os.path.join(illness_folder, age_file), parse_dates=['ParsedDateTime'])

        for lag in range(0, max_lag + 1):
            print(f"⏱️ Lag {lag}...")

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
                y = group['CaseCount'].values

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
                            'Year': year,
                            'AgeGroup': age_group_name
                        })

            # New folder structure: /base/AgeGroup_X/lag_Y/
            age_group_folder = os.path.join(output_base_dir, age_group_name)
            lag_output_dir = os.path.join(age_group_folder, f"lag_{lag}")
            os.makedirs(lag_output_dir, exist_ok=True)

            for group_letter, records in grouped_results.items():
                df = pd.DataFrame(records)
                df['AbsR'] = df['PearsonR'].abs()
                df.sort_values(by='AbsR', ascending=False, inplace=True)
                df.drop(columns='AbsR', inplace=True)
                out_path = os.path.join(lag_output_dir, f"Group_{group_letter}_PearsonR.csv")
                df.to_csv(out_path, index=False)
                print(f"✅ Saved: {out_path}")

# === Example usage
if __name__ == "__main__":
    run_correlation_for_all_age_groups(
        env_data_template="/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data/{year}_merged_with_lag.csv",
        illness_folder="/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/relevant illnesses age",
        output_base_dir="/Users/jay/Desktop/Illness Prediction/Processed Data/Correlation Results/age_filtered_correlation_2019",
        year=2019,
        max_lag=7
    )