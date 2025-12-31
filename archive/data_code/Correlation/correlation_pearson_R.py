import pandas as pd
import numpy as np
import os
from collections import defaultdict

def run_correlation_analysis_all_regions(
    env_data_path: str,
    illness_dir: str,
    output_dir: str,
    years: list
):
    os.makedirs(output_dir, exist_ok=True)
    
    grouped_results = defaultdict(list)

    for year in years:
        print(f"\n📅 Running correlation for {year}...")

        try:
            # Load data
            env_df = pd.read_csv(env_data_path.format(year=year), parse_dates=['DateTime'])
            illness_df = pd.read_csv(os.path.join(illness_dir, f"aggregated_illness_{year}.csv"), parse_dates=['ParsedDateTime'])

            # Step 1: Aggregate illness data across Age and Sex
            illness_df = illness_df.groupby(['ParsedDateTime', 'RegionCode', 'IllnessCode'])['CaseCount'].sum().reset_index()

            # Step 2: Merge with environmental data
            merged = pd.merge(
                illness_df,
                env_df,
                left_on=['ParsedDateTime', 'RegionCode'],
                right_on=['DateTime', 'RegionCode'],
                how='inner'
            )

            # Step 3: Get environmental numeric columns
            env_features = env_df.select_dtypes(include='number').columns.tolist()
            env_features = [col for col in env_features if col != 'RegionCode']

            # Step 4: Correlation per illness code
            grouped = merged.groupby('IllnessCode')

            for illness_code, group in grouped:
                group_name = illness_code[0]  # Get group: A, B, C, etc.

                for feature in env_features:
                    x = group[feature].values
                    y = group['CaseCount'].values

                    if len(x) < 2 or np.isnan(x).any() or np.isnan(y).any():
                        continue  # Skip if insufficient or invalid data
                    
                    # Pearson r using np.corrcoef
                    r = np.corrcoef(x, y)[0, 1]
                    if pd.notnull(r):
                        grouped_results[group_name].append({
                            'IllnessCode': illness_code,
                            'Feature': feature,
                            'PearsonR': round(r, 4),
                            'Year': year
                        })

        except Exception as e:
            print(f"❌ Error processing {year}: {e}")

    # Step 5: Save each illness group to separate CSV
    for group_letter, records in grouped_results.items():
        df = pd.DataFrame(records)
        df['AbsR'] = df['PearsonR'].abs()
        df.sort_values(by='AbsR', ascending=False, inplace=True)
        df.drop(columns='AbsR', inplace=True)
        output_path = os.path.join(output_dir, f"Group_{group_letter}_PearsonR.csv")
        df.to_csv(output_path, index=False)
        print(f"✅ Saved correlations to {output_path}")

if __name__ == "__main__":
    env_data_template = "/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data/{year}_merged.csv"
    illness_data_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data"
    output_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Correlation Results/Pearson_R_2019"
    years = [2019]

    run_correlation_analysis_all_regions(env_data_template, illness_data_dir, output_dir, years)
