import os
import pandas as pd

def merge_aggregated_age_files(input_dir, output_file, year):
    # Match files like: aggregated_illness_2019_age_1_3.csv
    pattern_prefix = f"aggregated_illness_{year}_age_"
    csv_files = [f for f in os.listdir(input_dir) if f.startswith(pattern_prefix) and f.endswith(".csv")]

    if not csv_files:
        print(f"❌ No files found in {input_dir} for year {year}.")
        return

    print(f"📂 Found {len(csv_files)} files to merge.")
    
    all_dfs = []
    for file in sorted(csv_files):
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path, parse_dates=['ParsedDateTime'])
        all_dfs.append(df)
        print(f"   ✅ Loaded: {file} — Rows: {len(df)}")

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # 🛑 Removed sorting to reduce processing and avoid kernel crashes

    # Save merged result
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Merged file saved: {output_file} — Total rows: {len(merged_df)}")


# === Example usage
if __name__ == "__main__":
    input_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Aggregated_no_sex"
    output_file = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Aggregated_no_sex/aggregated_illness_2019.csv"
    year = 2019

    merge_aggregated_age_files(input_dir, output_file, year)