import os
import pandas as pd

folder_path = "/Users/jay/Desktop/Raw Data/Illness Data/2019"
output_path = "/Users/jay/Desktop/Raw Data/Illness Data/illness_2019.csv"

dfs = []
for filename in sorted(os.listdir(folder_path)):
    if filename.lower().endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path, encoding="euc-kr")  # or "cp949"
            dfs.append(df)
            print(f"Loaded: {filename} ({len(df):,} rows)")
        except Exception as e:
            print(f"Skipped: {filename} - {e}")

if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Merged CSV saved to: {output_path}")
    print(f"Total rows: {len(merged_df):,}")
else:
    print("❌ No dataframes were merged.")
