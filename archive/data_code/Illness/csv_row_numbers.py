import os
import pandas as pd

folder_path = "/Users/jay/Desktop/Raw Data/Illness Data"

print(f"Checking CSV files in: {folder_path}")

for filename in sorted(os.listdir(folder_path)):
    if filename.lower().endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path, encoding="euc-kr")  # or try "cp949" if this fails
            print(f"{filename}: {len(df):,} rows")
        except Exception as e:
            print(f"{filename}: Failed to read - {e}")
