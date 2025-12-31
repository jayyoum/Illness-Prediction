import pandas as pd
import os

climate_dir = "/Users/jay/Desktop/Processed Data/Climate Data"
atmospheric_dir = "/Users/jay/Desktop/Processed Data/Atmospheric Data"
output_dir = "/Users/jay/Desktop/Processed Data/Merged Data"
os.makedirs(output_dir, exist_ok=True)

years = ['2019', '2020', '2021', '2022', '2023']

for year in years:
    print(f"Merging year {year}...")

    # Load each dataset
    climate_file = os.path.join(climate_dir, f"{year}.csv")
    atm_file = os.path.join(atmospheric_dir, f"{year}.csv")
    
    climate_df = pd.read_csv(climate_file)
    atm_df = pd.read_csv(atm_file)

    # Merge on RegionCode and DateTime
    merged_df = pd.merge(climate_df, atm_df, on=['RegionCode', 'DateTime'], how='inner')

    # Save merged file
    merged_df.to_csv(os.path.join(output_dir, f"{year}_merged.csv"), index=False)

print("✅ All years merged successfully.")
