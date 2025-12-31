import pandas as pd
import os

# Set up paths
climate_dir = "/Users/jay/Desktop/Processed Data/Merged Data"
illness_dir = "/Users/jay/Desktop/Processed Data/Illness Data/Aggregated"
output_dir = "/Users/jay/Desktop/Processed Data/Merged All Years"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Years to include
years = ['2019', '2020', '2021', '2022', '2023']

# Initialize empty DataFrames for both datasets
all_climate = pd.DataFrame()
all_illness = pd.DataFrame()

# Merge all climate+atmo CSVs
for year in years:
    path = os.path.join(climate_dir, f"{year}_merged.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        all_climate = pd.concat([all_climate, df], ignore_index=True)

# Merge all illness CSVs
for year in years:
    path = os.path.join(illness_dir, f"aggregated_illness_{year}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        all_illness = pd.concat([all_illness, df], ignore_index=True)

# Save the merged datasets
climate_out_path = os.path.join(output_dir, "climate_atmo_all_years.csv")
illness_out_path = os.path.join(output_dir, "illness_all_years.csv")

all_climate.to_csv(climate_out_path, index=False)
all_illness.to_csv(illness_out_path, index=False)

print("Merged climate+atmo shape:", all_climate.shape)
print("Merged illness shape:", all_illness.shape)
print("Saved to:")
print(" -", climate_out_path)
print(" -", illness_out_path)
