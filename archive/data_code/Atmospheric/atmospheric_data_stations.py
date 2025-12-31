import os
import pandas as pd
from collections import defaultdict

def compare_station_codes(input_dir: str):
    year_station_map = defaultdict(set)

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_dir, filename)

            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                year = filename.split('_')[0]  # Extract year from filename
                if '측정소코드' in df.columns:
                    station_codes = set(df['측정소코드'].dropna().astype(str).unique())
                    year_station_map[year].update(station_codes)
                else:
                    print(f"[Warning] '측정소코드' not found in {filename}")

            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # Print and compare station codes
    all_years = sorted(year_station_map.keys())
    print("\n🛰️ Unique Station Counts by Year:")
    for year in all_years:
        print(f"  {year}: {len(year_station_map[year])} stations")

    print("\n📌 Differences Between Years:")
    for i in range(len(all_years)):
        for j in range(i + 1, len(all_years)):
            year1, year2 = all_years[i], all_years[j]
            diff1 = year_station_map[year1] - year_station_map[year2]
            diff2 = year_station_map[year2] - year_station_map[year1]

            if diff1:
                print(f"  {year1} has extra stations not in {year2}: {sorted(diff1)}")
            if diff2:
                print(f"  {year2} has extra stations not in {year1}: {sorted(diff2)}")

# === USAGE ===
if __name__ == "__main__":
    input_directory = "/Users/jay/Desktop/Raw Data/Atmospheric CSV"  # Update if needed
    compare_station_codes(input_directory)