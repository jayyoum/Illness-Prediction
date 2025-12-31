import os
import pandas as pd
import json

def load_station_to_region(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_and_process_csv(file_path: str, station_to_region: dict) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # Convert datetime and sort
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.dropna(subset=['DateTime'])  # Drop rows where datetime parsing failed
    df = df.sort_values('DateTime')

    # Set DateTime index for interpolation
    df.set_index('DateTime', inplace=True)

    # Interpolate all pollutant columns
    pollutant_cols = ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']  # adjust if more exist
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method='time')

    df.reset_index(inplace=True)

    # Add RegionCode
    df['RegionCode'] = df['측정소코드'].astype(str).map(station_to_region)

    # Drop rows with no region or all pollutants missing
    df.dropna(subset=['RegionCode'], inplace=True)
    df.dropna(subset=pollutant_cols, how='all', inplace=True)

    # Extract date for daily grouping
    df['DateTime'] = df['DateTime'].dt.date

    return df

def aggregate_by_region_and_day(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ['RegionCode', 'DateTime']
    exclude_columns = ['측정소코드', '측정일시']
    df = df.drop(columns=[col for col in exclude_columns if col in df.columns])
    agg_df = df.groupby(group_cols).mean(numeric_only=True).reset_index()
    return agg_df

def process_all_data(input_dir: str, station_json: str, output_dir: str):
    station_to_region = load_station_to_region(station_json)

    os.makedirs(output_dir, exist_ok=True)

    # Sort files by year
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])

    year_to_dfs = {}

    for file in all_files:
        year = file[:4]  # Assumes file name starts with year like "2023_1월.csv"
        file_path = os.path.join(input_dir, file)
        df = read_and_process_csv(file_path, station_to_region)

        if year not in year_to_dfs:
            year_to_dfs[year] = []
        year_to_dfs[year].append(df)

    # Combine, aggregate, and save
    for year, dfs in year_to_dfs.items():
        combined = pd.concat(dfs, ignore_index=True)
        daily_region_df = aggregate_by_region_and_day(combined)
        output_path = os.path.join(output_dir, f"{year}_region_daily.csv")
        daily_region_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ Saved: {output_path}")
        
        
if __name__ == "__main__":
    input_dir = "/Users/jay/Desktop/Raw Data/Atmospheric CSV"  # your monthly CSVs
    station_json = "/Users/jay/Desktop/station_to_region.json"
    output_dir = "/Users/jay/Desktop/Processed Regional Atmospheric"

    process_all_data(input_dir, station_json, output_dir)


