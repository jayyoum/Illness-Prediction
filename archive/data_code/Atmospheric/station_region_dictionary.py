import os
import pandas as pd

def extract_station_codes(input_dir, output_csv):
    station_set = set()

    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(input_dir, file)
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # Check if both columns exist
                if '측정소코드' in df.columns and '지역' in df.columns:
                    for _, row in df[['측정소코드', '지역']].drop_duplicates().iterrows():
                        station_set.add((row['측정소코드'], row['지역']))
                else:
                    print(f"Skipping {file} - missing required columns")
            except Exception as e:
                print(f"Error reading {file}: {e}")

    # Convert set to DataFrame
    station_df = pd.DataFrame(sorted(station_set), columns=['StationCode', 'Location'])

    # Save to CSV
    station_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n✅ Extracted {len(station_df)} unique stations. Saved to {output_csv}")

# === Usage ===
if __name__ == "__main__":
    input_directory = "/Users/jay/Desktop/Raw Data/Atmospheric CSV"
    output_file = "/Users/jay/Desktop/station_code_location_pairs.csv"
    extract_station_codes(input_directory, output_file)
