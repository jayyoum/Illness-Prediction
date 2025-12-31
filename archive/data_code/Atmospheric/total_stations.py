import os
import pandas as pd

def extract_unique_station_codes(directory: str, station_col: str = '측정소코드'):
    all_codes = set()

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path)
                if station_col in df.columns:
                    unique_codes = df[station_col].dropna().unique()
                    all_codes.update(unique_codes)
                else:
                    print(f"Warning: '{station_col}' not found in {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    all_codes = sorted(list(all_codes))
    print(f"Total unique station codes found: {len(all_codes)}")
    print("Sample station codes:", all_codes[:10])

    # Optional: Save to text file
    with open('all_station_codes.txt', 'w') as f:
        for code in all_codes:
            f.write(str(code) + '\n')

    return all_codes

# Example usage
if __name__ == "__main__":
    csv_directory = "/Users/jay/Desktop/Raw Data/Atmospheric CSV"
    extract_unique_station_codes(csv_directory)
