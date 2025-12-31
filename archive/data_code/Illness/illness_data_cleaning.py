import pandas as pd
import os

def clean_illness_data_add_datetime_column(file_path, output_path):
    try:
        # Try reading with multiple encodings
        for enc in ["utf-8", "euc-kr", "cp949"]:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                print(f"✅ Successfully loaded file with encoding: {enc}")
                break
            except Exception:
                continue
        else:
            raise ValueError("❌ Failed to read the file with all common encodings.")

        print(f"Initial data shape: {df.shape}")
        print("Initial columns:", df.columns.tolist())

        # Define possible formats (new format for 2022–2023, old format for 2021 and before)
        if '요양개시일자' in df.columns:
            column_map = {
                '요양개시일자': 'RawDate',
                '연령대코드': 'AgeCode',
                '성별코드': 'SexCode',
                '시도코드': 'RegionCode',
                '주상병코드': 'IllnessCode'
            }
        elif 'RECU_FR_DT' in df.columns:
            column_map = {
                'RECU_FR_DT': 'RawDate',
                'AGE_GROUP': 'AgeCode',
                'SEX': 'SexCode',
                'SIDO': 'RegionCode',
                'MAIN_SICK': 'IllnessCode'
            }
        else:
            raise ValueError("❌ Column structure not recognized. Cannot proceed.")

        # Rename columns to standardized names
        df.rename(columns=column_map, inplace=True)

        # Convert date to datetime
        df['ParsedDateTime'] = pd.to_datetime(df['RawDate'].astype(str), errors='coerce')
        df = df.dropna(subset=['ParsedDateTime'])

        # Filter invalid ages/sexes
        df = df[df['AgeCode'].between(1, 18)]
        df = df[df['SexCode'].isin([1, 2])]

        # Keep only relevant columns
        df = df[['ParsedDateTime', 'RegionCode', 'AgeCode', 'SexCode', 'IllnessCode']]
        print(f"Final cleaned shape: {df.shape}")
        print("Columns after cleaning:", df.columns.tolist())

        # Save cleaned data
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ Cleaned data saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error during processing: {e}")

# === Example Usage ===
input_file = "/Users/jay/Desktop/Illness Prediction/Raw Data/Illness data/2019.csv"
output_file = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness data/illness_2019_cleaned.csv"
clean_illness_data_add_datetime_column(input_file, output_file)
