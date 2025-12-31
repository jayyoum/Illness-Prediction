import os
import pandas as pd
from datetime import timedelta


def fix_hour_24(series: pd.Series) -> pd.Series:
    fixed_dates = []

    for val in series:
        try:
            val = str(val)
            if val[-2:] == '24':
                # Move to next day at 00:00
                base_date = pd.to_datetime(val[:-2], format='%Y%m%d')
                fixed_dates.append(base_date + timedelta(days=1))
            else:
                fixed_dates.append(pd.to_datetime(val, format='%Y%m%d%H'))
        except Exception:
            fixed_dates.append(pd.NaT)

    return pd.Series(fixed_dates)


def convert_and_save_sheets(
    input_dir: str,
    output_dir: str,
    year_date_formats: dict
):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".xlsx"):
            year = file.split('.')[0]
            file_path = os.path.join(input_dir, file)

            print(f"\nReading file: {file_path}")
            xl = pd.ExcelFile(file_path)

            # Get appropriate parser for the year
            if year in year_date_formats:
                parser = year_date_formats[year]['parser']
            else:
                parser = year_date_formats['default']['parser']

            for sheet_name in xl.sheet_names:
                print(f"  Processing sheet: {sheet_name}")
                df = xl.parse(sheet_name)
                pollutant_cols = ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']
                df.dropna(subset=pollutant_cols, how='all', inplace=True)

                # Make sure it has at least 5 columns
                if df.shape[1] < 5:
                    print(f"    Skipped: not enough columns in {sheet_name}")
                    continue

                try:
                    # Convert the 5th column to standardized DateTime
                    date_col = df.columns[4]
                    df['DateTime'] = parser(df[date_col].astype(str))

                    # Save as CSV
                    safe_sheet = sheet_name.replace(' ', '_').replace('/', '-')
                    out_filename = f"{year}_{safe_sheet}.csv"
                    out_path = os.path.join(output_dir, out_filename)
                    df.to_csv(out_path, index=False)

                    print(f"    Saved: {out_path}")
                except Exception as e:
                    print(f"    Failed to process {sheet_name}: {str(e)}")

# Main execution block
if __name__ == "__main__":
    input_directory = "/Users/jay/Desktop/Raw Data/Atmospheric Data"
    output_directory = "/Users/jay/Desktop/Raw Data/Atmospheric CSV"

    year_date_formats = {
        '2023': {
            'parser': lambda s: fix_hour_24(s.str.replace('-', ''))
            },
        'default': {
            'parser': fix_hour_24
            }
        }


    convert_and_save_sheets(
        input_dir=input_directory,
        output_dir=output_directory,
        year_date_formats=year_date_formats
    )
