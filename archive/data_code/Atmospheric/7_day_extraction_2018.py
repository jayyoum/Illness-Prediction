import pandas as pd
import os

def extract_last_7_days_from_xlsx(
    source_xlsx_path: str,
    output_xlsx_path: str
):
    print(f"📥 Reading from: {source_xlsx_path}")
    xl = pd.ExcelFile(source_xlsx_path)

    # Create a writer to save new workbook
    with pd.ExcelWriter(output_xlsx_path, engine='xlsxwriter') as writer:
        for sheet_name in xl.sheet_names:
            print(f"🔍 Processing sheet: {sheet_name}")
            try:
                df = xl.parse(sheet_name)

                # Ensure at least 5 columns
                if df.shape[1] < 5:
                    print(f"  ⚠️ Skipping {sheet_name} — too few columns.")
                    continue

                # Use column 5 (index 4) for datetime
                date_col = df.columns[4]
                datetime_series = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d%H', errors='coerce')

                # Filter to Dec 25–31, 2018
                mask = (datetime_series >= "2018-12-25") & (datetime_series <= "2018-12-31 23:59:59")
                filtered_df = df[mask]

                if not filtered_df.empty:
                    filtered_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  ✅ Saved filtered sheet: {sheet_name}")
                else:
                    print(f"  ⚠️ No data in range for sheet: {sheet_name}")

            except Exception as e:
                print(f"  ❌ Failed on {sheet_name}: {e}")

    print(f"\n✅ Output saved to: {output_xlsx_path}")
    
if __name__ == "__main__":
    source_path = "/Users/jay/Desktop/Illness Prediction/Raw Data/Atmospheric Data/2018.xlsx"
    output_path = "/Users/jay/Desktop/Illness Prediction/Raw Data/Atmospheric Data/2018_LAST7DAYS_ONLY.xlsx"
    
    extract_last_7_days_from_xlsx(source_path, output_path)