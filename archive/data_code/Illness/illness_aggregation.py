import pandas as pd
import os

def aggregate_illness_data(
    input_dir: str,
    output_dir: str,
    years: list
):
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        input_file = os.path.join(input_dir, f"illness_{year}_cleaned.csv")
        output_file = os.path.join(output_dir, f"aggregated_illness_{year}.csv")
        
        print(f"\nProcessing {year}...")

        try:
            df = pd.read_csv(input_file)

            # Group by date, region, illness, age group, sex
            grouped = df.groupby(
                ['ParsedDateTime', 'RegionCode', 'IllnessCode', 'AgeCode', 'SexCode']
            ).size().reset_index(name='CaseCount')

            grouped.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"Saved aggregated file to: {output_file}")

        except Exception as e:
            print(f"Failed to process {year}: {e}")

# Example usage
if __name__ == "__main__":
    input_dir = "/Users/jay/Desktop/Processed Data/Illness Data"
    output_dir = "/Users/jay/Desktop/Processed Data/Illness Data/Aggregated"
    years = [2019, 2020, 2021, 2022, 2023]

    aggregate_illness_data(input_dir, output_dir, years)
