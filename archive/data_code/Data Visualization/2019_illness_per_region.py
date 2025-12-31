import pandas as pd
import matplotlib.pyplot as plt
import os

# Load illness data
file_path = '/Users/jay/Desktop/Processed Data/Illness Data/Aggregated/aggregated_illness_2019.csv'
df = pd.read_csv(file_path)

# Ensure Date column is datetime
df['ParsedDateTime'] = pd.to_datetime(df['ParsedDateTime'])

# Group and count
illness_counts = df.groupby(['RegionCode', 'ParsedDateTime', 'IllnessCode']).size().reset_index(name='Count')

# Create output directory
output_dir = '/Users/jay/Desktop/Illness_Plots_By_Illness'
os.makedirs(output_dir, exist_ok=True)

# Unique region codes
region_codes = illness_counts['RegionCode'].unique()

for region in region_codes:
    region_data = illness_counts[illness_counts['RegionCode'] == region]
    # Define desired prefixes
    target_prefixes = ('A', 'B', 'E', 'I', 'J', 'L', 'S', 'T', 'G','F' )  # for example: infectious diseases (A), neoplasms (B)
    
    # Filter only rows where illness code starts with any of the prefixes
    filtered_data = region_data[region_data['IllnessCode'].str.startswith(target_prefixes)]
    
    # Get unique illness codes from the filtered data
    illness_codes = filtered_data['IllnessCode'].unique()


    for illness in illness_codes:
        illness_data = region_data[region_data['IllnessCode'] == illness].sort_values('ParsedDateTime')

        if illness_data.empty:
            continue  # skip empty
        
        plt.figure(figsize=(12, 6))
        plt.plot(illness_data['ParsedDateTime'], illness_data['Count'], label=f'Illness {illness}', color='red')
        plt.title(f'Illness Code {illness} in Region {region} - 2019')
        plt.xlabel('ParsedDateTime')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()

        # Save figure
        filename = f'Illness_{illness}_Region_{region}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()

print("✅ Done generating individual illness plots per region.")
