import pandas as pd
import matplotlib.pyplot as plt
import os

# Load illness data
file_path = '/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/aggregated_illness_2019.csv'
df = pd.read_csv(file_path)

# Ensure Date column is datetime
df['ParsedDateTime'] = pd.to_datetime(df['ParsedDateTime'])

# Filter for illness J638
df = df[df['IllnessCode'] == 'J704']

# Group and count
illness_counts = df.groupby(['RegionCode', 'ParsedDateTime']).size().reset_index(name='Count')

# Create output directory
output_dir = '/Users/jay/Desktop/Illness Prediction/Plots/J704 PLOT'
os.makedirs(output_dir, exist_ok=True)

# Plot J638 for each region
region_codes = illness_counts['RegionCode'].unique()

for region in region_codes:
    region_data = illness_counts[illness_counts['RegionCode'] == region].sort_values('ParsedDateTime')

    if region_data.empty:
        continue  # skip empty

    plt.figure(figsize=(12, 6))
    plt.plot(region_data['ParsedDateTime'], region_data['Count'], label='Illness J704', color='red')
    plt.title(f'Illness J704 in Region {region} - 2019')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    filename = f'Illness_J704_Region_{region}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()

print("✅ Done plotting illness J704 for all regions.")