import pandas as pd
import numpy as np
import os

# Load illness data
filepath = '/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/aggregated_illness_2019.csv'
df = pd.read_csv(filepath, parse_dates=['ParsedDateTime'])

# Filter by RegionCode
df = df[df['RegionCode'] == 11]

# Extract illness group prefix (first letter)
df['IllnessGroup'] = df['IllnessCode'].astype(str).str[0]

# Group by ParsedDateTime and IllnessCode to get daily counts
daily_counts = df.groupby(['ParsedDateTime', 'IllnessCode']).size().unstack(fill_value=0)

# Create output folder if needed
output_folder = '/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Illness Volatility Rankings'
os.makedirs(output_folder, exist_ok=True)

# Group illness codes by prefix and calculate stats for each group
for group_prefix in df['IllnessGroup'].unique():
    group_codes = [code for code in daily_counts.columns if str(code).startswith(group_prefix)]
    if not group_codes:
        continue
    
    group_data = daily_counts[group_codes]
    
    # Calculate metrics
    metrics = pd.DataFrame({
        'Total': group_data.sum(),
        'StdDev': group_data.std(),
        'Max': group_data.max(),
        'Mean': group_data.mean(),
    })
    metrics['PeakToMean'] = metrics['Max'] / metrics['Mean']
    
    # Sort by StdDev
    metrics = metrics.sort_values(by='StdDev', ascending=False)
    
    # Save CSV
    group_csv_path = os.path.join(output_folder, f'illness_group_{group_prefix}_metrics.csv')
    metrics.to_csv(group_csv_path)

print("✅ Illness volatility metrics saved per group.")
