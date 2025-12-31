import pandas as pd
import numpy as np

# Load illness data
df = pd.read_csv('/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/aggregated_illness_2019.csv', parse_dates=['ParsedDateTime'])

# Filter by Region
df = df[df['RegionCode'] == 11]

# Group by illness code per day
daily_counts = df.groupby(['ParsedDateTime', 'IllnessCode']).size().unstack(fill_value=0)

# Calculate metrics
metrics = pd.DataFrame({
    'Total': daily_counts.sum(),
    'StdDev': daily_counts.std(),
    'Max': daily_counts.max(),
    'Mean': daily_counts.mean(),
})

metrics['PeakToMean'] = metrics['Max'] / metrics['Mean']
metrics = metrics.sort_values(by='StdDev', ascending=False)

# Save for review
metrics.to_csv('illness_volatility_metrics.csv')