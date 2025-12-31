import pandas as pd

# Paths
env_2018_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data/2018_merged.csv"  # already last 7 days only
env_2019_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data/2019_merged.csv"
output_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data/2019_merged_with_lag.csv"

# Load
env_2018 = pd.read_csv(env_2018_path, parse_dates=['DateTime'])
env_2019 = pd.read_csv(env_2019_path, parse_dates=['DateTime'])

# Combine
combined = pd.concat([env_2018, env_2019], ignore_index=True)
combined.sort_values(by=['DateTime', 'RegionCode'], inplace=True)

# Save
combined.to_csv(output_path, index=False)
print(f"✅ Created: {output_path}")