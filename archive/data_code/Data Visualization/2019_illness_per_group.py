import pandas as pd
import matplotlib.pyplot as plt
import os

# Load illness data
df = pd.read_csv('/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/Aggregated/aggregated_illness_2019.csv', parse_dates=['ParsedDateTime'])

# Filter for Region 11
df = df[df['RegionCode'] == 11]

# Illness groups to consider
groups = ['A', 'B', 'E', 'I', 'J', 'L', 'S', 'T', 'G','F']

# Output folder
output_folder = '/Users/jay/Desktop/Illness Prediction/Plots/illness_plots_by_group_2019'
os.makedirs(output_folder, exist_ok=True)

# Group and plot
for group_prefix in groups:
    group_df = df[df['IllnessCode'].str.startswith(group_prefix)]

    if group_df.empty:
        continue

    # Count occurrences per illness per day
    daily_counts = group_df.groupby(['ParsedDateTime', 'IllnessCode']).size().unstack(fill_value=0)

    # Plot
    plt.figure(figsize=(12, 6))
    for illness_code in daily_counts.columns:
        plt.plot(daily_counts.index, daily_counts[illness_code], label=illness_code)

    plt.title(f"Region 11 - Illness Group {group_prefix}")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.tight_layout()

    # Save
    plt.savefig(os.path.join(output_folder, f"region_11_group_{group_prefix}.png"))
    plt.close()
