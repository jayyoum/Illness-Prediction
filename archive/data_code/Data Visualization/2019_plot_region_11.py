import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV
file_path = '/Users/jay/Desktop/Illness Prediction/Processed Data/Climate Data/2019.csv'
df = pd.read_csv(file_path)

# Convert 'DateTime' column to datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Filter for Region 11 only
region = 11
region_df = df[df['RegionCode'] == region].sort_values(by='DateTime')

# Get all variable names starting from column index 7
variable_columns = df.columns[6:]

# Create output directory for Region 11
output_dir = f'/Users/jay/Desktop/Illness Prediction/Plots/Region_11_Plots_2019'
os.makedirs(output_dir, exist_ok=True)

# Loop through each variable and plot
for var in variable_columns:
    plt.figure(figsize=(12, 6))
    plt.plot(region_df['DateTime'], region_df[var], label=var, color='blue')
    plt.title(f'{var} Over Time - Region {region} (2019)')
    plt.xlabel('DateTime')
    plt.ylabel(var)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plot_filename = f'{var}_Region_{region}.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()

print(f"✅ Plots saved for Region {region} with {len(variable_columns)} variables.")