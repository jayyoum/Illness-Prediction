import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV
file_path = '/Users/jay/Desktop/Processed Data/Climate Data/2019.csv'
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Get region codes
region_codes = df['RegionCode'].unique()

# Get all variable names starting from column index 8
variable_columns = df.columns[8:]

# Create base output directory
output_base_dir = '/Users/jay/Desktop/Region_Plots_All_Variables_2019'
os.makedirs(output_base_dir, exist_ok=True)

# Loop through each region and variable
for region in region_codes:
    region_df = df[df['RegionCode'] == region].sort_values(by='DateTime')
    
    # Make a subfolder for this region
    region_dir = os.path.join(output_base_dir, f'Region_{region}')
    os.makedirs(region_dir, exist_ok=True)

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
        plt.savefig(os.path.join(region_dir, plot_filename))
        plt.close()

print(f"✅ Plots saved for {len(region_codes)} regions and {len(variable_columns)} variables.")
