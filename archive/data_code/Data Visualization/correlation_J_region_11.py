import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==== Load data ====
illness_path = '/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/aggregated_illness_2019.csv'
climate_path = '/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data/2019_merged.csv'

illness_df = pd.read_csv(illness_path)
climate_df = pd.read_csv(climate_path)

# ==== Preprocess ====
# Fix column names
illness_df['ParsedDateTime'] = pd.to_datetime(illness_df['ParsedDateTime'])
climate_df['DateTime'] = pd.to_datetime(climate_df['DateTime'])

# Remove spaces in 'Region Code'
illness_df.rename(columns={'Region Code': 'RegionCode'}, inplace=True)
climate_df.rename(columns={'Region Code': 'RegionCode'}, inplace=True)

# ==== Filter to respiratory illnesses (J-codes) ====
illness_df = illness_df[illness_df['IllnessCode'].str.startswith('J')]

# Count illnesses per region per day
illness_counts = (
    illness_df.groupby(['ParsedDateTime', 'RegionCode'])
    .size()
    .reset_index(name='RespiratoryIllnessCount')
)

# ==== Prepare climate data ====
# Only keep relevant columns (col 7 onward are variables)
climate_vars = climate_df.columns[7:]
climate_selected = climate_df[['DateTime', 'RegionCode'] + list(climate_vars)]

# ==== Merge datasets ====
merged_df = pd.merge(climate_selected, illness_counts, left_on=['DateTime', 'RegionCode'],right_on=['ParsedDateTime', 'RegionCode'], how='inner')

# ==== Filter for one region (e.g., Region 11) ====
region_df = merged_df[merged_df['RegionCode'] == 11]

# ==== Compute correlations ====
# === Define illness and environmental variables ===
illness_vars = ['RespiratoryIllnessCount']
env_vars = [col for col in region_df.columns if col not in illness_vars + ['DateTime', 'RegionCode']]

# === Compute correlation between environmental and illness variables ===
corr = region_df[env_vars + illness_vars].corr().loc[env_vars, illness_vars]

# ==== Plot heatmap ====
plt.figure(figsize=(12, len(env_vars) * 0.3 + 2))  # dynamically scale height
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={'label': 'Correlation Coefficient'}
)

plt.title("Correlation Between Environmental Variables and Respiratory Illnesses (Region 11)")
plt.xlabel("Illness Variables")
plt.ylabel("Environmental Variables")
plt.tight_layout()
plt.show()
