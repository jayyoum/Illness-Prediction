import os
import pandas as pd

# === CONFIG ===
base_folder = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/Selected Variables"
output_csv = os.path.join(base_folder, "intersected_variables.csv")

all_common_vars = None  # Will hold the intersection across illnesses and lags

# === Loop through illness folders ===
for illness_folder in os.listdir(base_folder):
    illness_path = os.path.join(base_folder, illness_folder)
    if not os.path.isdir(illness_path):
        continue

    lag_files = [f for f in os.listdir(illness_path) if f.startswith("forward_selection_lag")]
    if len(lag_files) < 2:
        continue  # Only include illnesses with both lag3 and lag4

    lag_var_sets = []
    for file in lag_files:
        df = pd.read_csv(os.path.join(illness_path, file))
        variables = set(df["Selected_Variables"].dropna().astype(str))
        lag_var_sets.append(variables)

    # Find intersection across all lags for this illness
    common_vars = set.intersection(*lag_var_sets)

    if all_common_vars is None:
        all_common_vars = common_vars
    else:
        all_common_vars = all_common_vars.intersection(common_vars)

# === Save final intersected variables to CSV ===
final_df = pd.DataFrame({"Common_Variables": sorted(all_common_vars)})
final_df.to_csv(output_csv, index=False)

print(f"\n✅ Saved global intersected variables to:\n{output_csv}")