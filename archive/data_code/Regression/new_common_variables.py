import os
import pandas as pd

# === CONFIG ===
base_folder = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/Selected Variables/FS_grouped"

# === Loop through all lags (0 to 7) ===
for target_lag in range(8):
    print(f"\n🔍 Processing lag {target_lag}...")
    all_common_vars = None

    for illness_folder in os.listdir(base_folder):
        illness_path = os.path.join(base_folder, illness_folder)
        if not os.path.isdir(illness_path):
            continue

        lag_filename = f"forward_selection_lag{target_lag}.csv"
        lag_filepath = os.path.join(illness_path, lag_filename)

        if not os.path.exists(lag_filepath):
            print(f"⚠️ Missing for {illness_folder}, skipping.")
            continue

        df = pd.read_csv(lag_filepath)
        variables = set(df["Selected_Variables"].dropna().astype(str))

        if all_common_vars is None:
            all_common_vars = variables
        else:
            all_common_vars = all_common_vars.intersection(variables)

    # === Save intersected variables for this lag ===
    output_csv = os.path.join(base_folder, f"intersected_variables_lag{target_lag}.csv")
    final_df = pd.DataFrame({"Common_Variables": sorted(all_common_vars)})
    final_df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv} | Variables: {len(final_df)}")