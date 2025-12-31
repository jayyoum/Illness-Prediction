import os
import pandas as pd

# === CONFIG ===
xgb_plus_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/XGB GridSearch ++/FS_lag3_KoreaWide"
output_path = "/Users/jay/Desktop/Illness Prediction/Tables/XGB_CVpp_BestParams_Table.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === Collect parameters ===
all_params = []

for illness in os.listdir(xgb_plus_dir):
    illness_path = os.path.join(xgb_plus_dir, illness)
    if not os.path.isdir(illness_path):
        continue

    for file in os.listdir(illness_path):
        if file.endswith(".csv"):
            file_path = os.path.join(illness_path, file)
            try:
                df = pd.read_csv(file_path)
                row = {"Illness": illness}
                for col in df.columns:
                    if col.startswith("Param_"):
                        row[col.replace("Param_", "")] = df[col].iloc[0]
                all_params.append(row)
            except Exception as e:
                print(f"Error in {file_path}: {e}")

# === Save to CSV ===
params_df = pd.DataFrame(all_params)
params_df.to_csv(output_path, index=False)
print(f"✅ Saved parameter table to: {output_path}")