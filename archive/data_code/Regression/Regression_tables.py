import os
import pandas as pd

# === CONFIG ===
base_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results"
output_dir = "/Users/jay/Desktop/Illness Prediction/Tables"
os.makedirs(output_dir, exist_ok=True)

# === Helper function to extract R2s ===
def extract_r2_from_dir(dir_path):
    r2_dict = {}
    if not os.path.exists(dir_path):
        return r2_dict

    for illness in os.listdir(dir_path):
        illness_path = os.path.join(dir_path, illness)
        if not os.path.isdir(illness_path):
            continue

        for file in os.listdir(illness_path):
            if file.endswith(".csv"):
                file_path = os.path.join(illness_path, file)
                try:
                    df = pd.read_csv(file_path)
                    if "R²_Mean" in df.columns:
                        r2_dict[illness] = df["R²_Mean"].values[0]
                    elif "Best_R²_CV" in df.columns:
                        r2_dict[illness] = df["Best_R²_CV"].values[0]

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return r2_dict

# === Task 1: RF vs XGB (SS_lag3) ===
rf_r2 = extract_r2_from_dir(os.path.join(base_dir, "RF with CV", "SS_lag3_KoreaWide"))
xgb_r2 = extract_r2_from_dir(os.path.join(base_dir, "XGB with CV", "SS_lag3_KoreaWide"))

df1 = pd.DataFrame({"Illness": sorted(set(rf_r2.keys()).union(xgb_r2.keys()))})
df1["RF_R2"] = df1["Illness"].map(rf_r2)
df1["XGB_R2"] = df1["Illness"].map(xgb_r2)
df1.to_csv(os.path.join(output_dir, "RF_vs_XGB_SS_lag3.csv"), index=False)

# === Task 2: XGB BE vs FS ===
be_r2 = extract_r2_from_dir(os.path.join(base_dir, "XGB with CV", "BE_lag3_KoreaWide"))
fs_r2 = extract_r2_from_dir(os.path.join(base_dir, "XGB with CV", "FS_lag3_KoreaWide"))

df2 = pd.DataFrame({"Illness": sorted(set(be_r2.keys()).union(fs_r2.keys()))})
df2["BE_R2"] = df2["Illness"].map(be_r2)
df2["FS_R2"] = df2["Illness"].map(fs_r2)
df2.to_csv(os.path.join(output_dir, "XGB_CV_BE_vs_FS.csv"), index=False)

# === Task 3: XGB CV++ vs XGB CV (FS only) ===
plus_r2 = extract_r2_from_dir(os.path.join(base_dir, "XGB GridSearch ++", "FS_lag3_KoreaWide"))
std_r2 = extract_r2_from_dir(os.path.join(base_dir, "XGB with CV", "FS_lag3_KoreaWide"))

df3 = pd.DataFrame({"Illness": sorted(set(plus_r2.keys()).union(std_r2.keys()))})
df3["XGB_GridSearch++_R2"] = df3["Illness"].map(plus_r2)
df3["XGB_CV_R2"] = df3["Illness"].map(std_r2)
df3.to_csv(os.path.join(output_dir, "XGB_GridSearch++_vs_CV_FS.csv"), index=False)

print("✅ All comparison tables saved to:", output_dir)