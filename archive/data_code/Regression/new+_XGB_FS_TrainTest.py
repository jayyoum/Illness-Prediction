import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import product

# === USER CONFIGURATION ===
illness_name = "Acute_upper_respiratory_infections"
lag_number = 0
excluded_vars = {"MaxWindDir", "Year_x", "Year_y"}

# === PATH CONFIGURATION (auto-generated) ===
base_input_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental/Grouped"
base_output_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/New +/TT"

input_filename = f"merged_data_{illness_name}_lag{lag_number}.csv"
output_filename = f"XGB_FS_{illness_name}_lag{lag_number}_train_test.csv"

input_path = os.path.join(base_input_dir, input_filename)
output_folder = os.path.join(base_output_dir, illness_name)
output_path = os.path.join(output_folder, output_filename)

# ✅ Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# === LOAD AND PREPARE DATA ===
df = pd.read_csv(input_path)
excluded_cols = {"RegionCode", "Month", "CaseCount", "ParsedDateTime", "IllnessName", "DateTime"}
env_features = [
    col for col in df.columns
    if col not in excluded_cols
    and col not in excluded_vars
    and pd.api.types.is_numeric_dtype(df[col])
]

X_full = df[env_features]
y_full = df["CaseCount"]
valid_rows = X_full.notna().all(axis=1) & y_full.notna()
X_full = X_full[valid_rows]
y_full = y_full[valid_rows]

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=42)

# === MANUAL FORWARD SELECTION + GRID SEARCH ===
param_grid = {
    'n_estimators': [300, 600, 900],
    'learning_rate': [0.03, 0.05],
    'max_depth': [3, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.75],
    'gamma': [0.1]
}

selected_vars = []
remaining_vars = list(X_train.columns)
best_r2 = -np.inf
best_model = None
best_params = None

while remaining_vars:
    best_trial_r2 = best_r2
    best_candidate = None
    best_candidate_params = None

    for candidate in remaining_vars:
        trial_vars = selected_vars + [candidate]
        X_train_subset = X_train[trial_vars]
        X_test_subset = X_test[trial_vars]

        for params in product(*param_grid.values()):
            param_combo = dict(zip(param_grid.keys(), params))
            model = XGBRegressor(**param_combo, objective="reg:squarederror", random_state=42, verbosity=0)
            model.fit(X_train_subset, y_train)
            y_pred = model.predict(X_test_subset)
            r2 = r2_score(y_test, y_pred)

            if r2 > best_trial_r2:
                best_trial_r2 = r2
                best_candidate = candidate
                best_candidate_params = param_combo

    if best_trial_r2 > best_r2:
        selected_vars.append(best_candidate)
        remaining_vars.remove(best_candidate)
        best_r2 = best_trial_r2
        best_params = best_candidate_params
        print(f"✅ Added: {best_candidate} | R² = {best_r2:.4f}")
    else:
        print("⛔ No further improvement.")
        break

# === FINAL MODEL + FEATURE IMPORTANCE ===
final_model = XGBRegressor(**best_params, objective="reg:squarederror", random_state=42, verbosity=0)
final_model.fit(X_full[selected_vars], y_full)
importances = final_model.feature_importances_

# === SAVE RESULTS ===
results = {
    "Best_R2_Test": round(best_r2, 4),
    **{f"Param_{k}": v for k, v in best_params.items()}
}
for var, imp in zip(selected_vars, importances):
    results[f"Importance_{var}"] = round(imp, 4)

pd.DataFrame([results]).to_csv(output_path, index=False)
print(f"\n💾 Results saved to: {output_path}")
