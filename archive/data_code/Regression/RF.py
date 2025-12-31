import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# === USER CONFIG ===
lag_label = "lag4"
intersected_var_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/Selected Variables/intersected_variables_BE.csv"
merged_data_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental"
output_base_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/RF on Intersected/BE_lag3"

# === LOAD SELECTED VARIABLES ===
var_df = pd.read_csv(intersected_var_path)
selected_vars = var_df.iloc[:, 0].dropna().astype(str).tolist()
print(f"✅ Using variables: {selected_vars}")

# === LOOP THROUGH MERGED FILES ===
for filename in os.listdir(merged_data_dir):
    if not filename.endswith(f"_{lag_label}.csv"):
        continue

    illness_name = filename.replace(f"merged_data_", "").replace(f"_{lag_label}.csv", "")
    file_path = os.path.join(merged_data_dir, filename)
    df = pd.read_csv(file_path)

    # Drop NaNs
    valid_rows = df[selected_vars].notna().all(axis=1) & df["CaseCount"].notna()
    df = df[valid_rows]

    if len(df) == 0:
        print(f"⚠️ Skipping {illness_name}: no valid rows")
        continue

    results = []

    for region in df["RegionCode"].unique():
        region_df = df[df["RegionCode"] == region]
        if len(region_df) < 10:
            continue

        X = region_df[selected_vars]
        y = region_df["CaseCount"]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        importances = model.feature_importances_

        # Plot actual vs predicted
        plot_dir = os.path.join(output_base_dir, illness_name, "Plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.title(f"{illness_name} | Region {region} | R²={r2:.2f}")
        plt.xlabel("Actual Cases")
        plt.ylabel("Predicted Cases")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"Region_{region}.png"))
        plt.close()

        # Store results
        result = {"RegionCode": region, "R_squared": round(r2, 4), "MSE": round(mse, 2)}
        for var, imp in zip(selected_vars, importances):
            result[f"Importance_{var}"] = round(imp, 4)
        results.append(result)

    # Save results per illness
    output_dir = os.path.join(output_base_dir, illness_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"RF_{lag_label}.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"✅ RF complete for {illness_name} — saved to {output_path}")
