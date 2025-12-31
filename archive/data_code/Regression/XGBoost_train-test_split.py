import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# === CONFIG ===
lag_label = "lag3"
intersected_var_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/Selected Variables/intersected_variables_BE.csv"
merged_data_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental"
output_base_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/XGB with Split/BE_lag3"

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
        if len(region_df) < 20:
            continue

        X = region_df[selected_vars].values
        y = region_df["CaseCount"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        importances = model.feature_importances_

        # Plot (test set only)
        plot_dir = os.path.join(output_base_dir, illness_name, "Plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f"{illness_name} | Region {region} | Test R²={r2_test:.2f}")
        plt.xlabel("Actual Cases (Test)")
        plt.ylabel("Predicted Cases")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"Region_{region}.png"))
        plt.close()

        # Store results
        result = {
            "RegionCode": region,
            "R²_Train": round(r2_train, 4),
            "R²_Test": round(r2_test, 4),
            "MSE_Test": round(mse_test, 2)
        }
        for var, imp in zip(selected_vars, importances):
            result[f"Importance_{var}"] = round(imp, 4)
        results.append(result)

    # Save results
    output_dir = os.path.join(output_base_dir, illness_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"XGB_split_{lag_label}.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"✅ XGB with train-test split done for {illness_name} — saved to {output_path}")
