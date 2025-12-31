import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

# === CONFIG ===
lag_label = "lag3"
intersected_var_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/Selected Variables/intersected_variables_BE.csv"
merged_data_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental"
output_base_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/XGB with CV/BE_lag3"
k_folds = 5

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
        if len(region_df) < k_folds:
            continue

        X = region_df[selected_vars].values
        y = region_df["CaseCount"].values

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        r2_scores, mse_scores = [], []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1,
                reg_alpha=0,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2_scores.append(r2_score(y_test, y_pred))
            mse_scores.append(mean_squared_error(y_test, y_pred))

        # Final model for feature importances
        model.fit(X, y)
        importances = model.feature_importances_

        result = {
            "RegionCode": region,
            "R²_Mean": round(np.mean(r2_scores), 4),
            "R²_Std": round(np.std(r2_scores), 4),
            "MSE_Mean": round(np.mean(mse_scores), 2),
            "MSE_Std": round(np.std(mse_scores), 2)
        }
        for var, imp in zip(selected_vars, importances):
            result[f"Importance_{var}"] = round(imp, 4)

        results.append(result)

    # Save results
    output_dir = os.path.join(output_base_dir, illness_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"XGB_cv_{lag_label}.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"✅ XGB with CV done for {illness_name} — saved to {output_path}")
