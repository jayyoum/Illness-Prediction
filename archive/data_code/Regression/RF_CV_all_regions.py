import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# === CONFIG ===
lag_label = "lag3"
intersected_var_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/Selected Variables/intersected_variables_SS.csv"
merged_data_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental"
output_base_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/RF with CV/SS_lag3_KoreaWide"
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

    # Drop rows with NaNs in selected features or target
    valid_rows = df[selected_vars].notna().all(axis=1) & df["CaseCount"].notna()
    df = df[valid_rows]

    if len(df) < k_folds:
        print(f"⚠️ Skipping {illness_name}: not enough rows for CV")
        continue

    X = df[selected_vars].values
    y = df["CaseCount"].values

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    r2_scores, mse_scores = [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2_scores.append(r2_score(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))

    # Final model on all data (for feature importances)
    model.fit(X, y)
    importances = model.feature_importances_

    # Save results
    result = {
        "IllnessName": illness_name,
        "R²_Mean": round(np.mean(r2_scores), 4),
        "R²_Std": round(np.std(r2_scores), 4),
        "MSE_Mean": round(np.mean(mse_scores), 2),
        "MSE_Std": round(np.std(mse_scores), 2)
    }
    for var, imp in zip(selected_vars, importances):
        result[f"Importance_{var}"] = round(imp, 4)

    output_dir = os.path.join(output_base_dir, illness_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"RF_cv_KoreaWide_{lag_label}.csv")
    pd.DataFrame([result]).to_csv(output_path, index=False)
    print(f"✅ RF with CV (Korea-wide) done for {illness_name} — saved to {output_path}")
