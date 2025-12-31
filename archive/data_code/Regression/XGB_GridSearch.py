import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, r2_score

# === CONFIG ===
lag_label = "lag3"
intersected_var_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/Selected Variables/intersected_variables_FS.csv"
merged_data_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental"
output_base_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/XGB GridSearch/FS_lag3_KoreaWide"
os.makedirs(output_base_dir, exist_ok=True)

# === LOAD VARIABLES ===
var_df = pd.read_csv(intersected_var_path)
selected_vars = var_df.iloc[:, 0].dropna().astype(str).tolist()
print(f"✅ Using variables: {selected_vars}")

# === HYPERPARAMETER GRID (includes original settings) ===
param_grid = {
    'n_estimators': [100, 120, 150],
    'learning_rate': [0.1, 0.15],
    'max_depth': [4, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# === CUSTOM CV SPLITTER (matches your previous script) ===
cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)

# === LOOP THROUGH FILES ===
for filename in os.listdir(merged_data_dir):
    if not filename.endswith(f"_{lag_label}.csv"):
        continue

    illness_name = filename.replace(f"merged_data_", "").replace(f"_{lag_label}.csv", "")
    file_path = os.path.join(merged_data_dir, filename)
    df = pd.read_csv(file_path)

    # Drop rows with missing values
    valid_rows = df[selected_vars].notna().all(axis=1) & df["CaseCount"].notna()
    df = df[valid_rows]

    if len(df) < 50:
        print(f"⚠️ Skipping {illness_name}: not enough rows")
        continue

    X = df[selected_vars].values
    y = df["CaseCount"].values

    model = XGBRegressor(
        objective='reg:squarederror',
        verbosity=0,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(r2_score),
        cv=cv_splitter,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_r2 = grid_search.best_score_
    best_params = grid_search.best_params_

    # Refit on full data to get importances
    best_model.fit(X, y)
    importances = best_model.feature_importances_

    result = {
        "IllnessName": illness_name,
        "Best_R²_CV": round(best_r2, 4),
        **{f"Param_{k}": v for k, v in best_params.items()}
    }
    for var, imp in zip(selected_vars, importances):
        result[f"Importance_{var}"] = round(imp, 4)

    output_df = pd.DataFrame([result])
    output_dir = os.path.join(output_base_dir, illness_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"XGB_gridsearch_revised_{illness_name}.csv")
    output_df.to_csv(output_path, index=False)
    print(f"✅ GridSearch revised complete for {illness_name} — saved to {output_path}")
