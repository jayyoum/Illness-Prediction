import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

param_dist = {
    'n_estimators': [300, 500, 800, 1000, 1200],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}

# === CONFIG ===
lag_label = "lag3"
intersected_var_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/Selected Variables/intersected_variables_FS.csv"
merged_data_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental"
output_base_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/XGB RandomSearch/FS_lag3"
n_iter_search = 40  # Number of random combos to try

# === LOAD SELECTED VARIABLES ===
var_df = pd.read_csv(intersected_var_path)
selected_vars = var_df.iloc[:, 0].dropna().astype(str).tolist()
print(f"✅ Using variables: {selected_vars}")

for filename in os.listdir(merged_data_dir):
    if not filename.endswith(f"_{lag_label}.csv"):
        continue

    illness_name = filename.replace(f"merged_data_", "").replace(f"_{lag_label}.csv", "")
    file_path = os.path.join(merged_data_dir, filename)
    df = pd.read_csv(file_path)

    # Drop rows with NaNs
    valid_rows = df[selected_vars].notna().all(axis=1) & df["CaseCount"].notna()
    df = df[valid_rows]

    if len(df) < 10:
        print(f"⚠️ Skipping {illness_name}: Not enough data")
        continue

    X = df[selected_vars].values
    y = df["CaseCount"].values

    # Pipeline (optional scaling)
    pipeline = Pipeline([
        ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0))
    ])

    # Cross-validation & Random Search
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={f"xgb__{k}": v for k, v in param_dist.items()},
        n_iter=n_iter_search,
        scoring=make_scorer(r2_score),
        cv=kf,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X, y)
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = round(search.best_score_, 4)

    # Save results
    result = {"IllnessName": illness_name, "Best_R²": best_score}
    result.update({k.replace("xgb__", ""): v for k, v in best_params.items()})

    output_dir = os.path.join(output_base_dir, illness_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"XGB_random_search_{lag_label}.csv")
    pd.DataFrame([result]).to_csv(output_path, index=False)
    print(f"✅ RandomizedSearchCV done for {illness_name} — saved to {output_path}")
