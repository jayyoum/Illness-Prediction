import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, r2_score

# === CONFIGURATION ===
file_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental/merged_data_Acute_upper_respiratory_infections_lag0.csv"
excluded_vars = {"MaxWindDir", "Year_x", "Year_y"}
cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)

# === LOAD AND PREPARE DATA ===
df = pd.read_csv(file_path)

excluded_cols = {"RegionCode", "Month", "CaseCount", "ParsedDateTime", "IllnessName", "DateTime"}
env_features = [col for col in df.columns
                if col not in excluded_cols
                and col not in excluded_vars
                and "year" not in col.lower()
                and pd.api.types.is_numeric_dtype(df[col])]

X = df[env_features]
y = df["CaseCount"]

valid_rows = X.notna().all(axis=1) & y.notna()
X = X[valid_rows]
y = y[valid_rows]

# === PARAMETER GRID FOR RANDOM SEARCH ===
param_dist = {
    'n_estimators': [300, 600, 900, 1200, 1500, 1800],
    'learning_rate': [0.01, 0.03, 0.05, 0.07],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.6, 0.75, 0.9],
    'gamma': [0, 0.1, 0.3]
}

# === FORWARD SELECTION BASED ON XGBOOST + R² ===
def forward_selection_xgb(X, y, param_dist, verbose=True):
    remaining = list(X.columns)
    selected = []
    current_score = -np.inf
    best_params_overall = None

    while remaining:
        scores = []

        for candidate in remaining:
            trial_features = selected + [candidate]
            X_trial = X[trial_features]

            model = XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                scoring=make_scorer(r2_score),
                n_iter=15,
                cv=cv_splitter,
                n_jobs=-1,
                random_state=42
            )
            search.fit(X_trial, y)
            mean_r2 = search.best_score_
            scores.append((mean_r2, candidate, search.best_params_))

        scores.sort(reverse=True)
        best_new_score, best_candidate, best_candidate_params = scores[0]

        if best_new_score > current_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            best_params_overall = best_candidate_params
            if verbose:
                print(f"✅ Added: {best_candidate} | R² = {current_score:.4f}")
        else:
            if verbose:
                print("⛔ No improvement. Stopping.")
            break

    return selected, best_params_overall, current_score

# === RUN ===
print(f"\n📊 Running XGBoost-based Forward Selection with exclusions: {excluded_vars}")
selected_vars, best_params, best_r2 = forward_selection_xgb(X, y, param_dist)

print("\n✅ Final Selected Variables:")
print(selected_vars)
print("\n🧠 Best Hyperparameters:")
print(best_params)
print(f"\n📈 Best R² from CV: {best_r2:.4f}")

# === FINAL MODEL FIT + FEATURE IMPORTANCE ===
final_model = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42, verbosity=0)
final_model.fit(X[selected_vars], y)
importances = final_model.feature_importances_

# === SAVE TO CSV ===
results = {
    "Best_R2_CV": round(best_r2, 4),
    **{f"Param_{k}": v for k, v in best_params.items()}
}
for var, imp in zip(selected_vars, importances):
    results[f"Importance_{var}"] = round(imp, 4)

output_df = pd.DataFrame([results])
output_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/New +/XGB_FS_AcuteURI_lag0.csv"
output_df.to_csv(output_path, index=False)
print(f"\n💾 Results saved to: {output_path}")