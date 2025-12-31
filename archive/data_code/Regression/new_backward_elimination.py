import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === CONFIGURATION ===
grouped_illness_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/final_illnesses_grouped/combined_illness.csv"
merged_data_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental/Grouped"
output_base_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/Selected Variables/BE_grouped"
lags = range(0, 8)

# === LOAD UNIQUE GROUPED ILLNESSES ===
grouped_df = pd.read_csv(grouped_illness_path)
grouped_illnesses = grouped_df["IllnessName"].unique()

# === HELPER FUNCTIONS ===
def adjusted_r2(r2, n, k):
    return 1 - ((1 - r2) * (n - 1)) / (n - k - 1)

def backward_elimination(X, y, verbose=True):
    remaining = list(X.columns)
    current_score = -np.inf
    n = len(y)

    while len(remaining) > 1:
        scores = []
        for candidate in remaining:
            trial_features = [f for f in remaining if f != candidate]
            model = LinearRegression().fit(X[trial_features], y)
            r2 = r2_score(y, model.predict(X[trial_features]))
            adj_r2 = adjusted_r2(r2, n, len(trial_features))
            scores.append((adj_r2, candidate))

        scores.sort(reverse=True)
        best_new_score, worst_candidate = scores[0]

        if best_new_score > current_score:
            remaining.remove(worst_candidate)
            current_score = best_new_score
            if verbose:
                print(f"❌ Removed: {worst_candidate} | Adjusted R² = {current_score:.4f}")
        else:
            if verbose:
                print("✅ No improvement on removal. Stopping.")
            break

    return remaining

# === MAIN LOOP: Grouped Illnesses × Lags ===
for illness_name in grouped_illnesses:
    for lag_days in lags:
        safe_name = illness_name.replace(", ", "_").replace(" ", "_")
        filename = f"merged_data_{safe_name}_lag{lag_days}.csv"
        filepath = os.path.join(merged_data_dir, filename)

        if not os.path.exists(filepath):
            print(f"❌ File not found: {filename}")
            continue

        df = pd.read_csv(filepath)

        excluded_cols = {"RegionCode", "Month", "CaseCount", "ParsedDateTime", "IllnessName", "DateTime"}
        env_features = [
            col for col in df.columns
            if col not in excluded_cols
            and col != "MaxWindDir"  # ✅ EXCLUDE HERE
            and "year" not in col.lower()
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        X = df[env_features]
        y = df["CaseCount"]
        valid_rows = X.notna().all(axis=1) & y.notna()
        X = X[valid_rows]
        y = y[valid_rows]

        if len(X) < 10:
            print(f"⚠️ Skipping {illness_name} (lag {lag_days}): Not enough valid rows")
            continue

        print(f"\n📊 Backward Elimination | {illness_name} | Lag {lag_days}")
        selected_vars = backward_elimination(X, y, verbose=False)

        if not selected_vars:
            print(f"⚠️ No variables selected for {illness_name} (lag {lag_days})")
            continue

        model = LinearRegression().fit(X[selected_vars], y)
        coefs = model.coef_
        r2 = r2_score(y, model.predict(X[selected_vars]))

        results = pd.DataFrame({
            "Selected_Variables": selected_vars,
            "Coefficient": coefs,
            "R_squared": [r2] * len(selected_vars)
        })

        illness_dir = os.path.join(output_base_dir, illness_name)
        os.makedirs(illness_dir, exist_ok=True)
        output_path = os.path.join(illness_dir, f"backward_elimination_lag{lag_days}.csv")
        results.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")
