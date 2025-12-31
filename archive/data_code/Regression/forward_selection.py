import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === USER CONFIGURATION ===
illness_name = "Acute laryngopharyngitis"
lag_days = 4
merged_data_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental/merged_data_{illness_name.replace(', ', '_').replace(' ', '_')}_lag{lag_days}.csv"
output_csv_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/Selected Variables/{illness_name}/forward_selection_lag{lag_days}.csv"

os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# === LOAD MERGED DATA ===
df = pd.read_csv(merged_data_path)

# === PREPARE FEATURES ===
excluded_cols = {"RegionCode", "Month", "CaseCount", "ParsedDateTime", "IllnessName", "DateTime"}
env_features = [col for col in df.columns 
                if col not in excluded_cols 
                and "year" not in col.lower()
                and pd.api.types.is_numeric_dtype(df[col])]

X = df[env_features]
y = df["CaseCount"]

# Drop any rows where X or y have NaNs
valid_rows = X.notna().all(axis=1) & y.notna()
X = X[valid_rows]
y = y[valid_rows]

# === HELPER FUNCTIONS ===
def adjusted_r2(r2, n, k):
    return 1 - ((1 - r2) * (n - 1)) / (n - k - 1)

def forward_selection(X, y, verbose=True):
    remaining = list(X.columns)
    selected = []
    current_score, best_new_score = -np.inf, -np.inf
    n = len(y)

    while remaining:
        scores = []
        for candidate in remaining:
            trial_features = selected + [candidate]
            model = LinearRegression().fit(X[trial_features], y)
            r2 = r2_score(y, model.predict(X[trial_features]))
            adj_r2 = adjusted_r2(r2, n, len(trial_features))
            scores.append((adj_r2, candidate))

        scores.sort(reverse=True)
        best_new_score, best_candidate = scores[0]

        if best_new_score > current_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            if verbose:
                print(f"✅ Added: {best_candidate} | Adjusted R² = {current_score:.4f}")
        else:
            if verbose:
                print("⛔ No improvement. Stopping.")
            break

    return selected

# === RUN FORWARD SELECTION ===
print(f"\n📊 Running Forward Selection | Illness: {illness_name} | Lag: {lag_days}")
selected_vars = forward_selection(X, y)

# === FINAL MODEL FITTING ===
X_final = X[selected_vars]
model = LinearRegression().fit(X_final, y)
coefs = model.coef_
r2 = r2_score(y, model.predict(X_final))

# === SAVE SELECTED FEATURES + COEFFICIENTS + R² ===
results = pd.DataFrame({
    "Selected_Variables": selected_vars,
    "Coefficient": coefs,
    "R_squared": r2
})

results.to_csv(output_csv_path, index=False)
print(f"\n✅ Forward selection results saved to:\n{output_csv_path}")
