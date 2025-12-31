import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

# === CONFIGURATION ===
illness_name = "Acute laryngopharyngitis"
lag_label = "lag4"
merged_data_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental/merged_data_{illness_name.replace(', ', '_').replace(' ', '_')}_{lag_label}.csv"
output_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/Selected Variables/{illness_name}/backward_elimination_{lag_label}.csv"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(merged_data_path)

# === PREPARE X and y ===
excluded_cols = {"RegionCode", "CaseCount", "ParsedDateTime", "IllnessName", "DateTime", "Month"}
env_features = [col for col in df.columns 
                if col not in excluded_cols 
                and "year" not in col.lower()
                and pd.api.types.is_numeric_dtype(df[col])]

X = df[env_features]
y = df["CaseCount"]

# Drop rows with NaNs in X or y
valid_rows = X.notna().all(axis=1) & y.notna()
X = X[valid_rows]
y = y[valid_rows]

# === BACKWARD ELIMINATION ===
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()

# Elimination loop
pmax = 1
threshold = 0.05
features = list(X.columns)

while pmax > threshold:
    X_with_const = sm.add_constant(X[features])
    model = sm.OLS(y, X_with_const).fit()
    p_values = model.pvalues.iloc[1:]  # skip intercept
    pmax = p_values.max()
    worst_feature = p_values.idxmax()

    if pmax > threshold:
        print(f"🗑️ Removing: {worst_feature} (p = {pmax:.4f})")
        features.remove(worst_feature)
    else:
        print("✅ All remaining features are significant.")
        break

# === FINAL MODEL FITTING ===
X_final = X[features]
model = sm.OLS(y, sm.add_constant(X_final)).fit()
coefs = model.params[1:].values  # skip intercept
r2 = model.rsquared

# === SAVE SELECTED FEATURES + COEFFICIENTS + R² ===
results = pd.DataFrame({
    "Selected_Variables": features,
    "Coefficient": coefs,
    "R_squared": r2
})

results.to_csv(output_path, index=False)
print(f"\n✅ Backward elimination results saved to:\n{output_path}")