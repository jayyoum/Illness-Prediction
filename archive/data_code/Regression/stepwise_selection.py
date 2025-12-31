import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

# === CONFIGURATION ===
illness_name = "Acute laryngopharyngitis"
lag_label = "lag4"
merged_data_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental/merged_data_{illness_name.replace(', ', '_').replace(' ', '_')}_{lag_label}.csv"
output_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/Selected Variables/{illness_name}/stepwise_selection_{lag_label}.csv"
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

# === STEPWISE SELECTION ===
def stepwise_selection(X, y, threshold_in=0.05, threshold_out=0.05, verbose=True):
    included = []
    while True:
        changed = False

        # Forward Step
        excluded = list(set(X.columns) - set(included))
        new_pvals = pd.Series(index=excluded, dtype=float)

        for new_col in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_col]])).fit()
            new_pvals[new_col] = model.pvalues[new_col]

        if not new_pvals.empty:
            best_pval = new_pvals.min()
            if best_pval < threshold_in:
                best_feature = new_pvals.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f"✅ Added: {best_feature} (p = {best_pval:.4f})")

        # Backward Step
        if included:
            model = sm.OLS(y, sm.add_constant(X[included])).fit()
            pvals = model.pvalues.iloc[1:]  # skip intercept
            worst_pval = pvals.max()
            if worst_pval > threshold_out:
                worst_feature = pvals.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print(f"🗑️ Removed: {worst_feature} (p = {worst_pval:.4f})")

        if not changed:
            break

    return included

# Run stepwise selection
print(f"\n📊 Stepwise Selection | Illness: {illness_name} | Lag: {lag_label}")
selected_vars = stepwise_selection(X, y)

# === FINAL MODEL FITTING ===
X_final = X[selected_vars]
model = sm.OLS(y, sm.add_constant(X_final)).fit()
coefs = model.params[1:].values  # skip intercept
r2 = model.rsquared

# === SAVE SELECTED FEATURES + COEFFICIENTS + R² ===
results = pd.DataFrame({
    "Selected_Variables": selected_vars,
    "Coefficient": coefs,
    "R_squared": r2
})

results.to_csv(output_path, index=False)
print(f"\n✅ Stepwise selection results saved to:\n{output_path}")