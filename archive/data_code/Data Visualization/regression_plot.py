import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# === USER INPUTS ===
year = 2019
illness_code = "H2502"

# === FILE PATHS ===
illness_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/relevant illnesses/relevant_illness_{year}.csv"
env_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data/{year}_merged_with_lag.csv"
plot_output_dir = f"/Users/jay/Desktop/Illness Prediction/Plots/RegressionPlots_{illness_code}_{year}"
os.makedirs(plot_output_dir, exist_ok=True)

# === LOAD AND MERGE ===
illness_df = pd.read_csv(illness_path, parse_dates=["ParsedDateTime"])
env_df = pd.read_csv(env_path, parse_dates=["DateTime"])
df = illness_df[illness_df["IllnessCode"] == illness_code]
merged = pd.merge(df, env_df, left_on=["ParsedDateTime", "RegionCode"], right_on=["DateTime", "RegionCode"], how="inner")

# === CORRELATION ANALYSIS ===
env_features = [col for col in env_df.select_dtypes(include='number').columns if col != 'RegionCode']
correlations = {}

for feature in env_features:
    x = merged[feature].values
    y = merged["CaseCount"].values
    if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
        r = np.corrcoef(x, y)[0, 1]
        if pd.notnull(r):
            correlations[feature] = r

if not correlations:
    print(f"❌ No valid correlations for illness {illness_code}")
    exit()

# === SELECT TOP 50% FEATURES ===
sorted_feats = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
top_features = [feat for feat, _ in sorted_feats[:len(sorted_feats)//2]]

print(f"\n📊 Top Features for {illness_code}: {top_features}")

# === REGRESSION AND PLOTTING ===
for feature in top_features:
    x = merged[[feature]].values
    y = merged["CaseCount"].values
    mask = ~np.isnan(x).ravel() & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        continue

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n🔹 Feature: {feature}")
    print(f"   Coef: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}, MSE: {mean_squared_error(y_test, y_pred):.2f}")

    plt.figure()
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.title(f"{illness_code} vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("Case Count")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot to file
    filename = os.path.join(plot_output_dir, f"{illness_code}_{feature}_regression.png")
    plt.savefig(filename)
    print(f"📁 Plot saved to: {filename}")
    plt.show()