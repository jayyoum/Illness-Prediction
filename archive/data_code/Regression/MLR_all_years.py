import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# === USER CONFIG ===
illness_name = "Chronic rhinitis"
lag_days = 0  # ← Adjust lag here

# === FILE PATHS ===
illness_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/final_illnesses/combined_illness.csv"
env_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Merged Data/merged_env.csv"
output_path = f"/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/MLR all years/{illness_name}_lag{lag_days}_ALL_YEARS.csv"

# === LOAD DATA ===
illness_df = pd.read_csv(illness_path, parse_dates=["ParsedDateTime"])
env_df = pd.read_csv(env_path, parse_dates=["DateTime"])

# === Filter and apply lag ===
illness_df = illness_df[illness_df["IllnessName"] == illness_name].copy()
illness_df["ParsedDateTime"] = illness_df["ParsedDateTime"] + pd.Timedelta(days=lag_days)

# === Merge with environmental data ===
merged = pd.merge(
    illness_df,
    env_df,
    left_on=["ParsedDateTime", "RegionCode"],
    right_on=["DateTime", "RegionCode"],
    how="inner"
)

# === Get environmental features, excluding RegionCode, Year, Month ===
excluded_cols = {"RegionCode", "Year", "Month"}
env_features = [col for col in env_df.select_dtypes(include='number').columns if col not in excluded_cols]

# === Drop rows with NaNs in X or y ===
merged.dropna(subset=env_features + ["CaseCount"], inplace=True)

# === Loop through regions and run regression ===
regions = merged["RegionCode"].unique()
records = []

print(f"\n📊 MLR for Illness: {illness_name} | Lag: {lag_days} | Years: ALL")

for region in regions:
    region_data = merged[merged["RegionCode"] == region]

    X = region_data[env_features].values
    y = region_data["CaseCount"].values

    if len(y) < 10:
        print(f"⚠️ Region {region}: Not enough data ({len(y)} rows), skipping.")
        continue

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    coefs = model.coef_

    # Store results
    result = {
        "RegionCode": region,
        "R_squared": round(r2, 4),
        "MSE": round(mse, 2)
    }
    for feat, coef in zip(env_features, coefs):
        result[f"β_{feat}"] = round(coef, 4)
    
    records.append(result)

# === Save to CSV ===
results_df = pd.DataFrame(records)
results_df.sort_values(by="RegionCode", inplace=True)
results_df.to_csv(output_path, index=False)
print(f"\n✅ Coefficients saved to: {output_path}")
