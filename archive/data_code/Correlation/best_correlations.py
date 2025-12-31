import os
import pandas as pd

# === USER INPUTS ===
base_dir = "/Users/jay/Desktop/Illness Prediction/Processed Data/Correlation Results"
result_subfolder = "filtered_correlation_"  # e.g., filtered_correlation_2019
years = [2019, 2020, 2021, 2022, 2023]
max_lag = 7
output_csv = "/Users/jay/Desktop/Illness Prediction/Processed Data/Correlation Results/top15_consistent_correlations_no_2020.csv"

all_data = []

# === Load all PearsonR values across years/lags ===
for year in years:
    year_path = os.path.join(base_dir, f"{result_subfolder}{year}")
    
    for lag in range(0, max_lag + 1):
        lag_path = os.path.join(year_path, f"lag_{lag}")
        if not os.path.exists(lag_path):
            continue
        
        for file in os.listdir(lag_path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(lag_path, file))
                df["Year"] = year
                df["Lag"] = lag
                df["AbsR"] = df["PearsonR"].abs()
                all_data.append(df)

# === Combine and group by IllnessCode + Feature ===
if all_data:
    full_df = pd.concat(all_data, ignore_index=True)
    
    summary = (
        full_df.groupby(["IllnessCode", "Feature"])
        .agg(
            MeanAbsR=("AbsR", "mean"),
            Appearances=("AbsR", "count"),
            YearsSeen=("Year", pd.Series.nunique),
            MedianR=("PearsonR", "median"),
        )
        .reset_index()
    )

    # Filter out those seen in fewer than, say, 3 different years (you can tweak this)
    summary = summary[summary["YearsSeen"] >= 3]

    # Get top 15 by mean absolute correlation
    top15 = summary.sort_values(by="MeanAbsR", ascending=False).head(15)

    # Save output
    top15.to_csv(output_csv, index=False)
    print(f"\n✅ Top 15 consistent correlations saved to: {output_csv}")
else:
    print("❌ No correlation files found.")
