import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_env_feature_correlation_matrix(env_csv_path, year, base_output_dir):
    df = pd.read_csv(env_csv_path, parse_dates=["DateTime"])

    # Keep only numeric environmental variables (drop ID columns)
    df = df.select_dtypes(include='number')
    df = df.drop(columns=['RegionCode'], errors='ignore')

    # Compute Pearson correlation
    corr = df.corr()

    # Set larger figure size for readability
    plt.figure(figsize=(18, 16))
    sns.heatmap(
        corr, cmap="coolwarm", annot=True, fmt=".2f", square=True,
        cbar_kws={"label": "Pearson Correlation"}, annot_kws={"size": 9}
    )
    plt.title(f"Environmental Feature Correlation Matrix | {year}", fontsize=18)
    plt.tight_layout()

    # Save to the corresponding Lag_{year} folder
    output_dir = os.path.join(base_output_dir, f"lag_{year}")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"env_corr_matrix_{year}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    # Base input and output paths
    merged_dir = os.path.expanduser("~/Desktop/Illness Prediction/Processed Data/Merged Data")
    base_output_dir = os.path.expanduser("~/Desktop/Illness Prediction/Processed Data/Correlation Results")

    for filename in os.listdir(merged_dir):
        if filename.endswith("_merged_with_lag.csv"):
            year = filename.split('_')[0]
            file_path = os.path.join(merged_dir, filename)
            plot_env_feature_correlation_matrix(file_path, year, base_output_dir)
