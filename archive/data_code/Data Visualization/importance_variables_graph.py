import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the main folder path
main_folder = "/Users/jay/Desktop/Illness Prediction/Processed Data/Regression Results/New +/TT"

# Loop through each subfolder inside TT
for illness_folder in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, illness_folder)
    if os.path.isdir(folder_path):
        # Look for the lag0 XGB_FS file in each illness folder
        for file_name in os.listdir(folder_path):
            if file_name.startswith("XGB_FS_") and file_name.endswith("_lag0_train_test.csv"):
                file_path = os.path.join(folder_path, file_name)
                try:
                    df = pd.read_csv(file_path)

                    # Extract feature importance columns only
                    importance_cols = [col for col in df.columns if col.startswith("Importance_")]

                    if importance_cols:
                        # Convert to tidy format: feature name and importance score
                        importance_data = []
                        for col in importance_cols:
                            feature_name = col.replace("Importance_", "")
                            score = df[col].iloc[0]  # Assuming importance values are on the first row
                            importance_data.append((feature_name, score))

                        # Create DataFrame for plotting
                        importance_df = pd.DataFrame(importance_data, columns=["Feature", "Importance"])
                        importance_df = importance_df.sort_values(by="Importance", ascending=True)

                        # Plotting
                        plt.figure(figsize=(8, 5))
                        plt.barh(importance_df["Feature"], importance_df["Importance"])
                        plt.xlabel("Importance Score")
                        plt.title(f"Feature Importance - {illness_folder}")
                        plt.tight_layout()

                        # Save plot
                        plot_path = os.path.join(folder_path, f"{illness_folder}_importance.png")
                        plt.savefig(plot_path)
                        plt.close()
                        print(f"Saved plot for {illness_folder} to {plot_path}")
                    else:
                        print(f"No importance columns found in {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
