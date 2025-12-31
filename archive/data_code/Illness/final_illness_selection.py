import pandas as pd
import os

# === 1. Setup ===
input_folder = '/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/relevant illnesses'  # Folder containing the CSVs
output_folder = '/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/final_illnesses'  # You can change this
os.makedirs(output_folder, exist_ok=True)

# === 2. Illness codes you want to keep ===
target_illnesses = ['J310', 'J060', 'K297', 'N185']  # Replace with your four illnesses

# === 3. Loop through files and filter ===
for filename in os.listdir(input_folder):
    if filename.endswith('.csv') and filename.startswith('relevant_illness_'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        # Filter only the rows with desired illnesses
        filtered_df = df[df['IllnessCode'].isin(target_illnesses)]  # Adjust column name if needed

        # Save to output folder
        output_path = os.path.join(output_folder, filename)
        filtered_df.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")
