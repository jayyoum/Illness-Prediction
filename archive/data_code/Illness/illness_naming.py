import pandas as pd
import os

# === USER INPUT ===
input_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/final_illnesses/relevant_illness_2023.csv"
output_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/final_illnesses/relevant_illness_2023.csv"

# === Mapping from code to illness name ===
code_to_name = {
    "J060": "Acute laryngopharyngitis",
    "J310": "Chronic rhinitis",
    "K297": "Gastritis, unspecified",
    "N185": "Chronic kidney disease, stage 5"
}

# === Load and replace codes ===
df = pd.read_csv(input_path)
if "IllnessCode" in df.columns:
    df["IllnessCode"] = df["IllnessCode"].replace(code_to_name)
    df.rename(columns={"IllnessCode": "IllnessName"}, inplace=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Illness codes replaced and saved to: {output_path}")
else:
    print("❌ Column 'IllnessCode' not found in input file.")