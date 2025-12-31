import pandas as pd
import os

# === 1. Setup ===
input_folder = '/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/relevant illnesses'
output_folder = '/Users/jay/Desktop/Illness Prediction/Processed Data/Illness Data/final_illnesses_grouped'
os.makedirs(output_folder, exist_ok=True)

# === 2. Illness groupings and their descriptive names only ===
group_mappings = {
    "J30-J39": "Other diseases of upper respiratory tract",
    "J00-J06": "Acute upper respiratory infections",
    "K20-K31": "Diseases of oesophagus, stomach and duodenum",
    "N17-N19": "Renal failure",
    "J09-J18": "Influenza and pneumonia"
}

# === 3. Helper to check if a code is in a group ===
def is_code_in_range(code, group_range):
    try:
        prefix = group_range[0]
        start = int(group_range.split('-')[0][1:])
        end = int(group_range.split('-')[1][1:])
        return code.startswith(prefix) and start <= int(code[1:]) <= end
    except:
        return False

# === 4. Process each file ===
for filename in os.listdir(input_folder):
    if filename.endswith('.csv') and filename.startswith('relevant_illness_'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        # Add descriptive label column
        df['IllnessName'] = None

        for code_range, group_name in group_mappings.items():
            match = df['IllnessCode'].astype(str).apply(lambda x: is_code_in_range(x, code_range))
            df.loc[match, 'IllnessName'] = group_name

        # Keep only grouped illnesses
        grouped_df = df[df['IllnessName'].notna()].copy()

        if not grouped_df.empty:
            # Aggregate
            agg_df = grouped_df.groupby(
                ['ParsedDateTime', 'RegionCode', 'IllnessName'],
                as_index=False
            )['CaseCount'].sum()

            # Save output
            output_path = os.path.join(output_folder, filename)
            agg_df.to_csv(output_path, index=False)
            print(f"✅ Saved grouped file: {output_path}")