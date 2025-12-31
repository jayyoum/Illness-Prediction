import os
import zipfile
from collections import defaultdict

# === CONFIG ===
input_folder = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental/Grouped copy"
output_zip_path = "/Users/jay/Desktop/Illness Prediction/Processed Data/Illness & Environmental/Grouped_copy.zip"

# === GROUP FILES BY ILLNESS ===
illness_files = defaultdict(list)
for filename in os.listdir(input_folder):
    if filename.startswith("merged_data_") and filename.endswith(".csv"):
        parts = filename.replace(".csv", "").split("_lag")
        illness = parts[0].replace("merged_data_", "")
        illness_files[illness].append(filename)

# === CREATE TEMPORARY ILLNESS-LEVEL ZIP FILES ===
temp_zip_paths = []
for illness, files in illness_files.items():
    zip_name = f"{illness}.zip"
    zip_path = os.path.join(input_folder, zip_name)
    temp_zip_paths.append(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as illness_zip:
        for file in files:
            file_path = os.path.join(input_folder, file)
            illness_zip.write(file_path, arcname=file)

# === PACKAGE ALL ILLNESS ZIPs INTO ONE ZIP ===
with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as final_zip:
    for zip_path in temp_zip_paths:
        final_zip.write(zip_path, arcname=os.path.basename(zip_path))

# === CLEANUP TEMP ZIP FILES ===
for zip_path in temp_zip_paths:
    os.remove(zip_path)

print(f"\n✅ Final ZIP created: {output_zip_path}")