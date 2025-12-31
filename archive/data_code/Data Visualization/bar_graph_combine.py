from PIL import Image

# Replace with your actual file paths
filepaths = [
    "/Users/jay/Desktop/Illness Prediction/Plots/Feature importance/Acute_upper_respiratory_infections_importance.png",
    "/Users/jay/Desktop/Illness Prediction/Plots/Feature importance/Diseases_of_oesophagus_stomach_and_duodenum_importance.png",
    "/Users/jay/Desktop/Illness Prediction/Plots/Feature importance/Other_diseases_of_upper_respiratory_tract_importance.png"
]

# Load images
images = [Image.open(fp) for fp in filepaths]

# === Layout Option ===
# Change to 'horizontal' or 'vertical'
layout = 'horizontal'  # or 'vertical'

# Get dimensions
if layout == 'horizontal':
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    new_img = Image.new('RGBA', (total_width, max_height))
    
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

elif layout == 'vertical':
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    new_img = Image.new('RGBA', (max_width, total_height))
    
    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.height

# Save the combined image
new_img.save("combined_image.png")
print("Saved as combined_image.png")