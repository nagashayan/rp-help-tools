import os

from PIL import Image

input_folder = (
    "/Users/nagashayanaramamurthy/GitHub/rp-help-tools/images/train_dataset/none/"
)
output_folder = (
    "/Users/nagashayanaramamurthy/GitHub/rp-help-tools/"
    "images/"
    "train_dataset/"
    "none_converted/"
)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        webp_path = os.path.join(input_folder, filename)
        try:
            # Open and convert the image to JPEG
            img = Image.open(webp_path)
            output_filename = os.path.splitext(filename)[0] + ".jpg"
            output_path = os.path.join(output_folder, output_filename)
            img.convert("RGB").save(output_path, "JPEG")
            print(f"Converted {filename} to {output_filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
