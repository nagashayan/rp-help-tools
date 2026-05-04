import os
import cv2
import glob

print("==================================================")
print("⬜ BUILDING GRAYSCALE DATASET")
print("==================================================")

INPUT_DIR = "../images/train_dataset_v2"
OUTPUT_DIR = "../images/train_dataset_v2_gray"
categories = ['none', 'handshake']

# Create output directories
for cat in categories:
    os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)

total_processed = 0

for category in categories:
    print(f"Processing category: {category.upper()}...")
    image_paths = glob.glob(os.path.join(INPUT_DIR, category, "*.jpg"))
    
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None: continue
            
        # Convert to Grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Save the grayscale image at original resolution
        filename = os.path.basename(img_path)
        out_path = os.path.join(OUTPUT_DIR, category, f"gray_{filename}")
        
        cv2.imwrite(out_path, gray_frame)
        total_processed += 1

print("\n==================================================")
print(f"✅ Grayscale Dataset Complete!")
print(f"Successfully converted and saved {total_processed} images to {OUTPUT_DIR}.")
print("==================================================")