import os
import cv2
import glob
import numpy as np

print("==================================================")
print("🎨 FIXING MAC PREVIEW ARTIFACTS (WHITE TO BLACK)")
print("==================================================")

# Point this to your manually curated folder!
TARGET_DIR = "../images/train_dataset_v2_masked" 
categories = ['none', 'handshake']

fixed_count = 0

for category in categories:
    image_paths = glob.glob(os.path.join(TARGET_DIR, category, "*.jpg"))
    
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Mac Preview white is usually exactly 255, but JPEG compression 
        # can sometimes blur the edges to 250+. 
        # This creates a mask of all pixels brighter than 245.
        white_pixels = img > 245 
        
        # If the script finds white patches, flip them to pure black (0)
        if np.any(white_pixels):
            img[white_pixels] = 0
            
            # Overwrite the file with the corrected black mask
            cv2.imwrite(img_path, img)
            fixed_count += 1

print("\n==================================================")
print(f"✅ Clean-up Complete!")
print(f"Found and fixed white artifact patches in {fixed_count} images.")
print("==================================================")