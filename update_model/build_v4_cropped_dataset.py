import os
import glob
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("==================================================")
print("✂️ BUILDING V4 DATASET (TIGHT HAND CROPS)")
print("==================================================")

# 1. Configuration
INPUT_DIR = "../images/train_dataset_v2"  # This is the original dataset with full frames
OUTPUT_DIR = "../images/train_dataset_v2_cropped"
categories = ['none', 'handshake']

# Create output directories
for cat in categories:
    os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)

# 2. Initialize MediaPipe
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=3)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Bounding Box Function (Same as your benchmark)
def get_hand_bbox(landmarks, img_w, img_h):
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    
    pad_x = (xmax - xmin) * 0.2
    pad_y = (ymax - ymin) * 0.2
    
    px_xmin = max(0, int((xmin - pad_x) * img_w))
    px_xmax = min(img_w, int((xmax + pad_x) * img_w))
    px_ymin = max(0, int((ymin - pad_y) * img_h))
    px_ymax = min(img_h, int((ymax + pad_y) * img_h))
    
    return px_xmin, px_ymin, px_xmax, px_ymax

# 4. Processing Loop
total_processed = 0
total_saved = 0

for category in categories:
    print(f"\nProcessing category: {category.upper()}...")
    image_paths = glob.glob(os.path.join(INPUT_DIR, category, "*.jpg"))
    
    for img_path in image_paths:
        total_processed += 1
        frame = cv2.imread(img_path)
        if frame is None: continue
            
        img_h, img_w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        result = detector.detect(mp_image)
        
        if result.hand_landmarks:
            # Calculate reach for all detected hands
            hands_data = []
            for landmarks in result.hand_landmarks:
                reach = landmarks[0].z - landmarks[12].z
                hands_data.append({'landmarks': landmarks, 'reach': reach})
            
            # Target Lock: Pick the hand reaching the most
            best_hand = max(hands_data, key=lambda x: x['reach'])
            lms = best_hand['landmarks']
            
            # Crop the hand
            xmin, ymin, xmax, ymax = get_hand_bbox(lms, img_w, img_h)
            
            if xmax > xmin and ymax > ymin:
                hand_crop = frame[ymin:ymax, xmin:xmax] # Save in original color, CNN will convert to gray
                
                # We resize it to 160x160 here so the dataset is perfectly uniform
                hand_resized = cv2.resize(hand_crop, (160, 160))
                
                # Save it
                filename = os.path.basename(img_path)
                out_path = os.path.join(OUTPUT_DIR, category, f"crop_{filename}")
                cv2.imwrite(out_path, hand_resized)
                total_saved += 1

print("\n==================================================")
print(f"✅ V4 Dataset Complete!")
print(f"Analyzed {total_processed} total images.")
print(f"Successfully cropped and saved {total_saved} hands.")
print("==================================================")