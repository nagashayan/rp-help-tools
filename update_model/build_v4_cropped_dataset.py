import os
import cv2
import glob
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("==================================================")
print("✂️ GENERATING MACRO-TO-MICRO CROPPED DATASET")
print("==================================================")

# --- CONFIGURATION ---
INPUT_DATASET_DIR = "../images/train_dataset_v2"  # Ensure this is your original color dataset!
OUTPUT_DATASET_DIR = "../images/train_dataset_v3_cropped"
CATEGORIES = ["none", "handshake"]

# 1. Initialize MediaPipe Tasks Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=4,  # Allow multiple hands so we can filter for the reaching one
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Create output directories
for category in CATEGORIES:
    os.makedirs(os.path.join(OUTPUT_DATASET_DIR, category), exist_ok=True)

processed_count = 0
skipped_count = 0

for category in CATEGORIES:
    print(f"\nProcessing category: {category}...")
    
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(INPUT_DATASET_DIR, category, ext)))
        
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None: continue
            
        img_h, img_w, _ = frame.shape
        
        # 2. Process with Tasks API
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect(mp_image)
        
        if not results.hand_landmarks:
            skipped_count += 1
            continue
            
        # --- 3. TRAINING-INFERENCE PARITY (Z-Index / Reach Logic) ---
        best_lms = None
        min_z = float('inf') 
        
        for hand_landmarks in results.hand_landmarks:
            # Calculate a Z-based "reach" score (lower Z generally means closer to camera)
            # Replace this with your EXACT math from benchmark_hybrid.py!
            avg_z = sum([lm.z for lm in hand_landmarks]) / len(hand_landmarks)
            hand_reach_score = avg_z 
            
            if hand_reach_score < min_z:
                min_z = hand_reach_score
                best_lms = hand_landmarks
                
        lms = best_lms
        
        # 4. Get standard hand bounding box coordinates in pixels
        x_coords = [int(lm.x * img_w) for lm in lms]
        y_coords = [int(lm.y * img_h) for lm in lms]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # --- 5. DYNAMIC PROPORTIONAL PADDING ---
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        # Fingers get 60% padding. Arm gets 150% to capture the sleeve.
        pad_fingers_x = int(box_w * 0.6)
        pad_fingers_y = int(box_h * 0.6)
        pad_arm_x = int(box_w * 1.5)
        pad_arm_y = int(box_h * 1.5)
        
        # Landmark 0 = Wrist | Landmark 9 = Middle finger knuckle
        wrist_x, wrist_y = int(lms[0].x * img_w), int(lms[0].y * img_h)
        mid_x, mid_y = int(lms[9].x * img_w), int(lms[9].y * img_h)
        
        arm_dx = wrist_x - mid_x
        arm_dy = wrist_y - mid_y
        
        # Dynamically stretch the box toward the arm direction using the proportional padding
        x_min_adj = max(0, x_min - (pad_arm_x if arm_dx < 0 else pad_fingers_x))
        x_max_adj = min(img_w, x_max + (pad_arm_x if arm_dx > 0 else pad_fingers_x))
        y_min_adj = max(0, y_min - (pad_arm_y if arm_dy < 0 else pad_fingers_y))
        y_max_adj = min(img_h, y_max + (pad_arm_y if arm_dy > 0 else pad_fingers_y))
        
        # 6. Crop and Resize
        hand_arm_crop = frame[y_min_adj:y_max_adj, x_min_adj:x_max_adj]
        
        if hand_arm_crop.size == 0:
            skipped_count += 1
            continue
            
        crop_resized = cv2.resize(hand_arm_crop, (160, 160))
        
        # Save output
        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DATASET_DIR, category, filename)
        cv2.imwrite(save_path, crop_resized)
        processed_count += 1

print("\n" + "="*50)
print(f"✅ Cropping Complete!")
print(f"Images Successfully Cropped: {processed_count}")
print(f"Images Skipped (No Hand Detected): {skipped_count}")
print("="*50)