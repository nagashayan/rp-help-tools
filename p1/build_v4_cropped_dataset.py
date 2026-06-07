import os
import cv2
import glob
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("==================================================")
print("✂️ GENERATING HARD-NEGATIVE CROPPED DATASET (BENCHMARK PARITY)")
print("==================================================")

# --- CONFIGURATION ---
INPUT_DATASET_DIR = "../images/train_dataset_v2" # Your original color images
OUTPUT_DATASET_DIR = "../images/train_dataset_v3_cropped"
CATEGORIES = ["none", "handshake"]

# 1. Initialize MediaPipe Tasks Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=4,  
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

for category in CATEGORIES:
    os.makedirs(os.path.join(OUTPUT_DATASET_DIR, category), exist_ok=True)

processed_count = 0
skipped_count = 0

# --- EXACT BENCHMARK LOGIC RANKING ---
def calculate_handshake_score(lms):
    """
    Ranks a hand using the EXACT formulas from benchmark_custom_cnn.py
    """
    score = 0.0
    
    # 1. REACH (Wrist Z - Middle Finger Tip Z)
    reach = lms[0].z - lms[12].z
    # Multiply by 1000 to scale the small float into a meaningful point value
    score += (reach * 1000) 
    
    # 2. TILT (Mid-Prone Check)
    dx = lms[5].x - lms[17].x
    dy = lms[5].y - lms[17].y
    tilt = abs(math.degrees(math.atan2(dy, dx)))
    
    # If it falls within your benchmark's valid tilt range (15 to 165)
    if 20 < tilt < 120:
        score += 50.0  # Massive bonus for having the correct palm angle
        
    # 3. THUMB DISTANCE (Thumb Tip to Index Tip)
    thumb_dist = math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y)
    # Scale distance to points (wider thumb = higher score)
    score += (thumb_dist * 100) 
        
    return score

# --- MAIN PROCESSING LOOP ---
for category in CATEGORIES:
    print(f"\nProcessing category: {category}...")
    
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(INPUT_DATASET_DIR, category, ext)))
        
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None: continue
            
        img_h, img_w, _ = frame.shape
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect(mp_image)
        
        if not results.hand_landmarks:
            skipped_count += 1
            continue
            
        # --- FIND THE HIGHEST RANKING HAND ---
        best_lms = None
        highest_score = -float('inf')
        
        for hand_landmarks in results.hand_landmarks:
            hand_score = calculate_handshake_score(hand_landmarks)
            
            if hand_score > highest_score:
                highest_score = hand_score
                best_lms = hand_landmarks
                
        lms = best_lms
        
        # --- DYNAMIC PROPORTIONAL PADDING ---
        x_coords = [int(lm.x * img_w) for lm in lms]
        y_coords = [int(lm.y * img_h) for lm in lms]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
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
        
        # Stretch box dynamically
        x_min_adj = max(0, x_min - (pad_arm_x if arm_dx < 0 else pad_fingers_x))
        x_max_adj = min(img_w, x_max + (pad_arm_x if arm_dx > 0 else pad_fingers_x))
        y_min_adj = max(0, y_min - (pad_arm_y if arm_dy < 0 else pad_fingers_y))
        y_max_adj = min(img_h, y_max + (pad_arm_y if arm_dy > 0 else pad_fingers_y))
        
        hand_arm_crop = frame[y_min_adj:y_max_adj, x_min_adj:x_max_adj]
        
        if hand_arm_crop.size == 0:
            skipped_count += 1
            continue
            
        crop_resized = cv2.resize(hand_arm_crop, (160, 160))
        
        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DATASET_DIR, category, filename)
        cv2.imwrite(save_path, crop_resized)
        processed_count += 1

print("\n" + "="*50)
print(f"✅ Hard-Negative Cropping Complete!")
print(f"Images Successfully Cropped: {processed_count}")
print(f"Images Skipped: {skipped_count}")
print("="*50)
