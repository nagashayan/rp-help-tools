import os
import glob
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

print("==================================================")
print("🌳 EXTRACTING FEATURES FROM COMBINED DATASETS...")
print("==================================================")

# 1. Setup MediaPipe
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=3)
detector = vision.HandLandmarker.create_from_options(options)

# --- FOLDER PATHS (Adjust STATIC_DIR if necessary) ---
STATIC_DIR = "../images/train_dataset_v2"  
ADVERSARY_DIR = "../images/p1_dataset_adversaries"
categories = ["none", "handshake"]

X = [] 
y = [] 

def process_image(img_path, label):
    frame = cv2.imread(img_path)
    if frame is None: return
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)
    
    if result.hand_landmarks:
        best_hand = None
        max_reach_found = -999.0

        # --- TARGET LOCK: Find the incoming hand ---
        for landmarks in result.hand_landmarks:
            reach = landmarks[0].z - landmarks[12].z
            if reach > max_reach_found:
                max_reach_found = reach
                best_hand = landmarks
        
        # --- EXTRACT FEATURES ---
        dx = best_hand[5].x - best_hand[17].x
        dy = best_hand[5].y - best_hand[17].y
        tilt = abs(math.degrees(math.atan2(dy, dx)))
        
        thumb_tip = np.array([best_hand[4].x, best_hand[4].y])
        index_mcp = np.array([best_hand[5].x, best_hand[5].y])
        thumb_dist = np.linalg.norm(thumb_tip - index_mcp)
        
        wrist_y = best_hand[0].y
        
        X.append([max_reach_found, tilt, thumb_dist, wrist_y])
        y.append(label)

# 2. Extract Data from STATIC Images (Flat Structure)
print(f"Scanning Static Images in {STATIC_DIR}...")
if os.path.exists(STATIC_DIR):
    for category in categories:
        label = 1 if category == "handshake" else 0
        cat_path = os.path.join(STATIC_DIR, category)
        for img_path in glob.glob(f"{cat_path}/*.jpg"):
            process_image(img_path, label)
else:
    print(f"Warning: Static dir {STATIC_DIR} not found.")

# 3. Extract Data from ADVERSARY Sequences (Nested Structure)
print(f"Scanning Adversary Sequences in {ADVERSARY_DIR}...")
if os.path.exists(ADVERSARY_DIR):
    for category in categories:
        label = 1 if category == "handshake" else 0
        cat_path = os.path.join(ADVERSARY_DIR, category)
        if not os.path.exists(cat_path): continue
            
        for clip in os.listdir(cat_path):
            clip_path = os.path.join(cat_path, clip)
            if not os.path.isdir(clip_path): continue
                
            for img_path in glob.glob(f"{clip_path}/*.jpg"):
                process_image(img_path, label)
else:
    print(f"Warning: Adversary dir {ADVERSARY_DIR} not found.")

print(f"\nExtraction Complete! Total valid hand frames: {len(X)}")
if len(X) == 0:
    print("Error: No data extracted. Check your dataset paths.")
    exit()

# 4. Train the Decision Tree
X = np.array(X)
y = np.array(y)

# Limit depth to 4 to keep the rules human-readable
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

# 5. Print the Rules
print("\n==================================================")
print("🤖 DECISION TREE RULES LEARNED:")
print("==================================================")
feature_names = ["Reach_Z", "Palm_Tilt", "Thumb_Distance", "Wrist_Altitude_Y"]
tree_rules = export_text(clf, feature_names=feature_names)
print(tree_rules)

print("\n==================================================")
print("📊 ML ACCURACY ON COMBINED DATASET:")
print("==================================================")
y_pred = clf.predict(X)
print(classification_report(y, y_pred, target_names=["No Handshake", "Handshake"]))