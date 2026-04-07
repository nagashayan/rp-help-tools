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

print("==================================================")
print("🌳 EXTRACTING FEATURES FOR DECISION TREE...")
print("==================================================")

# 1. Setup MediaPipe
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
# Keep num_hands high enough to see both the user and the other person
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=3)
detector = vision.HandLandmarker.create_from_options(options)

BENCHMARK_DIR = "../images/p1_dataset_adversaries"
categories = ["none", "handshake"]

X = [] # This will hold our 4 features: [Reach, Tilt, Thumb_Dist, Wrist_Y]
y = [] # This will hold our labels: 0 for none, 1 for handshake

# 2. Extract Data from Images
for category in categories:
    label = 1 if category == "handshake" else 0
    cat_path = os.path.join(BENCHMARK_DIR, category)
    
    if not os.path.exists(cat_path): 
        print(f"Warning: Could not find path {cat_path}")
        continue
        
    for clip in os.listdir(cat_path):
        clip_path = os.path.join(cat_path, clip)
        if not os.path.isdir(clip_path): continue
            
        for img_path in glob.glob(f"{clip_path}/*.jpg"):
            frame = cv2.imread(img_path)
            if frame is None: continue
                
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
                
                # --- EXTRACT FEATURES for the best hand ---
                dx = best_hand[5].x - best_hand[17].x
                dy = best_hand[5].y - best_hand[17].y
                tilt = abs(math.degrees(math.atan2(dy, dx)))
                
                thumb_tip = np.array([best_hand[4].x, best_hand[4].y])
                index_mcp = np.array([best_hand[5].x, best_hand[5].y])
                thumb_dist = np.linalg.norm(thumb_tip - index_mcp)
                
                wrist_y = best_hand[0].y
                
                # Save to our dataset
                X.append([max_reach_found, tilt, thumb_dist, wrist_y])
                y.append(label)

print(f"Extraction Complete! Extracted {len(X)} valid hand frames.")

if len(X) == 0:
    print("Error: No data extracted. Check your dataset paths.")
    exit()

# 3. Train the Decision Tree
X = np.array(X)
y = np.array(y)

# Limit depth to 4 so it produces human-readable rules instead of memorizing noise
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

# 4. Print the Rules
print("\n==================================================")
print("🤖 DECISION TREE RULES LEARNED:")
print("==================================================")
feature_names = ["Reach_Z", "Palm_Tilt", "Thumb_Distance", "Wrist_Altitude_Y"]
tree_rules = export_text(clf, feature_names=feature_names)
print(tree_rules)

print("\n==================================================")
print("📊 ML ACCURACY ON INDIVIDUAL FRAMES:")
print("==================================================")
y_pred = clf.predict(X)
print(classification_report(y, y_pred, target_names=["No Handshake", "Handshake"]))
