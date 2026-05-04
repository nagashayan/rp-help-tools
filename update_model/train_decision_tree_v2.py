import os
import glob
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("==================================================")
print("🌳 EXTRACTING FEATURES FOR DECISION TREE...")
print("==================================================")

# 1. Configuration
DATASET_DIR = "../images/train_dataset_v2"
CATEGORIES = ["none", "handshake"]
MODEL_SAVE_PATH = "decision_tree_model.pkl"

# 2. Initialize MediaPipe
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=3)
detector = vision.HandLandmarker.create_from_options(options)

X_data = [] # Will hold our [thumb_dist, reach_z, wrist_y, palm_tilt] arrays
y_data = [] # Will hold 0 (none) or 1 (handshake)

valid_frames = 0

# 3. Feature Extraction Loop
for class_value, category in enumerate(CATEGORIES):
    cat_path = os.path.join(DATASET_DIR, category)
    if not os.path.exists(cat_path): 
        print(f"Warning: Path not found {cat_path}")
        continue
        
    images = sorted(glob.glob(f"{cat_path}/*.jpg"))
    
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None: continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_image)
        
        if result.hand_landmarks:
            # --- TARGET LOCK ---
            # If multiple hands exist in the static frame, pick the one reaching the most
            best_hand = max(result.hand_landmarks, key=lambda lms: lms[0].z - lms[12].z)
            
            # --- FEATURE ENGINEERING ---
            # 1. Thumb Distance
            thumb_dist = math.hypot(best_hand[4].x - best_hand[8].x, best_hand[4].y - best_hand[8].y)
            
            # 2. Reach Z
            reach_z = best_hand[0].z - best_hand[12].z
            
            # 3. Wrist Altitude Y
            wrist_y = best_hand[0].y
            
            # 4. Palm Tilt
            dx = best_hand[5].x - best_hand[17].x
            dy = best_hand[5].y - best_hand[17].y
            palm_tilt = abs(math.degrees(math.atan2(dy, dx)))
            
            # Store extracted math
            X_data.append([thumb_dist, reach_z, wrist_y, palm_tilt])
            y_data.append(class_value)
            valid_frames += 1

print(f"Extraction Complete! Extracted {valid_frames} valid hand frames.")

# Convert to Numpy Arrays for Scikit-Learn
X = np.array(X_data)
y = np.array(y_data)

# ==================================================
# 4. Train the Decision Tree
# ==================================================
print("\n==================================================")
print("🤖 TRAINING DECISION TREE...")
print("==================================================")

# We limit max_depth to prevent severe screen-coordinate overfitting 
# and keep the tree rules readable for the IEEE paper.
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
tree_model.fit(X, y)

# ==================================================
# 5. Output Autopsy and Save
# ==================================================
# Print the exact logical rules the tree learned
feature_names = ['Thumb_Distance', 'Reach_Z', 'Wrist_Altitude_Y', 'Palm_Tilt']
tree_rules = export_text(tree_model, feature_names=feature_names)

print("\nDECISION TREE RULES LEARNED:")
print(tree_rules)

# Print Training Accuracy (Note: This is just training accuracy, 
# your benchmark script will find the TRUE video accuracy later)
y_pred = tree_model.predict(X)
print("\n📊 ML ACCURACY ON TRAINING FRAMES:")
print(classification_report(y, y_pred, target_names=["No Handshake", "Handshake"]))

# Save the model so benchmark_decision_tree.py can load it
joblib.dump(tree_model, MODEL_SAVE_PATH)
print(f"\n✅ Success! Decision Tree saved to: {MODEL_SAVE_PATH}")
print("==================================================")