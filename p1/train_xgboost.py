import os
import glob
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import xgboost as xgb
from sklearn.metrics import classification_report
import joblib

print("==================================================")
print("🚀 EXTRACTING FEATURES FOR XGBOOST...")
print("==================================================")

# 1. Configuration
DATASET_DIR = "../images/train_dataset_v2" 
CATEGORIES = ["none", "handshake"]
MODEL_SAVE_PATH = "xgboost_model.pkl"

# 2. Initialize MediaPipe
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=3)
detector = vision.HandLandmarker.create_from_options(options)

X_data = [] 
y_data = [] 
valid_frames = 0

# 3. Feature Extraction Loop
for class_value, category in enumerate(CATEGORIES):
    cat_path = os.path.join(DATASET_DIR, category)
    if not os.path.exists(cat_path): continue
        
    images = sorted(glob.glob(f"{cat_path}/*.jpg"))
    
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None: continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_image)
        
        if result.hand_landmarks:
            best_hand = max(result.hand_landmarks, key=lambda lms: lms[0].z - lms[12].z)
            
            thumb_dist = math.hypot(best_hand[4].x - best_hand[8].x, best_hand[4].y - best_hand[8].y)
            reach_z = best_hand[0].z - best_hand[12].z
            wrist_y = best_hand[0].y
            dx = best_hand[5].x - best_hand[17].x
            dy = best_hand[5].y - best_hand[17].y
            palm_tilt = abs(math.degrees(math.atan2(dy, dx)))
            
            X_data.append([thumb_dist, reach_z, wrist_y, palm_tilt])
            y_data.append(class_value)
            valid_frames += 1

print(f"Extraction Complete! Extracted {valid_frames} valid hand frames.")

X = np.array(X_data)
y = np.array(y_data)

# ==================================================
# 4. Train XGBoost
# ==================================================
print("\n==================================================")
print("🤖 TRAINING XGBOOST...")
print("==================================================")

# XGBoost requires slightly different hyperparameter naming
xgb_model = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=5, 
    learning_rate=0.1, 
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='logloss',
    scale_pos_weight=1.0 # Adjust this if classes are highly imbalanced
)
xgb_model.fit(X, y)

# ==================================================
# 5. Output Autopsy and Save
# ==================================================
y_pred = xgb_model.predict(X)
print("\n📊 ML ACCURACY ON TRAINING FRAMES:")
print(classification_report(y, y_pred, target_names=["No Handshake", "Handshake"]))

print("\n🔍 FEATURE IMPORTANCES:")
feature_names = ['Thumb_Distance', 'Reach_Z', 'Wrist_Altitude_Y', 'Palm_Tilt']
importances = xgb_model.feature_importances_
for name, importance in zip(feature_names, importances):
    print(f" - {name}: {importance * 100:.2f}%")

joblib.dump(xgb_model, MODEL_SAVE_PATH)
print(f"\n✅ Success! XGBoost saved to: {MODEL_SAVE_PATH}")
print("==================================================")