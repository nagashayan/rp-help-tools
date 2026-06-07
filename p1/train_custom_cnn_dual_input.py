import os
import glob
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

print("==================================================")
print("🧠 BUILDING DUAL-INPUT (FUSION) NEURAL NETWORK")
print("==================================================")

DATASET_DIR = "../images/train_dataset_v2" 
CATEGORIES = ["none", "handshake"]
MODEL_SAVE_PATH = "dual_input_cnn.tflite"

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=3)
detector = vision.HandLandmarker.create_from_options(options)

X_images = []
X_geometry = []
y_data = []

# --- 1. MULTIMODAL EXTRACTION ---
print("Extracting Pixels + Geometry...")
for class_value, category in enumerate(CATEGORIES):
    cat_path = os.path.join(DATASET_DIR, category)
    if not os.path.exists(cat_path): continue
        
    for img_path in glob.glob(f"{cat_path}/*.jpg"):
        frame = cv2.imread(img_path)
        if frame is None: continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_image)
        
        if result.hand_landmarks:
            best_hand = max(result.hand_landmarks, key=lambda lms: lms[0].z - lms[12].z)
            
            # Geometry Branch Data
            thumb_dist = math.hypot(best_hand[4].x - best_hand[8].x, best_hand[4].y - best_hand[8].y)
            reach_z = best_hand[0].z - best_hand[12].z
            wrist_y = best_hand[0].y
            dx = best_hand[5].x - best_hand[17].x
            dy = best_hand[5].y - best_hand[17].y
            palm_tilt = abs(math.degrees(math.atan2(dy, dx)))
            
            # Pixel Branch Data
            frame_resized = cv2.resize(frame_gray, (160, 160))
            
            X_geometry.append([reach_z, wrist_y, palm_tilt])
            X_images.append(frame_resized)
            y_data.append(class_value)

X_img_arr = np.array(X_images).reshape(-1, 160, 160, 1).astype('float32') / 255.0
X_geo_arr = np.array(X_geometry).astype('float32')
y_arr = np.array(y_data).astype('float32')

print(f"Extraction Complete! {len(y_arr)} valid multimodal frames.")

# --- 2. BUILD THE TWO-BRANCH ARCHITECTURE ---
print("\nCompiling Dual-Branch CNN...")

# Branch A: The Texture Extractor (CNN)
img_input = Input(shape=(160, 160, 1), name="pixel_input")
x1 = Conv2D(16, (3,3), activation='relu')(img_input)
x1 = MaxPooling2D(2,2)(x1)
x1 = Conv2D(32, (3,3), activation='relu')(x1)
x1 = MaxPooling2D(2,2)(x1)
x1 = Flatten()(x1)
x1 = Dense(32, activation='relu')(x1)

# Branch B: The Geometry Extractor
geo_input = Input(shape=(3,), name="geometry_input")
x2 = Dense(16, activation='relu')(geo_input)

# Early Fusion: Merge the math and the pixels
merged = Concatenate()([x1, x2])
fused_dense = Dense(32, activation='relu')(merged)
output = Dense(1, activation='sigmoid')(fused_dense)

model = Model(inputs=[img_input, geo_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 3. TRAIN AND CONVERT ---
print("\nTraining Dual-Input Model...")
model.fit({"pixel_input": X_img_arr, "geometry_input": X_geo_arr}, y_arr, epochs=10, batch_size=16)

print("\nConverting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(MODEL_SAVE_PATH, 'wb') as f:
    f.write(tflite_model)
    
print(f"✅ Success! Dual-Input TFLite saved to {MODEL_SAVE_PATH}")
print("==================================================")