import os
import glob
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
from collections import deque

print("==================================================")
print("🚀 INITIALIZING HYBRID GATED PIPELINE (IEEE FINAL)")
print("==================================================")

# 1. Load MediaPipe (The SBF Geometry Extractor)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=3)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Load Custom 1-Channel CNN (The Texture Verifier)
TFLITE_MODEL_PATH = "custom_handshake_cnn.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Configuration & Thresholds
BENCHMARK_DIR = "../images/p1_dataset_adversaries"
categories = ["none", "handshake"]

STABILITY_FRAMES = 5
MAX_WRIST_DRIFT = 0.08  # Wrist can't move more than 8% of screen width/height
REACH_MIN = 0.03        # Relaxed SBF: Must be reaching forward at least a little
TILT_MIN = 15           # Relaxed SBF: Palm can't be perfectly flat
TILT_MAX = 165          # Relaxed SBF: Palm can't be perfectly flat
CNN_THRESHOLD = 0.60    # CNN final verdict threshold

def get_hand_bbox(landmarks, img_w, img_h):
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    
    # Add 20% padding
    pad_x = (xmax - xmin) * 0.2
    pad_y = (ymax - ymin) * 0.2
    
    px_xmin = max(0, int((xmin - pad_x) * img_w))
    px_xmax = min(img_w, int((xmax + pad_x) * img_w))
    px_ymin = max(0, int((ymin - pad_y) * img_h))
    px_ymax = min(img_h, int((ymax + pad_y) * img_h))
    
    return px_xmin, px_ymin, px_xmax, px_ymax

# 4. Benchmark Loop
for category in categories:
    cat_path = os.path.join(BENCHMARK_DIR, category)
    if not os.path.exists(cat_path): continue
        
    print(f"\nEvaluating Category: {category.upper()}")
    
    for clip in os.listdir(cat_path):
        clip_path = os.path.join(cat_path, clip)
        if not os.path.isdir(clip_path): continue
            
        images = sorted(glob.glob(f"{clip_path}/*.jpg"))
        if not images: continue
            
        # --- DUAL MEMORY BANKS ---
        # We track left and right hand stability independently to solve index swapping!
        history = {'Left': deque(maxlen=STABILITY_FRAMES), 'Right': deque(maxlen=STABILITY_FRAMES)}
        
        trigger_fired = False
        max_cnn_score_recorded = 0.0
        
        for img_path in images:
            frame = cv2.imread(img_path)
            img_h, img_w, _ = frame.shape
            
            # Convert to RGB for MediaPipe, and Grayscale for our CNN
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect(mp_image)
            
            stable_hands = []
            
            if result.hand_landmarks and result.handedness:
                # 1. Update Dual Memory Banks
                for idx, landmarks in enumerate(result.hand_landmarks):
                    handedness = result.handedness[idx][0].category_name
                    wrist = landmarks[0]
                    history[handedness].append((wrist.x, wrist.y))
                    
                    # 2. Check Temporal Stability
                    if len(history[handedness]) == STABILITY_FRAMES:
                        xs = [p[0] for p in history[handedness]]
                        ys = [p[1] for p in history[handedness]]
                        drift = max(max(xs) - min(xs), max(ys) - min(ys))
                        
                        if drift <= MAX_WRIST_DRIFT:
                            reach = landmarks[0].z - landmarks[12].z
                            stable_hands.append({
                                'landmarks': landmarks,
                                'reach': reach
                            })
            
            # 3. Target Lock: If multiple hands are stable, pick the one reaching the most
            if stable_hands:
                best_hand = max(stable_hands, key=lambda x: x['reach'])
                lms = best_hand['landmarks']
                reach_val = best_hand['reach']
                
                # 4. The SBF Gate (Relaxed Geometry)
                dx = lms[5].x - lms[17].x
                dy = lms[5].y - lms[17].y
                tilt = abs(math.degrees(math.atan2(dy, dx)))
                
                # If it's reaching AND palm is somewhat sideways -> Open the Gate!
                if reach_val > REACH_MIN and TILT_MIN < tilt < TILT_MAX:
                    
                    # 5. The Texture Verifier (Custom 1-Channel CNN)
                    # FIX: Pass the FULL frame just like we did during training!
                    frame_resized = cv2.resize(frame_gray, (160, 160))
                    
                    # Format for TFLite: shape (1, 160, 160, 1)
                    input_tensor = np.expand_dims(frame_resized, axis=-1)
                    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
                    
                    interpreter.set_tensor(input_details[0]['index'], input_tensor)
                    interpreter.invoke()
                    cnn_score = interpreter.get_tensor(output_details[0]['index'])[0][0]
                    
                    if cnn_score > max_cnn_score_recorded:
                        max_cnn_score_recorded = cnn_score
                        
                    # Final Trigger
                    if cnn_score >= CNN_THRESHOLD:
                        trigger_fired = True
                        break # Success! Move to next clip

        # Autopsy Report
        status = "✅ CORRECT" if (trigger_fired and category == "handshake") or (not trigger_fired and category == "none") else "❌ FAIL"
        print(f"[{status}] Clip: {clip} | Triggered: {trigger_fired} | Max CNN Score: {max_cnn_score_recorded:.3f}")

print("\n==================================================")
print("🏁 BENCHMARK COMPLETE")
print("==================================================")