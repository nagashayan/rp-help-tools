import os
import glob
import cv2
import math
import numpy as np
import time  
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import joblib

print("==================================================")
print("🌳 BENCHMARKING DECISION TREE (PURE GEOMETRY)")
print("==================================================")

# 1. Load MediaPipe
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=3)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Load Trained Decision Tree
DT_MODEL_PATH = "decision_tree_model.pkl" 
tree_model = joblib.load(DT_MODEL_PATH)

# 3. Configuration
BENCHMARK_DIR = "../images/p1_dataset_combined"
categories = ["none", "handshake"]
STABILITY_FRAMES = 5
MAX_WRIST_DRIFT = 0.08

# --- Tracking Variables ---
total_pipeline_time = 0.0
total_frames_processed = 0

# Counters for our confusion matrix
tp, tn, fp, fn = 0, 0, 0, 0

for category in categories:
    cat_path = os.path.join(BENCHMARK_DIR, category)
    if not os.path.exists(cat_path): continue
        
    print(f"\nEvaluating Category: {category.upper()}")
    
    for clip in os.listdir(cat_path):
        clip_path = os.path.join(cat_path, clip)
        if not os.path.isdir(clip_path): continue
            
        images = sorted(glob.glob(f"{clip_path}/*.jpg"))
        if not images: continue
            
        history = {'Left': deque(maxlen=STABILITY_FRAMES), 'Right': deque(maxlen=STABILITY_FRAMES)}
        trigger_fired = False
        
        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            # ⏱️ START STOPWATCH
            start_time = time.perf_counter()
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect(mp_image)
            
            stable_hands = []
            prediction = 0 
            
            if result.hand_landmarks and result.handedness:
                for idx, landmarks in enumerate(result.hand_landmarks):
                    handedness = result.handedness[idx][0].category_name
                    wrist = landmarks[0]
                    history[handedness].append((wrist.x, wrist.y))
                    
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
            
            # --- TARGET LOCK ---
            if stable_hands:
                best_hand = max(stable_hands, key=lambda x: x['reach'])
                lms = best_hand['landmarks']
                
                # --- FEATURE ENGINEERING ---
                thumb_dist = math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y)
                reach_z = lms[0].z - lms[12].z
                wrist_y = lms[0].y
                
                dx = lms[5].x - lms[17].x
                dy = lms[5].y - lms[17].y
                palm_tilt = abs(math.degrees(math.atan2(dy, dx)))
                
                # --- DECISION TREE INFERENCE ---
                features = np.array([[thumb_dist, reach_z, wrist_y, palm_tilt]])
                prediction = tree_model.predict(features)[0]
                
            # ⏱️ STOP STOPWATCH
            end_time = time.perf_counter()
            total_pipeline_time += (end_time - start_time)
            total_frames_processed += 1
            
            if prediction == 1: 
                trigger_fired = True
                break 

        # Tally the results for the Confusion Matrix
        if trigger_fired and category == "handshake":
            tp += 1
            print(f"[✅ CORRECT - TP] Clip: {clip}")
        elif not trigger_fired and category == "none":
            tn += 1
            print(f"[✅ CORRECT - TN] Clip: {clip}")
        elif trigger_fired and category == "none":
            fp += 1
            print(f"[❌ FAIL - FP] Clip: {clip}")
        else:
            fn += 1
            print(f"[❌ FAIL - FN] Clip: {clip}")

# ==========================================
# 📊 CALCULATE FINAL METRICS
# ==========================================
print("\n==================================================")
print("🏁 DT BENCHMARK COMPLETE")
print("==================================================")

total_clips = tp + tn + fp + fn
accuracy = (tp + tn) / total_clips if total_clips > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print("--- 📈 STATISTICAL METRICS ---")
print(f"Total Clips Evaluated: {total_clips}")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")

print("\n--- ⚡ HARDWARE METRICS ---")
if total_frames_processed > 0:
    avg_latency_ms = (total_pipeline_time / total_frames_processed) * 1000
    est_fps = 1000 / avg_latency_ms if avg_latency_ms > 0 else 0
    print(f"Total Frames Processed : {total_frames_processed}")
    print(f"Avg Latency per Frame  : {avg_latency_ms:.2f} ms")
    print(f"Estimated Real-Time FPS: {est_fps:.1f} FPS")
else:
    print("No frames were processed.")
print("==================================================")
