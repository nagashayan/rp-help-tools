import os
import glob
import cv2
import math
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
from collections import deque

print("==================================================")
print("🚀 BENCHMARKING MULTIMODAL FUSION (PIXELS + MATH)")
print("==================================================")

# 1. Load Models
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=3)
detector = vision.HandLandmarker.create_from_options(options)

TFLITE_MODEL_PATH = "dual_input_cnn.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Crucial: A dual-input model has TWO input indices. We must find which is which.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for detail in input_details:
    if len(detail['shape']) == 4:
        img_input_idx = detail['index']
    elif len(detail['shape']) == 2:
        geo_input_idx = detail['index']

# 2. Configuration
BENCHMARK_DIR = "../images/p1_dataset_combined"
categories = ["none", "handshake"]
STABILITY_FRAMES = 5
MAX_WRIST_DRIFT = 0.08
CNN_THRESHOLD = 0.60  # Keeping standard threshold

total_pipeline_time = 0.0
total_frames_processed = 0
tp, tn, fp, fn = 0, 0, 0, 0

# 3. Benchmark Loop
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
        max_cnn_score_recorded = 0.0
        
        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            start_time = time.perf_counter()
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect(mp_image)
            
            stable_hands = []
            
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
                            stable_hands.append({'landmarks': landmarks, 'reach': reach})
            
            if stable_hands:
                best_hand = max(stable_hands, key=lambda x: x['reach'])
                lms = best_hand['landmarks']
                
                # Extract the 4 Geometry Features
                thumb_dist = math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y)
                reach_z = lms[0].z - lms[12].z
                wrist_y = lms[0].y
                dx = lms[5].x - lms[17].x
                dy = lms[5].y - lms[17].y
                palm_tilt = abs(math.degrees(math.atan2(dy, dx)))
                
                # --- CONTINUOUS DUAL INFERENCE ---
                # 1. Prepare Pixels
                frame_resized = cv2.resize(frame_gray, (160, 160))
                img_tensor = np.expand_dims(frame_resized, axis=-1)
                img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32) / 255.0
                
                # 2. Prepare Geometry
                geo_tensor = np.array([[reach_z, wrist_y, palm_tilt]]).astype(np.float32)
                
                # 3. Feed Both to TFLite
                interpreter.set_tensor(img_input_idx, img_tensor)
                interpreter.set_tensor(geo_input_idx, geo_tensor)
                interpreter.invoke()
                cnn_score = interpreter.get_tensor(output_details[0]['index'])[0][0]
                
                if cnn_score > max_cnn_score_recorded:
                    max_cnn_score_recorded = cnn_score
                    
                if cnn_score >= CNN_THRESHOLD:
                    trigger_fired = True
                        
            end_time = time.perf_counter()
            total_pipeline_time += (end_time - start_time)
            total_frames_processed += 1
            
            if trigger_fired:
                break

        # Confusion Matrix
        if trigger_fired and category == "handshake":
            tp += 1
            print(f"[✅ CORRECT - TP] Clip: {clip} | Score: {max_cnn_score_recorded:.2f}")
        elif not trigger_fired and category == "none":
            tn += 1
            print(f"[✅ CORRECT - TN] Clip: {clip} | Score: {max_cnn_score_recorded:.2f}")
        elif trigger_fired and category == "none":
            fp += 1
            print(f"[❌ FAIL - FP] Clip: {clip} | Score: {max_cnn_score_recorded:.2f}")
        else:
            fn += 1
            print(f"[❌ FAIL - FN] Clip: {clip} | Score: {max_cnn_score_recorded:.2f}")

print("\n==================================================")
print("🏁 DUAL-INPUT BENCHMARK COMPLETE")
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

if total_frames_processed > 0:
    avg_latency_ms = (total_pipeline_time / total_frames_processed) * 1000
    est_fps = 1000 / avg_latency_ms if avg_latency_ms > 0 else 0
    print("\n--- ⚡ HARDWARE METRICS ---")
    print(f"Total Frames Processed : {total_frames_processed}")
    print(f"Avg Latency per Frame  : {avg_latency_ms:.2f} ms")
    print(f"Estimated Real-Time FPS: {est_fps:.1f} FPS")
print("==================================================")
