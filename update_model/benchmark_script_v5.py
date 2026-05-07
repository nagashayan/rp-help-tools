import os
import glob
import cv2
import math
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ai_edge_litert.interpreter import Interpreter
from collections import deque

print("==================================================")
print("🚀 BENCHMARKING MACRO-TO-MICRO HYBRID PIPELINE")
print("==================================================")

# 1. Load MediaPipe (The SBF Geometry Extractor)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=3)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Load Custom MobileNetV2 (The Texture Verifier)
TFLITE_MODEL_PATH = "mobilenet_handshake.tflite"
interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Configuration & Thresholds
BENCHMARK_DIR = "../images/p1_dataset_combined"
categories = ["none", "handshake"]

STABILITY_FRAMES = 5
MAX_WRIST_DRIFT = 0.08  
REACH_MIN = 0.03        
TILT_MIN = 20           
TILT_MAX = 120          
THUMB_MIN = 0       
CNN_THRESHOLD = 0.50

# Tracking Variables
total_pipeline_time = 0.0
total_frames_processed = 0
tp, tn, fp, fn = 0, 0, 0, 0

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
            
        history = {'Left': deque(maxlen=STABILITY_FRAMES), 'Right': deque(maxlen=STABILITY_FRAMES)}
        trigger_fired = False
        max_cnn_score_recorded = 0.0
        
        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            # ⏱️ START STOPWATCH
            start_time = time.perf_counter()
            
            img_h, img_w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
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
                            stable_hands.append({
                                'landmarks': landmarks,
                                'reach': reach
                            })
            
            # --- THE HYBRID CASCADE ---
            if stable_hands:
                best_hand = max(stable_hands, key=lambda x: x['reach'])
                lms = best_hand['landmarks']
                reach_val = best_hand['reach']
                
                # STAGE 1: The Geometric Gate
                dx = lms[5].x - lms[17].x
                dy = lms[5].y - lms[17].y
                tilt = abs(math.degrees(math.atan2(dy, dx)))
                thumb_dist = math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y)
                
                if reach_val > REACH_MIN and TILT_MIN < tilt < TILT_MAX and thumb_dist > THUMB_MIN:
                    
                    # --- MACRO TO MICRO CROP ---
                    x_coords = [int(lm.x * img_w) for lm in lms]
                    y_coords = [int(lm.y * img_h) for lm in lms]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    box_w = x_max - x_min
                    box_h = y_max - y_min
                    
                    pad_fingers_x = int(box_w * 0.6)
                    pad_fingers_y = int(box_h * 0.6)
                    pad_arm_x = int(box_w * 1.5)
                    pad_arm_y = int(box_h * 1.5)
                    
                    wrist_x, wrist_y = int(lms[0].x * img_w), int(lms[0].y * img_h)
                    mid_x, mid_y = int(lms[9].x * img_w), int(lms[9].y * img_h)
                    
                    arm_dx = wrist_x - mid_x
                    arm_dy = wrist_y - mid_y
                    
                    x_min_adj = max(0, x_min - (pad_arm_x if arm_dx < 0 else pad_fingers_x))
                    x_max_adj = min(img_w, x_max + (pad_arm_x if arm_dx > 0 else pad_fingers_x))
                    y_min_adj = max(0, y_min - (pad_arm_y if arm_dy < 0 else pad_fingers_y))
                    y_max_adj = min(img_h, y_max + (pad_arm_y if arm_dy > 0 else pad_fingers_y))
                    
                    # Crop using RGB for MobileNet
                    hand_arm_crop = frame_rgb[y_min_adj:y_max_adj, x_min_adj:x_max_adj]
                    
                    if hand_arm_crop.size > 0:
                        # STAGE 2: The CNN Verifier
                        frame_resized = cv2.resize(hand_arm_crop, (160, 160))
                        input_tensor = frame_resized.astype(np.float32)
                        
                        # MobileNet preprocess_input: Scale pixels from [0, 255] to [-1.0, 1.0]
                        input_tensor = (input_tensor / 127.5) - 1.0
                        input_tensor = np.expand_dims(input_tensor, axis=0)
                        
                        interpreter.set_tensor(input_details[0]['index'], input_tensor)
                        interpreter.invoke()
                        cnn_score = interpreter.get_tensor(output_details[0]['index'])[0][0]
                        
                        if cnn_score > max_cnn_score_recorded:
                            max_cnn_score_recorded = cnn_score
                            
                        if cnn_score >= CNN_THRESHOLD:
                            trigger_fired = True
                        
            # ⏱️ STOP STOPWATCH
            end_time = time.perf_counter()
            total_pipeline_time += (end_time - start_time)
            total_frames_processed += 1
            
            if trigger_fired:
                break # Success! Move to next clip

        # Confusion Matrix Tally
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

# ==========================================
# 📊 CALCULATE FINAL METRICS
# ==========================================
print("\n==================================================")
print("🏁 HYBRID BENCHMARK COMPLETE")
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