import os
import glob
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ai_edge_litert.interpreter import Interpreter
from collections import deque

print("==================================================")
print("📸 EXTRACTING PAPER EVALUATION FRAMES (CLEAN VERSION)")
print("==================================================")

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
BENCHMARK_DIR = "../images/p1_dataset_combined"
OUTPUT_DIR = "../images/paper_collage_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe Setup (Tasks API Only)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# TFLite Setup (MobileNetV2)
TFLITE_MODEL_PATH = "mobilenet_handshake.tflite"
interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Hardcoded logic thresholds from your benchmark script
STABILITY_FRAMES = 5
MAX_WRIST_DRIFT = 0.08  
TILT_MIN = 60           
TILT_MAX = 120          
CNN_THRESHOLD = 0.60

# Manual skeletal connections to bypass the missing 'solutions' module
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

captured = {"TP": False, "TN": False, "FP": False, "FN": False}

# ==========================================
# 2. DRAWING FUNCTION (CUSTOM OPENCV)
# ==========================================
def draw_paper_overlay(frame, lms, condition, text, color):
    """Manually draws the skeleton and text labels."""
    h, w, _ = frame.shape
    
    # 1. Draw Skeleton Lines
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection[0], connection[1]
        start_point = (int(lms[start_idx].x * w), int(lms[start_idx].y * h))
        end_point = (int(lms[end_idx].x * w), int(lms[end_idx].y * h))
        cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
        
    # 2. Draw Skeleton Joints
    for lm in lms:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    
    # 3. Draw Text Box
    cv2.rectangle(frame, (10, 10), (550, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"{condition}: {text}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return frame

# ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================
categories = ["none", "handshake"]

for category in categories:
    if all(captured.values()): break
    
    cat_path = os.path.join(BENCHMARK_DIR, category)
    if not os.path.exists(cat_path): continue
    
    for clip in os.listdir(cat_path):
        if all(captured.values()): break
        clip_path = os.path.join(cat_path, clip)
        if not os.path.isdir(clip_path): continue
            
        images = sorted(glob.glob(f"{clip_path}/*.jpg"))
        history = {'Left': deque(maxlen=STABILITY_FRAMES), 'Right': deque(maxlen=STABILITY_FRAMES)}
        
        for img_path in images:
            if all(captured.values()): break
                
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            img_h, img_w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect(mp_image)
            
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
                            # We have a stable hand! 
                            lms = landmarks
                            
                            # Calculate Tilt
                            dx = lms[5].x - lms[17].x
                            dy = lms[5].y - lms[17].y
                            tilt = abs(math.degrees(math.atan2(dy, dx)))
                            
                            passed_tilt = (TILT_MIN <= tilt <= TILT_MAX)
                            cnn_score = 0.0
                            
                            # Macro-to-Micro Crop (Mirrors your v5 builder exact logic)
                            x_coords = [int(lm.x * img_w) for lm in lms]
                            y_coords = [int(lm.y * img_h) for lm in lms]
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            
                            box_w, box_h = x_max - x_min, y_max - y_min
                            pad_fingers_x, pad_fingers_y = int(box_w * 0.6), int(box_h * 0.6)
                            pad_arm_x, pad_arm_y = int(box_w * 1.5), int(box_h * 1.5)
                            
                            wrist_x, wrist_y = int(lms[0].x * img_w), int(lms[0].y * img_h)
                            mid_x, mid_y = int(lms[9].x * img_w), int(lms[9].y * img_h)
                            arm_dx, arm_dy = wrist_x - mid_x, wrist_y - mid_y
                            
                            x_min_adj = max(0, x_min - (pad_arm_x if arm_dx < 0 else pad_fingers_x))
                            x_max_adj = min(img_w, x_max + (pad_arm_x if arm_dx > 0 else pad_fingers_x))
                            y_min_adj = max(0, y_min - (pad_arm_y if arm_dy < 0 else pad_fingers_y))
                            y_max_adj = min(img_h, y_max + (pad_arm_y if arm_dy > 0 else pad_fingers_y))
                            
                            hand_arm_crop = frame_rgb[y_min_adj:y_max_adj, x_min_adj:x_max_adj]
                            
                            if passed_tilt and hand_arm_crop.size > 0:
                                frame_resized = cv2.resize(hand_arm_crop, (160, 160))
                                input_tensor = (frame_resized.astype(np.float32) / 127.5) - 1.0
                                input_tensor = np.expand_dims(input_tensor, axis=0)
                                
                                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                                interpreter.invoke()
                                cnn_score = interpreter.get_tensor(output_details[0]['index'])[0][0]

                            # --- SCENARIO CAPTURE LOGIC ---
                            is_gt_handshake = (category == "handshake")
                            predicted = (cnn_score >= CNN_THRESHOLD)
                            
                            # 1. TRUE POSITIVE (GT: Handshake, Pred: Pass)
                            if is_gt_handshake and predicted and not captured["TP"]:
                                res = draw_paper_overlay(frame, lms, "TP", f"Valid Handshake (CNN: {cnn_score:.2f})", (0, 255, 0))
                                cv2.imwrite(os.path.join(OUTPUT_DIR, "A_True_Positive.jpg"), res)
                                captured["TP"] = True
                                print("✅ Captured True Positive")

                            # 2. TRUE NEGATIVE (GT: None, Blocked by Tilt Gate)
                            elif not is_gt_handshake and not passed_tilt and not captured["TN"]:
                                res = draw_paper_overlay(frame, lms, "TN", f"Blocked by GRL Tilt ({int(tilt)} deg)", (255, 200, 0))
                                cv2.imwrite(os.path.join(OUTPUT_DIR, "B_True_Negative_GRL.jpg"), res)
                                captured["TN"] = True
                                print("✅ Captured True Negative (GRL Block)")

                            # 3. FALSE POSITIVE (GT: None, Pred: Pass)
                            elif not is_gt_handshake and predicted and not captured["FP"]:
                                res = draw_paper_overlay(frame, lms, "FP", f"CNN Hallucination ({cnn_score:.2f})", (0, 0, 255))
                                cv2.imwrite(os.path.join(OUTPUT_DIR, "C_False_Positive.jpg"), res)
                                captured["FP"] = True
                                print("✅ Captured False Positive")

                            # 4. FALSE NEGATIVE (GT: Handshake, Pred: Fail)
                            elif is_gt_handshake and not predicted and not captured["FN"]:
                                reason = "Blocked by GRL Tilt" if not passed_tilt else f"CNN Rejected ({cnn_score:.2f})"
                                res = draw_paper_overlay(frame, lms, "FN", reason, (0, 165, 255))
                                cv2.imwrite(os.path.join(OUTPUT_DIR, "D_False_Negative.jpg"), res)
                                captured["FN"] = True
                                print("✅ Captured False Negative")

print("\n==================================================")
print(f"🏁 Execution Complete! Check the '{OUTPUT_DIR}' folder.")
print("==================================================")