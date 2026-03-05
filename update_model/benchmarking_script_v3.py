import numpy as np
import cv2
import math
import mediapipe as mp
import time
import psutil
import glob
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# --- MAC OPTIMIZATION: Using standard TensorFlow Lite ---
from tensorflow.lite.python.interpreter import Interpreter

# ==========================================
# Configurations & Settings
# ==========================================
WINDOW_NAME = "Handshake Z-Vector Logic (MAC BENCHMARK)"
WINDOW_WIDTH = 450
WINDOW_HEIGHT = 850

# Temporal Settings
WINDOW_SIZE = 10
THRESHOLD = 0.50
prediction_queue = deque(maxlen=WINDOW_SIZE)

STABILITY_HISTORY = 10
coord_history = deque(maxlen=STABILITY_HISTORY)
MAX_STABILITY_STD = 0.060

# Logic Thresholds (The SBF Gates)
REACH_THRESHOLD = 0.05
THUMB_OPEN_THRESHOLD = 0.06
TILT_MIN = 20   # Widened for extreme perspectives
TILT_MAX = 160  # Widened for extreme perspectives

HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),      
    (0, 5), (5, 6), (6, 7), (7, 8),      
    (5, 9), (9, 10), (10, 11), (11, 12), 
    (9, 13), (13, 14), (14, 15), (15, 16), 
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
])

# ==========================================
# Initialization
# ==========================================
def init_mediapipe():
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=3, 
        running_mode=vision.RunningMode.VIDEO 
    )
    return vision.HandLandmarker.create_from_options(options)

detector = init_mediapipe()

# --- Load the TFLite Model ---
interpreter = Interpreter(model_path="handshake_model_optimized.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL) 
cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT) 

# ==========================================
# Helper Functions
# ==========================================
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_pointing_vector(landmarks):
    return landmarks[0].z - landmarks[12].z

def get_palm_tilt(landmarks):
    dx = landmarks[5].x - landmarks[17].x
    dy = landmarks[5].y - landmarks[17].y
    return abs(math.degrees(math.atan2(dy, dx)))

def check_thumb_open(landmarks):
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    index_mcp = np.array([landmarks[5].x, landmarks[5].y])
    return np.linalg.norm(thumb_tip - index_mcp) > THUMB_OPEN_THRESHOLD

def draw_skeleton(frame, landmarks, w, h):
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(frame, p1, p2, (255, 0, 255), 2)
        cv2.circle(frame, p1, 4, (0, 255, 0), -1)

def preprocess_input_edge(x):
    x = x.astype(np.float32)
    x /= 127.5
    x -= 1.0
    return x

# ==========================================
# Benchmark Setup
# ==========================================
BENCHMARK_DIR = "../images/sequence_datasetv3"
if not os.path.exists(BENCHMARK_DIR):
    print(f"ERROR: Please create the folder structure '{BENCHMARK_DIR}/handshake' and '{BENCHMARK_DIR}/none'.")
    exit()

# Metric Trackers
total_neural_ms = 0.0
total_sbf_ms = 0.0
total_cnn_ms = 0.0
valid_frame_count = 0
global_frame_tracker = 0
WARMUP_FRAMES = 5 

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

print("==================================================")
print("🚀 STARTING NESTED BATCH BENCHMARK ON MAC (WITH AUTOPSY)...")
print("==================================================")

categories = sorted([d for d in os.listdir(BENCHMARK_DIR) if os.path.isdir(os.path.join(BENCHMARK_DIR, d))])

# ==========================================
# Main Batch Loop
# ==========================================
for category in categories:
    category_path = os.path.join(BENCHMARK_DIR, category)
    is_actual_handshake = "handshake" in category.lower()
    
    clip_folders = sorted([d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))])
    
    for clip_name in clip_folders:
        clip_path = os.path.join(category_path, clip_name)
        image_paths = sorted(glob.glob(f"{clip_path}/*.jpg"))
        
        if not image_paths:
            continue

        system_triggered_handshake = False

        # Reset the temporal queues and latches for the new sequence
        coord_history.clear()
        prediction_queue.clear()
        sbf_latch_counter = 0 # NEW: Reset the latch for each clip
        
        # --- NEW: Clip-Level Diagnostics Trackers ---
        clip_max_reach = 0.0
        clip_min_tilt, clip_max_tilt = 999.0, 0.0
        clip_thumb_open = False
        clip_max_stability = 0.0
        clip_max_cnn = 0.0
        clip_max_avg_conf = 0.0
        clip_sbf_passed = False

        print(f"Processing {category}/{clip_name} ({len(image_paths)} frames)...")

        for img_path in image_paths:
            global_frame_tracker += 1
            
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            
            # --- PROBE 1: MediaPipe ---
            t_start = time.perf_counter()
            detection_result = detector.detect_for_video(mp_image, timestamp)
            neural_ms = (time.perf_counter() - t_start) * 1000

            # --- PROBE 2: SBF Logic (MULTI-HAND DYNAMIC GATE) ---
            t_start = time.perf_counter()
            
            target_landmarks = None
            is_reaching = is_open = is_vertical = False
            pose_score, stability_score = 0.0, 0.0
            
            if detection_result.hand_landmarks:
                for landmarks in detection_result.hand_landmarks:
                    reach_val = get_pointing_vector(landmarks)
                    temp_reaching = reach_val > REACH_THRESHOLD
                    
                    palm_tilt = get_palm_tilt(landmarks)
                    temp_vertical = TILT_MIN < palm_tilt < TILT_MAX
                    temp_open = check_thumb_open(landmarks)
                    
                    # Log highest values seen in the frame for the autopsy
                    clip_max_reach = max(clip_max_reach, reach_val)
                    clip_min_tilt = min(clip_min_tilt, palm_tilt)
                    clip_max_tilt = max(clip_max_tilt, palm_tilt)
                    if temp_open: clip_thumb_open = True
                    
                    if temp_reaching and temp_open and temp_vertical:
                        target_landmarks = landmarks
                        is_reaching, is_open, is_vertical = True, True, True
                        pose_score = 1.0
                        sbf_latch_counter = 5  # NEW: Keep gate open for 5 frames
                        clip_sbf_passed = True
                        break 
                
                # NEW: If strict gate failed, but latch is active, keep it open!
                if not target_landmarks and sbf_latch_counter > 0:
                    target_landmarks = detection_result.hand_landmarks[0]
                    is_reaching, is_open, is_vertical = True, True, True
                    pose_score = 0.5 # Slightly lower score while coasting
                    sbf_latch_counter -= 1
                elif not target_landmarks:
                    target_landmarks = detection_result.hand_landmarks[0]

                draw_skeleton(frame, target_landmarks, w, h)
                wrist = target_landmarks[0]
                coord_history.append((wrist.x, wrist.y))
                
                if len(coord_history) == STABILITY_HISTORY:
                    avg_std = (np.std([c[0] for c in coord_history]) + np.std([c[1] for c in coord_history])) / 2
                    stability_score = np.clip(1.0 - (avg_std / MAX_STABILITY_STD), 0, 1)
            else:
                coord_history.clear()
                
            sbf_ms = (time.perf_counter() - t_start) * 1000

            # --- PROBE 3: TFLITE CNN Inference (SBF GATED) ---
            t_start = time.perf_counter()
            
            if is_reaching and is_open and is_vertical:
                gray_1c = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                gray_3c = cv2.cvtColor(gray_1c, cv2.COLOR_GRAY2RGB)
                
                img_cnn = cv2.resize(gray_3c, (160, 160))
                img_cnn = cv2.GaussianBlur(img_cnn, (5, 5), 0)
                img_cnn = preprocess_input_edge(img_cnn) 
                img_cnn = np.expand_dims(img_cnn, axis=0)
                
                interpreter.set_tensor(input_details[0]['index'], img_cnn)
                interpreter.invoke()
                cnn_raw = interpreter.get_tensor(output_details[0]['index'])[0][0]
            else:
                cnn_raw = 0.0
                
            cnn_ms = (time.perf_counter() - t_start) * 1000

            # --- FUSION & ACCURACY TRACKING ---
            # 1. Calculate the instantaneous fused score for THIS frame
            fused_pred = (0.4 * cnn_raw) + (0.4 * pose_score) + (0.2 * stability_score)

            # 2. Update Autopsy Trackers
            clip_max_stability = max(clip_max_stability, stability_score)
            clip_max_cnn = max(clip_max_cnn, cnn_raw)
            clip_max_avg_conf = max(clip_max_avg_conf, fused_pred)

            # 3. THE INSTANT LATCH: If THIS frame is good, and the wrist has been STABLE for 10 frames, trigger!
            if fused_pred > THRESHOLD and stability_score > 0.6:
                system_triggered_handshake = True
                
            # Keep the UI colors updated
            if system_triggered_handshake:
                label, color, text_x = "HANDSHAKE", (50, 255, 50), 100
            else:
                label, color, text_x = "NO HANDSHAKE", (50, 50, 255), 60

            # --- METRICS ACCUMULATION ---
            if global_frame_tracker > WARMUP_FRAMES:
                total_neural_ms += neural_ms
                total_sbf_ms += sbf_ms
                total_cnn_ms += cnn_ms
                valid_frame_count += 1

            # --- UI DISPLAY ---
            scale = WINDOW_HEIGHT / h
            new_w = int(w * scale)
            resized_frame = cv2.resize(frame, (new_w, WINDOW_HEIGHT))
            start_x = max(0, (new_w // 2) - (WINDOW_WIDTH // 2))
            display_frame = resized_frame[:, start_x:start_x+WINDOW_WIDTH]
            display_frame = cv2.resize(display_frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, 300), (20, 20, 20), -1) 
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(display_frame, f"CNN Confidence: {cnn_raw:.2f}", (20, 30), font, 0.5, (240,240,240), 1)
            cv2.putText(display_frame, f"Pose Score:     {pose_score:.2f}", (20, 60), font, 0.5, (240,240,240), 1)
            cv2.putText(display_frame, f"Stability:      {stability_score:.2f}", (20, 90), font, 0.5, (240,240,240), 1)
            cv2.putText(display_frame, f"Neural Latency: {neural_ms:.1f} ms", (20, 150), font, 0.4, (0, 255, 255), 1)
            cv2.putText(display_frame, f"SBF Logic:      {sbf_ms:.2f} ms", (20, 170), font, 0.4, (0, 255, 255), 1)
            cv2.putText(display_frame, f"CNN Latency:    {cnn_ms:.1f} ms", (20, 190), font, 0.4, (0, 255, 255), 1)

            cv2.rectangle(display_frame, (0, WINDOW_HEIGHT-70), (WINDOW_WIDTH, WINDOW_HEIGHT), (0,0,0), -1)
            cv2.putText(display_frame, label, (text_x, WINDOW_HEIGHT-25), font, 1.2, color, 2)

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        # Clip-Level Accuracy Evaluation & Autopsy Reporting
        if is_actual_handshake and system_triggered_handshake:
            true_positives += 1
        elif not is_actual_handshake and system_triggered_handshake:
            false_positives += 1
        elif not is_actual_handshake and not system_triggered_handshake:
            true_negatives += 1
        elif is_actual_handshake and not system_triggered_handshake:
            false_negatives += 1
            # --- THE AUTOPSY PRINT ---
            print(f"\n❌ FALSE NEGATIVE AUTOPSY: {category}/{clip_name}")
            print(f"   -> SBF Gate Ever Opened? : {clip_sbf_passed}")
            print(f"   -> Max Reach (Needs >{REACH_THRESHOLD}): {clip_max_reach:.3f}")
            print(f"   -> Palm Tilt (Needs {TILT_MIN}-{TILT_MAX}): {clip_min_tilt:.1f} to {clip_max_tilt:.1f}")
            print(f"   -> Thumb Ever Open?      : {clip_thumb_open}")
            print(f"   -> Max Stability Score   : {clip_max_stability:.3f} (Needs >0.6)")
            print(f"   -> Max CNN Raw Score     : {clip_max_cnn:.3f}")
            print(f"   -> Max Fused Avg Conf    : {clip_max_avg_conf:.3f} (Needs >{THRESHOLD})\n")

cv2.destroyAllWindows()

# ==========================================
# FINAL IEEE TABLE OUTPUT
# ==========================================
if valid_frame_count > 0:
    avg_neural = total_neural_ms / valid_frame_count
    avg_sbf = total_sbf_ms / valid_frame_count
    avg_cnn = total_cnn_ms / valid_frame_count
    total_pipeline_ms = avg_neural + avg_sbf + avg_cnn
    estimated_fps = 1000.0 / total_pipeline_ms

    print("\n==================================================")
    print("📊 FINAL HARDWARE INFERENCE LATENCY (Averaged)")
    print("==================================================")
    print(f"Total Frames Benchmarked : {valid_frame_count}")
    print(f"MediaPipe Tracking (ms)  : {avg_neural:.2f} ms")
    print(f"Spatial SBF Logic (ms)   : {avg_sbf:.2f} ms")
    print(f"MobileNetV2 CNN (ms)     : {avg_cnn:.2f} ms")
    print(f"--------------------------------------------------")
    print(f"Total Pipeline Latency   : {total_pipeline_ms:.2f} ms")
    print(f"Estimated Real-Time FPS  : {estimated_fps:.1f} FPS")

    print("\n==================================================")
    print("🎯 DYNAMIC SEQUENCE ACCURACY (Clip-Level)")
    print("==================================================")
    print(f"True Positives (Hit)     : {true_positives}")
    print(f"False Positives (Miss)   : {false_positives}")
    print(f"True Negatives (Correct) : {true_negatives}")
    print(f"False Negatives (Miss)   : {false_negatives}")
else:
    print("No valid frames processed to calculate metrics.")

