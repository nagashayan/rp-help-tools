import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import time
import psutil
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Profiling Helper ---
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

# --- 1. Manually Define Hand Connections ---
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),      
    (0, 5), (5, 6), (6, 7), (7, 8),      
    (5, 9), (9, 10), (10, 11), (11, 12), 
    (9, 13), (13, 14), (14, 15), (15, 16), 
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
])

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO 
)
detector = vision.HandLandmarker.create_from_options(options)

model = tf.keras.models.load_model("handshake_model.keras")

# Configuration (UNTOUCHED)
WINDOW_SIZE = 10 
THRESHOLD = 0.60 
prediction_queue = deque(maxlen=WINDOW_SIZE)

STABILITY_HISTORY = 10
coord_history = deque(maxlen=STABILITY_HISTORY)
MAX_STABILITY_STD = 0.060

def get_pointing_vector(landmarks):
    """
    Calculates if the hand is pointing towards the camera (Z-axis).
    """
    wrist = landmarks[0]
    finger_tip = landmarks[12] 
    reach_z = wrist.z - finger_tip.z 
    return reach_z

cap = cv2.VideoCapture(0)

# Print Header for Data Collection
print(f"{'Neural(ms)':<12} | {'SBF(ms)':<12} | {'CNN(ms)':<12} | {'RAM(MB)':<12}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    
    # --- PROBE 1: Neural Landmark Extraction ---
    t_start_neural = time.perf_counter()
    detection_result = detector.detect_for_video(mp_image, timestamp)
    t_end_neural = time.perf_counter()
    neural_ms = (t_end_neural - t_start_neural) * 1000

    # Initialize variables
    k_score = 0
    p_score = 0
    vector_label = "No Hand"
    reach_val = 0.0
    is_pointing_at_camera = False
    
    # --- PROBE 2: SBF Logic (Symbolic) ---
    t_start_sbf = time.perf_counter()
    
    if detection_result.hand_landmarks:
        current_landmarks = detection_result.hand_landmarks[0]
        
        # Draw Skeleton
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            p1_norm = current_landmarks[start_idx]
            p2_norm = current_landmarks[end_idx]
            p1 = (int(p1_norm.x * w), int(p1_norm.y * h))
            p2 = (int(p2_norm.x * w), int(p2_norm.y * h))
            cv2.line(frame, p1, p2, (255, 0, 255), 2)
            cv2.circle(frame, p1, 4, (0, 255, 0), -1)

        # --- Z-VECTOR ANALYSIS (Pointing Towards You) ---
        reach_val = get_pointing_vector(current_landmarks)
        
        # Threshold Logic:
        # reach_val < 0.05: Hand is flat (High Five, Salute, Stop Sign)
        # reach_val > 0.10: Hand is reaching out (Handshake, Pointing)
        is_pointing_at_camera = reach_val > 0.08 # Adjusted threshold for "Reach"
        
        if is_pointing_at_camera:
            vector_label = f"Reaching Forward (Z={reach_val:.2f})"
        else:
            vector_label = f"Flat/Vertical (Z={reach_val:.2f})"

        # --- KINEMATIC SCORE ---
        # 1. Base Openness Check (V-Angle)
        thumb_tip = np.array([current_landmarks[4].x, current_landmarks[4].y])
        index_mcp = np.array([current_landmarks[5].x, current_landmarks[5].y])
        v_dist = np.linalg.norm(thumb_tip - index_mcp)
        is_open = v_dist > 0.08
        
        # 2. HARD GATES
        if not is_pointing_at_camera:
            k_score = 0.0 # Reject if not pointing at person
        elif not is_open:
            k_score = 0.0 # Reject if thumb is tucked (Fist pointing)
        else:
            k_score = 1.0 # Perfect handshake candidate

        # Persistence Tracking
        wrist = current_landmarks[0]
        coord_history.append((wrist.x, wrist.y))
        if len(coord_history) == STABILITY_HISTORY:
            avg_std = (np.std([c[0] for c in coord_history]) + np.std([c[1] for c in coord_history])) / 2
            p_score = np.clip(1.0 - (avg_std / MAX_STABILITY_STD), 0, 1)
    else:
        coord_history.clear()
        
    t_end_sbf = time.perf_counter()
    sbf_ms = (t_end_sbf - t_start_sbf) * 1000

    # --- PROBE 3: CNN Prediction ---
    t_start_cnn = time.perf_counter()
    
    img_cnn = cv2.resize(frame_rgb, (224, 224))
    img_cnn = cv2.GaussianBlur(img_cnn, (5, 5), 0)
    img_cnn = preprocess_input(img_cnn.astype(np.float32))
    img_cnn = np.expand_dims(img_cnn, axis=0)
    cnn_raw = model.predict(img_cnn, verbose=0)[0][0]
    
    t_end_cnn = time.perf_counter()
    cnn_ms = (t_end_cnn - t_start_cnn) * 1000

    # 3-Way Fusion (UNTOUCHED)
    fused_pred = (0.5 * cnn_raw) + (0.3 * k_score) + (0.2 * p_score)
    prediction_queue.append(fused_pred)
    avg_conf = sum(prediction_queue) / len(prediction_queue)

    # Get RAM Usage
    ram_usage = get_memory_usage()

# --- Dashboard Overlay (Expanded for Bigger Font) ---
    # Increased height to 400 to fit the large text
    cv2.rectangle(frame, (5, 5), (600, 360), (0, 0, 0), -1) 
    
    # Increased spacing to 40 pixels between lines
    cv2.putText(frame, f"CNN: {cnn_raw:.2f}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Pose (Z-Reach): {k_score:.2f}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Still: {p_score:.2f}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    v_color = (0, 255, 0) if is_pointing_at_camera else (0, 0, 255)
    cv2.putText(frame, vector_label, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, v_color, 2)

    # --- NEW PERFORMANCE METRICS ON SCREEN ---
    cv2.line(frame, (15, 180), (580, 180), (100, 100, 100), 2) # Separator line
    
    cv2.putText(frame, f"Neural Latency: {neural_ms:.1f} ms", (15, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"SBF Logic: {sbf_ms:.2f} ms", (15, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"CNN Latency: {cnn_ms:.1f} ms", (15, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"Memory: {ram_usage:.1f} MB", (15, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    if avg_conf > THRESHOLD and p_score > 0.6:
        label, color = "VERIFIED: WAITING", (0, 255, 0)
    elif avg_conf > THRESHOLD:
        label, color = "Stabilizing...", (0, 255, 255)
    else:
        label, color = "Scanning...", (0, 0, 255)

    cv2.putText(frame, label, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Handshake Z-Vector Logic", frame)
    
    # Console Print for Excel Data
    # Only print every 10th frame to avoid flooding
    if int(timestamp) % 10 == 0:
        print(f"{neural_ms:<12.2f} | {sbf_ms:<12.4f} | {cnn_ms:<12.2f} | {ram_usage:<12.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
