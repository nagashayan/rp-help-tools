import tensorflow as tf
import numpy as np
import cv2
import math
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

# --- CRITICAL: Force Window Size for Mobile Simulation ---
window_name = "Handshake Z-Vector Logic"

# 1. Create the window with the 'GUI_NORMAL' flag (often fixes resize issues on Mac)
cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL) 

# 2. Force the resize immediately
cv2.resizeWindow(window_name, 450, 850) 

# 3. (Optional) Move it to top-left so it doesn't get hidden
cv2.moveWindow(window_name, 100, 100)

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

def get_palm_tilt(landmarks):
    """
    Calculates the vertical tilt of the palm.
    Returns the angle in degrees (0 to 180).
    A perfect handshake is around 90 degrees.

    This is to avoid palm facing up/down (0 or 180) which can cause false positives in certain poses. We want to ensure the hand is vertical enough to be a handshake, not a gesture of asking money or just pointing hand towards the user.
    """
    index_base = landmarks[5]
    pinky_base = landmarks[17]
    
    # Calculate difference in X and Y
    dx = index_base.x - pinky_base.x
    dy = index_base.y - pinky_base.y
    
    # Calculate the angle using arctangent
    angle = math.degrees(math.atan2(dy, dx))
    
    # We only care about the absolute vertical tilt, so we normalize to 0-180
    return abs(angle)

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
        # reach_val > 0.08: Hand is reaching out (Handshake, Pointing)
        is_pointing_at_camera = reach_val > 0.08 # Adjusted threshold for "Reach"

        if is_pointing_at_camera:
            vector_label = f"Reaching Forward (Z={reach_val:.2f})"
        else:
            vector_label = f"Flat/Vertical (Z={reach_val:.2f})"

        # 3. Tilt Check (Palm is vertical, not flat)
        palm_tilt = get_palm_tilt(landmarks)
        is_vertical = 60 < palm_tilt < 120  # True if hand is sideways/vertical

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
        elif not is_vertical:
            k_score = 0 # Penalize if palm is not vertical (could be a "Money" gesture or just pointing)
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
# ... (Keep your neural/logic code above) ...
    ram_usage = get_memory_usage()

    # --- BEAUTIFIER DISPLAY LOGIC ---
    
    # 1. Smart Crop (Fixes the "Squashed Face" look)
    # We resize height to 850, then crop the center 450 width
    target_h = 850
    target_w = 450
    scale = target_h / h
    new_w = int(w * scale)
    resized_frame = cv2.resize(frame, (new_w, target_h))
    
    center_x = new_w // 2
    start_x = max(0, center_x - (target_w // 2))
    display_frame = resized_frame[:, start_x:start_x+target_w]

    # Double check we hit exact size (handle edge cases)
    display_frame = cv2.resize(display_frame, (target_w, target_h))

    # 2. "Glass" Overlay (Semi-Transparent Background)
    overlay = display_frame.copy()
    # Draw a dark box from top to y=320
    cv2.rectangle(overlay, (0, 0), (450, 320), (20, 20, 20), -1) 
    
    # Apply the transparency (0.7 = 70% visible video, 30% dark tint)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

    # 3. High-Quality Text (Anti-Aliased)
    # Using FONT_HERSHEY_DUPLEX for a cleaner look
    # lineType=cv2.LINE_AA is the secret to smooth text
    text_color = (240, 240, 240)
    label_color = (180, 180, 180) # Grey for labels
    
    # Row 1
    cv2.putText(display_frame, "CNN Confidence:", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, label_color, 1, cv2.LINE_AA)
    cv2.putText(display_frame, f"{cnn_raw:.2f}", (240, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1, cv2.LINE_AA)

    # Row 2
    cv2.putText(display_frame, "Pose Score:", (20, 75), cv2.FONT_HERSHEY_DUPLEX, 0.6, label_color, 1, cv2.LINE_AA)
    cv2.putText(display_frame, f"{k_score:.2f}", (240, 75), cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1, cv2.LINE_AA)

    # Row 3
    cv2.putText(display_frame, "Stability:", (20, 110), cv2.FONT_HERSHEY_DUPLEX, 0.6, label_color, 1, cv2.LINE_AA)
    cv2.putText(display_frame, f"{p_score:.2f}", (240, 110), cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1, cv2.LINE_AA)

    # Row 4 (The Status)
    v_color = (50, 255, 50) if is_pointing_at_camera else (50, 50, 255)
    cv2.putText(display_frame, vector_label, (20, 150), cv2.FONT_HERSHEY_DUPLEX, 0.7, v_color, 1, cv2.LINE_AA)

    # Separator
    cv2.line(display_frame, (20, 170), (430, 170), (100, 100, 100), 1)

    # Performance Stats (Smaller font)
    cv2.putText(display_frame, f"Neural Latency: {neural_ms:.1f} ms", (20, 200), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display_frame, f"SBF Logic:      {sbf_ms:.2f} ms", (20, 225), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display_frame, f"CNN Latency:    {cnn_ms:.1f} ms", (20, 250), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display_frame, f"Memory Usage:   {ram_usage:.1f} MB", (20, 275), cv2.FONT_HERSHEY_DUPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # Final Status Label at Bottom
    if avg_conf > THRESHOLD and p_score > 0.6:
        label, color = "VERIFIED", (50, 255, 50)
    else:
        label, color = "SCANNING...", (50, 50, 255)

    # Add a bottom bar for the status
    cv2.rectangle(display_frame, (0, 780), (450, 850), (0,0,0), -1)
    cv2.putText(display_frame, label, (110, 825), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

    cv2.imshow(window_name, display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
