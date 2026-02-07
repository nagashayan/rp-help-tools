import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

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

# Configuration
WINDOW_SIZE = 10 
THRESHOLD = 0.60 
prediction_queue = deque(maxlen=WINDOW_SIZE)

STABILITY_HISTORY = 15  
coord_history = deque(maxlen=STABILITY_HISTORY)
MAX_STABILITY_STD = 0.020 

def get_kinematic_score(landmarks):
    tip_y = landmarks[8].y
    knuckle_y = landmarks[5].y
    # Note: For horizontal handshake, 'straightness' might be x-based, 
    # but strictly speaking, finger curling is still best checked relatively.
    # We will stick to the V-distance check for openness.
    
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    index_mcp = np.array([landmarks[5].x, landmarks[5].y])
    v_dist = np.linalg.norm(thumb_tip - index_mcp)
    
    return 1.0 if (v_dist > 0.08) else 0.2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp)

    # Initialize variables
    k_score = 0
    p_score = 0
    orientation_label = "No Hand"
    is_handshake_orientation = False 
    
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

        # --- UPDATED ORIENTATION LOGIC: Horizontal Dominance ---
        # Pixel coordinates
        p5_x, p5_y = current_landmarks[5].x * w, current_landmarks[5].y * h
        p8_x, p8_y = current_landmarks[8].x * w, current_landmarks[8].y * h
        
        dx = abs(p8_x - p5_x)
        dy = abs(p8_y - p5_y)
        
        # LOGIC FLIP: Handshake is when X-diff is bigger than Y-diff
        # We use a slight multiplier (0.8) to allow for a slightly diagonal natural shake
        is_handshake_orientation = dx > (dy * 0.8)
        
        orientation_label = "Horizontal (Handshake)" if is_handshake_orientation else "Vertical (High-Five/Stop)"
        
        # --- HARD VETO: If Vertical, KILL the score ---
        base_k_score = get_kinematic_score(current_landmarks)
        if is_handshake_orientation:
            k_score = base_k_score
        else:
            k_score = 0.0 # Force rejection if vertical

        # Persistence Tracking
        wrist = current_landmarks[0]
        coord_history.append((wrist.x, wrist.y))
        if len(coord_history) == STABILITY_HISTORY:
            avg_std = (np.std([c[0] for c in coord_history]) + np.std([c[1] for c in coord_history])) / 2
            p_score = np.clip(1.0 - (avg_std / MAX_STABILITY_STD), 0, 1)
    else:
        coord_history.clear()

    # CNN Prediction
    img_cnn = cv2.resize(frame_rgb, (224, 224))
    img_cnn = cv2.GaussianBlur(img_cnn, (5, 5), 0)
    img_cnn = preprocess_input(img_cnn.astype(np.float32))
    img_cnn = np.expand_dims(img_cnn, axis=0)
    cnn_raw = model.predict(img_cnn, verbose=0)[0][0]

    # 3-Way Fusion
    fused_pred = (0.5 * cnn_raw) + (0.3 * k_score) + (0.2 * p_score)
    prediction_queue.append(fused_pred)
    avg_conf = sum(prediction_queue) / len(prediction_queue)

    # Dashboard Overlay
    cv2.rectangle(frame, (5, 5), (350, 160), (0, 0, 0), -1)
    cv2.putText(frame, f"CNN: {cnn_raw:.2f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Pose: {k_score:.2f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Still: {p_score:.2f}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Color code orientation label
    o_color = (0, 255, 0) if is_handshake_orientation else (0, 0, 255) # Red if Vertical (Rejected)
    cv2.putText(frame, orientation_label, (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, o_color, 1)

    if avg_conf > THRESHOLD and p_score > 0.6:
        label, color = "VERIFIED: WAITING", (0, 255, 0)
    elif avg_conf > THRESHOLD:
        label, color = "Stabilizing...", (0, 255, 255)
    else:
        label, color = "Scanning...", (0, 0, 255)

    cv2.putText(frame, label, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Handshake Horizontal Logic", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
