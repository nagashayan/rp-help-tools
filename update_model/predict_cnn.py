import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- STEP 1: Initialize MediaPipe Tasks ---
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
THRESHOLD = 0.65  # Lowered slightly because Fusion is more robust
prediction_queue = deque(maxlen=WINDOW_SIZE)

# Persistence Configuration
STABILITY_HISTORY = 15  
coord_history = deque(maxlen=STABILITY_HISTORY)
# Range of movement allowed (Standard Deviation)
MAX_STABILITY_STD = 0.020 

def get_kinematic_score(landmarks):
    tip_y = landmarks[8].y
    knuckle_y = landmarks[5].y
    is_straight = tip_y < knuckle_y
    
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    index_mcp = np.array([landmarks[5].x, landmarks[5].y])
    v_dist = np.linalg.norm(thumb_tip - index_mcp)
    
    return 1.0 if (is_straight and v_dist > 0.08) else 0.2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp)

    k_score = 0
    p_score = 0 # Persistence Score
    
    if detection_result.hand_landmarks:
        current_landmarks = detection_result.hand_landmarks[0]
        k_score = get_kinematic_score(current_landmarks)
        
        # --- Persistence Logic: Track Wrist Stability ---
        wrist = current_landmarks[0]
        coord_history.append((wrist.x, wrist.y))
        
        if len(coord_history) == STABILITY_HISTORY:
            std_x = np.std([c[0] for c in coord_history])
            std_y = np.std([c[1] for c in coord_history])
            avg_std = (std_x + std_y) / 2
            
            # Soft Scoring: Persistence reward proportional to stillness
            # If avg_std is 0.005 (very still), p_score is high. If 0.02 (moving), p_score is 0.
            p_score = np.clip(1.0 - (avg_std / MAX_STABILITY_STD), 0, 1)
    else:
        coord_history.clear()

    # CNN Prediction
    img_cnn = cv2.resize(frame_rgb, (224, 224))
    img_cnn = cv2.GaussianBlur(img_cnn, (5, 5), 0)
    img_cnn = preprocess_input(img_cnn.astype(np.float32))
    img_cnn = np.expand_dims(img_cnn, axis=0)
    cnn_raw = model.predict(img_cnn, verbose=0)[0][0]

# --- STEP 4: 3-WAY FUSION LOGIC ---
    # CNN (50%) + Kinematics (30%) + Persistence (20%)
    fused_pred = (0.5 * cnn_raw) + (0.3 * k_score) + (0.2 * p_score)
    prediction_queue.append(fused_pred)
    avg_conf = sum(prediction_queue) / len(prediction_queue)

    # --- DIAGNOSTIC DASHBOARD ---
    # Create a background for better readability for low-vision users
    cv2.rectangle(frame, (5, 5), (450, 220), (0, 0, 0), -1)
    
    # Define color-coded thresholds (Green if contributing well, Red if low)
    c_color = (0, 255, 0) if cnn_raw > 0.5 else (0, 0, 255)
    k_color = (0, 255, 0) if k_score > 0.5 else (0, 0, 255)
    p_color = (0, 255, 0) if p_score > 0.5 else (0, 0, 255)

    # Render individual scores
    cv2.putText(frame, f"CNN (Pixels): {cnn_raw:.2f}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_color, 2)
    cv2.putText(frame, f"Kinematic (Pose): {k_score:.2f}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, k_color, 2)
    cv2.putText(frame, f"Persistence (Still): {p_score:.2f}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, p_color, 2)
    cv2.putText(frame, f"TOTAL FUSED: {avg_conf:.2f}", (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Final Decision Output
    if avg_conf > THRESHOLD and p_score > 0.6:
        label, color = "VERIFIED: WAITING", (0, 255, 0)
    elif avg_conf > THRESHOLD:
        label, color = "Stabilizing...", (0, 255, 255)
    else:
        label, color = "Scanning...", (0, 0, 255)

    cv2.putText(frame, label, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.imshow("Handshake 3-Way Fusion", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
