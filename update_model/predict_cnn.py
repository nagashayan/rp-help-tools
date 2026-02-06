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
THRESHOLD = 0.7   
MIN_CONSISTENCY = 0.8 
prediction_queue = deque(maxlen=WINDOW_SIZE)

# Persistence Configuration
STABILITY_HISTORY = 15  
coord_history = deque(maxlen=STABILITY_HISTORY)
STABILITY_THRESHOLD = 0.012  # Adjustable: lower is stricter

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
    is_still = False
    
    if detection_result.hand_landmarks:
        current_landmarks = detection_result.hand_landmarks[0]
        k_score = get_kinematic_score(current_landmarks)
        
        # --- Persistence Logic: Track Wrist Stability ---
        wrist = current_landmarks[0]
        coord_history.append((wrist.x, wrist.y))
        
        if len(coord_history) == STABILITY_HISTORY:
            std_x = np.std([c[0] for c in coord_history])
            std_y = np.std([c[1] for c in coord_history])
            # Hand is 'still' if coordinate variance is below threshold
            is_still = (std_x < STABILITY_THRESHOLD and std_y < STABILITY_THRESHOLD)
    else:
        coord_history.clear()

    # CNN Prediction
    img_cnn = cv2.resize(frame_rgb, (224, 224))
    img_cnn = cv2.GaussianBlur(img_cnn, (5, 5), 0)
    img_cnn = preprocess_input(img_cnn.astype(np.float32))
    img_cnn = np.expand_dims(img_cnn, axis=0)
    cnn_raw = model.predict(img_cnn, verbose=0)[0][0]

    # Decision Fusion with Persistence
    fused_pred = (0.5 * cnn_raw) + (0.5 * k_score)
    prediction_queue.append(fused_pred)
    
    avg_conf = sum(prediction_queue) / len(prediction_queue)
    
    # State Logic: Only Verify if Fused Score is high AND Hand is Stationary
    if avg_conf > THRESHOLD and is_still:
        label, color = "VERIFIED: WAITING", (0, 255, 0)
    elif avg_conf > THRESHOLD and not is_still:
        label, color = "Hand Detected: Keep Still", (0, 255, 255) # Yellow warning
    else:
        label, color = "Scanning...", (0, 0, 255)

    cv2.putText(frame, f"{label} ({avg_conf:.2f})", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.imshow("Handshake Fusion with Persistence", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
