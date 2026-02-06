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
    running_mode=vision.RunningMode.VIDEO # Optimized for webcam
)
detector = vision.HandLandmarker.create_from_options(options)

# Load CNN
model = tf.keras.models.load_model("handshake_model.keras")

# Configuration
WINDOW_SIZE = 10 
THRESHOLD = 0.7   
MIN_CONSISTENCY = 0.8 
prediction_queue = deque(maxlen=WINDOW_SIZE)

# Configuration for Persistence
STABILITY_HISTORY = 15  # Check the last 15 frames
# Track the x,y coordinates of the wrist (landmark 0)
coord_history = deque(maxlen=STABILITY_HISTORY)

cap = cv2.VideoCapture(0)

def get_kinematic_score(landmarks):
    # landmarks is a list of normalized landmark objects
    # Index Tip (8) vs Index Knuckle (5)
    tip_y = landmarks[8].y
    knuckle_y = landmarks[5].y
    
    # Check finger straightness: tip 'higher' (lower y) than knuckle
    is_straight = tip_y < knuckle_y
    
    # Simple V-shape check (Thumb tip 4 vs Index base 5)
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    index_mcp = np.array([landmarks[5].x, landmarks[5].y])
    v_dist = np.linalg.norm(thumb_tip - index_mcp)
    
    return 1.0 if (is_straight and v_dist > 0.08) else 0.2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- STEP 2: MediaPipe Detection ---
    # Convert to MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    # Use timestamp for VIDEO mode
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp)

    k_score = 0
    if detection_result.hand_landmarks:
        # Get the first hand's landmarks
        current_landmarks = detection_result.hand_landmarks[0]
        k_score = get_kinematic_score(current_landmarks)
        # (Optional) You can add drawing logic here if needed for debugging

    # --- STEP 3: CNN Prediction ---
    img_cnn = cv2.resize(frame_rgb, (224, 224))
    img_cnn = cv2.GaussianBlur(img_cnn, (5, 5), 0)
    img_cnn = preprocess_input(img_cnn.astype(np.float32))
    img_cnn = np.expand_dims(img_cnn, axis=0)
    cnn_raw = model.predict(img_cnn, verbose=0)[0][0]

    # --- STEP 4: Fusion & Display ---
    fused_pred = (0.6 * cnn_raw) + (0.4 * k_score)
    prediction_queue.append(fused_pred)
    
    avg_conf = sum(prediction_queue) / len(prediction_queue)
    label = "VERIFIED" if avg_conf > THRESHOLD else "Scanning..."
    color = (0, 255, 0) if label == "VERIFIED" else (0, 0, 255)

    cv2.putText(frame, f"{label} ({avg_conf:.2f})", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    cv2.imshow("Handshake Fusion", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()