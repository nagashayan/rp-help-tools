import tensorflow as tf
import numpy as np
import cv2  # For image preprocessing if using webcam or local images
from collections import deque 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the model
model = tf.keras.models.load_model("handshake_model.keras")

# Configuration
WINDOW_SIZE = 10  # Analyze the last 10 frames
THRESHOLD = 0.7   # Confidence threshold
MIN_CONSISTENCY = 0.8 # 80% of frames in window must be positive

# Buffer to store recent predictions
prediction_queue = deque(maxlen=WINDOW_SIZE)

# 4. Real-time Handshake Detection
cap = cv2.VideoCapture(0)  # Start webcam feed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. FIX COLOR: Convert BGR (OpenCV) to RGB (Model Expectation)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. Resize
    img = cv2.resize(frame_rgb, (224, 224))

    # 3. APPLY BLUR: Match the training data augmentation
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 4. Normalization for MobileNetV2 [-1, 1]
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    # --- DEBUG WINDOW (Should now show natural colors) ---
    debug_img = (img[0] + 1) / 2 * 255
    debug_img = debug_img.astype(np.uint8)
    debug_img_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("What the Model Sees", debug_img_bgr)
    # ----------------------------------------------------

    # Raw Prediction
    raw_pred = model.predict(img, verbose=0)[0][0]
    prediction_queue.append(raw_pred)

    # Temporal Smoothing Logic
    avg_conf = sum(prediction_queue) / len(prediction_queue)
    positive_frames = sum(1 for p in prediction_queue if p > THRESHOLD)
    consistency = positive_frames / len(prediction_queue)
    # This method suppresses transient false negatives and ensures that the assistive alert is only triggered by persistent initiatory gestures".
    if consistency >= MIN_CONSISTENCY and avg_conf > 0.5:
        label = "VERIFIED"
        color = (0, 255, 0)
    else:
        label = "Scanning..."
        color = (0, 0, 255)

    # Display Label (Font Scale 3 as requested for better visibility)
    cv2.putText(frame, f"{label} ({avg_conf:.2f})", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 3, color, 4, cv2.LINE_AA)
    cv2.imshow("Assistive Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
