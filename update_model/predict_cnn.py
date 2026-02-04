import tensorflow as tf
import numpy as np
import cv2  # For image preprocessing if using webcam or local images
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the model
model = tf.keras.models.load_model("handshake_model.keras")

# 4. Real-time Handshake Detection
cap = cv2.VideoCapture(0)  # Start webcam feed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (224, 224))
    img = preprocess_input(img)  # Use correct normalization for MobileNetV2
    img = img.reshape(1, 224, 224, 3)

    # Prediction
    prediction = model.predict(img)[0][0]
    label = "Handshake Detected" if prediction > 0.6 else "No Handshake"

    cv2.putText(frame, f"{label} ({prediction:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
