import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---- Load Trained CNN Model ----
model = tf.keras.models.load_model("handshake_model.keras")

# ---- Initialize MediaPipe Hand Landmarker ----
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# ---- Helper: Depth / Reach Function ----
def get_pointing_vector(landmarks):
    wrist = landmarks[0]
    finger_tip = landmarks[12]
    return wrist.z - finger_tip.z

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true_labels, preds, title):
    cm = confusion_matrix(true_labels, preds)

    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Background', 'Handshake'],
                yticklabels=['Background', 'Handshake'])

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.savefig("cnn_confusion_matrix.png", dpi=300)


def evaluate_sequence_dataset(dataset_dir, threshold=0.60):
    """
    Sequence-based evaluation that mirrors the live pipeline:
      - Maintain a rolling WINDOW_SIZE average of:
          * CNN-only: cnn_raw
          * Hybrid: fused = 0.5*cnn_raw + 0.3*k_score + 0.2*p_score
      - Trigger handshake for a clip if at ANY point:
          * CNN-only: avg_cnn > threshold
          * Hybrid: avg_hybrid > threshold AND p_score > 0.6
    Dataset folder structure:
        dataset_dir/
            handshake/clip_xx/frame_###.jpg
            none/clip_xx/frame_###.jpg
    """
    from sklearn.metrics import confusion_matrix, classification_report
    import os
    import cv2
    import time
    from collections import deque

    WINDOW_SIZE = 10
    STABILITY_HISTORY_LOCAL = 10
    MAX_STABILITY_STD_LOCAL = 0.060

    true_labels: list[int] = []
    preds_cnn: list[int] = []
    preds_hybrid: list[int] = []

    for label_name in ["none", "handshake"]:
        class_dir = os.path.join(dataset_dir, label_name)
        if not os.path.exists(class_dir):
            continue

        for clip in os.listdir(class_dir):
            clip_dir = os.path.join(class_dir, clip)
            if not os.path.isdir(clip_dir):
                continue

            # Per-clip rolling buffers (matches live behavior, avoids cross-clip leakage)
            prediction_queue_cnn = deque(maxlen=WINDOW_SIZE)
            prediction_queue_hybrid = deque(maxlen=WINDOW_SIZE)
            coord_history_local = deque(maxlen=STABILITY_HISTORY_LOCAL)

            cnn_triggered = False
            hybrid_triggered = False

            # Process frames in temporal order
            for frame_file in sorted(os.listdir(clip_dir)):
                frame_path = os.path.join(clip_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ---------------- CNN ----------------
                img_cnn = cv2.resize(frame_rgb, (224, 224))
                img_cnn = cv2.GaussianBlur(img_cnn, (5, 5), 0)
                img_cnn = preprocess_input(img_cnn.astype(np.float32))
                img_cnn = np.expand_dims(img_cnn, axis=0)
                cnn_raw = model.predict(img_cnn, verbose=0)[0][0]

                # ---------------- SBF / Stability ----------------
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp = int(time.time() * 1000)
                detection_result = detector.detect_for_video(mp_image, timestamp)

                k_score = 0.0
                p_score = 0.0

                if detection_result.hand_landmarks:
                    current_landmarks = detection_result.hand_landmarks[0]

                    # Z reach
                    reach_val = get_pointing_vector(current_landmarks)
                    is_reaching = reach_val > 0.08

                    # Openness
                    thumb_tip = np.array([current_landmarks[4].x, current_landmarks[4].y])
                    index_mcp = np.array([current_landmarks[5].x, current_landmarks[5].y])
                    openness = np.linalg.norm(thumb_tip - index_mcp)
                    is_open = openness > 0.08

                    if is_reaching and is_open:
                        k_score = 1.0

                    # Stability (wrist jitter over last STABILITY_HISTORY frames)
                    wrist = current_landmarks[0]
                    coord_history_local.append((wrist.x, wrist.y))

                    if len(coord_history_local) == STABILITY_HISTORY_LOCAL:
                        std_x = np.std([c[0] for c in coord_history_local])
                        std_y = np.std([c[1] for c in coord_history_local])
                        avg_std = (std_x + std_y) / 2.0
                        p_score = np.clip(
                            1.0 - (avg_std / MAX_STABILITY_STD_LOCAL),
                            0.0,
                            1.0
                        )
                else:
                    coord_history_local.clear()

                # ---------------- Rolling-window decision (matches live) ----------------
                # CNN-only rolling average
                prediction_queue_cnn.append(float(cnn_raw))
                avg_cnn = sum(prediction_queue_cnn) / len(prediction_queue_cnn)

                # Hybrid rolling average
                fused = (0.5 * float(cnn_raw)) + (0.3 * float(k_score)) + (0.2 * float(p_score))
                prediction_queue_hybrid.append(float(fused))
                avg_hybrid = sum(prediction_queue_hybrid) / len(prediction_queue_hybrid)

                # Trigger if at ANY time during the clip we would show VERIFIED (or CNN crosses)
                if avg_cnn > threshold:
                    cnn_triggered = True

                if avg_hybrid > threshold and p_score > 0.6:
                    hybrid_triggered = True

                # Early exit once both decisions are made
                if cnn_triggered and hybrid_triggered:
                    break

            true_label = 1 if label_name == "handshake" else 0
            pred_cnn = 1 if cnn_triggered else 0
            pred_hybrid = 1 if hybrid_triggered else 0

            true_labels.append(true_label)
            preds_cnn.append(pred_cnn)
            preds_hybrid.append(pred_hybrid)

    if len(true_labels) == 0:
        print("No valid clips found. Check dataset_dir structure.")
        return

    print("\nCNN Sequence Confusion Matrix (rolling window)")
    print(confusion_matrix(true_labels, preds_cnn))

    print("\nHybrid Sequence Confusion Matrix (rolling window + stability gate)")
    print(confusion_matrix(true_labels, preds_hybrid))

    print("\nCNN Classification Report (sequence-level)")
    print(classification_report(true_labels, preds_cnn, zero_division=0))

    print("\nHybrid Classification Report (sequence-level)")
    print(classification_report(true_labels, preds_hybrid, zero_division=0))

    plot_confusion_matrix(true_labels, preds_cnn, 
                      "CNN Sequence-Level Confusion Matrix")

    plot_confusion_matrix(true_labels, preds_hybrid, 
                        "Hybrid Neuro-Symbolic Confusion Matrix")

evaluate_sequence_dataset("../images/sequence_dataset", threshold=0.60)