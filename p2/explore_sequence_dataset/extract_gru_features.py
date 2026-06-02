# extract_gru_features.py

import os
import glob
from time import time
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATASET_DIR = "/Users/nagashayanaramamurthy/GitHub/rp-help-tools/images/p1_dataset_combined"
OUTPUT_X = "X_gru.npy"
OUTPUT_Y = "y_gru.npy"

SEQ_LEN = 30  # 1 second at 30fps; adjust later


def init_detector():
    base_options = python.BaseOptions(model_asset_path="../../p1/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def palm_tilt(landmarks):
    dx = landmarks[5].x - landmarks[17].x
    dy = landmarks[5].y - landmarks[17].y
    return abs(math.degrees(math.atan2(dy, dx))) / 180.0


def thumb_open(landmarks):
    return math.dist(
        [landmarks[4].x, landmarks[4].y],
        [landmarks[5].x, landmarks[5].y],
    )


def frame_features(landmarks):
    wrist = landmarks[0]
    middle_tip = landmarks[12]

    reach = wrist.z - middle_tip.z
    tilt = palm_tilt(landmarks)
    thumb = thumb_open(landmarks)

    # Basic compact feature vector
    return np.array(
        [
            wrist.x,
            wrist.y,
            wrist.z,
            middle_tip.x,
            middle_tip.y,
            middle_tip.z,
            reach,
            tilt,
            thumb,
        ],
        dtype=np.float32,
    )


def pad_or_trim(seq, seq_len):
    if len(seq) == 0:
        return np.zeros((seq_len, 9), dtype=np.float32)

    seq = np.asarray(seq, dtype=np.float32)

    if len(seq) >= seq_len:
        return seq[-seq_len:]

    pad = np.zeros((seq_len - len(seq), seq.shape[1]), dtype=np.float32)
    return np.vstack([pad, seq])


def extract_clip_features(detector, clip_dir):
    image_paths = sorted(
        glob.glob(os.path.join(clip_dir, "*.jpg"))
        + glob.glob(os.path.join(clip_dir, "*.png"))
        + glob.glob(os.path.join(clip_dir, "*.jpeg"))
    )

    seq = []

    for i, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            seq.append(frame_features(landmarks))
        else:
            seq.append(np.zeros(9, dtype=np.float32))

    return pad_or_trim(seq, SEQ_LEN)


def main():
    detector = init_detector()

    X = []
    y = []

    class_map = {
        "none": 0,
        "background": 0,
        "non_handshake": 0,
        "handshake": 1,
    }

    for class_name in sorted(os.listdir(DATASET_DIR)):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        label = class_map.get(class_name.lower())
        if label is None:
            print(f"Skipping unknown class folder: {class_name}")
            continue

        clip_dirs = sorted(
            d for d in glob.glob(os.path.join(class_dir, "*"))
            if os.path.isdir(d)
        )

        for clip_dir in clip_dirs:
            print(f"Processing {class_name}/{os.path.basename(clip_dir)}")
            features = extract_clip_features(detector, clip_dir)
            X.append(features)
            y.append(label)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Class counts:", np.bincount(y))

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)

    print(f"Saved {OUTPUT_X} and {OUTPUT_Y}")


if __name__ == "__main__":
    main()