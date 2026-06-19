"""
Build a TensorFlow/Keras replica of the Temporal CNN and export TFLite assets.

The exported model includes the train-time z-score normalization, so runtime
clients can continue sending raw feature-major landmark windows.

Run with:
    /Users/nagashayanaramamurthy/GitHub/rp-help-tools/p2/.venv_tf/bin/python 10_convert_temporal_cnn_to_tflite.py
"""

from __future__ import annotations

import os

import numpy as np
import tensorflow as tf

WEIGHTS_PATH = "android/WizardOfOzOnDevice/model_assets/temporal_cnn_raw_weights.npz"
NORMALIZATION_PATH = "android/WizardOfOzOnDevice/model_assets/temporal_cnn_raw_normalization.npz"
KERAS_PATH = "android/WizardOfOzOnDevice/model_assets/temporal_cnn_raw.keras"
TFLITE_PATH = "android/WizardOfOzOnDevice/app/src/main/assets/temporal_cnn_raw.tflite"


def load_normalization() -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(NORMALIZATION_PATH):
        raise FileNotFoundError(f"Missing {NORMALIZATION_PATH}. Run 09_export_temporal_cnn_reference.py first.")
    stats = np.load(NORMALIZATION_PATH)
    mean = stats["mean"].astype(np.float32).reshape(1, -1, 1)
    std = stats["std"].astype(np.float32).reshape(1, -1, 1)
    return mean, std


def build_model(mean: np.ndarray, std: np.ndarray) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(63, 30), name="input")
    x = tf.keras.layers.Lambda(lambda tensor: (tensor - mean) / std, name="normalize")(inputs)
    x = tf.keras.layers.Permute((2, 1), name="to_time_major")(x)
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu", name="conv1")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    x = tf.keras.layers.Permute((2, 1), name="to_channel_major_before_flatten")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(2, name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="temporal_cnn_raw")


def load_weights(model: tf.keras.Model) -> None:
    weights = np.load(WEIGHTS_PATH)
    model.get_layer("conv1").set_weights(
        [
            np.transpose(weights["conv1_weight"], (2, 1, 0)),
            weights["conv1_bias"],
        ]
    )
    model.get_layer("conv2").set_weights(
        [
            np.transpose(weights["conv2_weight"], (2, 1, 0)),
            weights["conv2_bias"],
        ]
    )
    model.get_layer("fc1").set_weights(
        [
            np.transpose(weights["fc1_weight"], (1, 0)),
            weights["fc1_bias"],
        ]
    )
    model.get_layer("output").set_weights(
        [
            np.transpose(weights["fc2_weight"], (1, 0)),
            weights["fc2_bias"],
        ]
    )


def convert_to_tflite(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    return converter.convert()


def main() -> None:
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing {WEIGHTS_PATH}. Run 09_export_temporal_cnn_reference.py first.")

    mean, std = load_normalization()
    os.makedirs(os.path.dirname(TFLITE_PATH), exist_ok=True)
    model = build_model(mean, std)
    model(np.zeros((1, 63, 30), dtype=np.float32))
    load_weights(model)
    model.save(KERAS_PATH)

    tflite_model = convert_to_tflite(model)
    with open(TFLITE_PATH, "wb") as output_file:
        output_file.write(tflite_model)

    print(f"Saved Keras checkpoint to {KERAS_PATH}")
    print(f"Saved TFLite model to {TFLITE_PATH}")


if __name__ == "__main__":
    main()
