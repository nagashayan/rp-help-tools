"""
Validate TFLite parity against PyTorch reference logits exported from Script 09.

Run with:
    /Users/nagashayanaramamurthy/GitHub/rp-help-tools/p2/.venv_tf/bin/python 11_validate_tflite_parity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

SAMPLES_PATH = "android/WizardOfOzOnDevice/model_assets/temporal_cnn_parity_samples.npz"
TFLITE_PATH = "android/WizardOfOzOnDevice/app/src/main/assets/temporal_cnn_raw.tflite"
THRESHOLD = 0.80


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def main() -> None:
    if not os.path.exists(SAMPLES_PATH):
        raise FileNotFoundError(f"Missing {SAMPLES_PATH}. Run 09_export_temporal_cnn_reference.py first.")
    if not os.path.exists(TFLITE_PATH):
        raise FileNotFoundError(f"Missing {TFLITE_PATH}. Run 10_convert_temporal_cnn_to_tflite.py first.")

    samples = np.load(SAMPLES_PATH)
    inputs = samples["inputs"].astype(np.float32)
    reference_logits = samples["logits"].astype(np.float32)
    reference_probs = samples["probabilities"].astype(np.float32)

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predicted_logits = []
    for sample in inputs:
        interpreter.set_tensor(input_details["index"], sample[np.newaxis, ...])
        interpreter.invoke()
        predicted_logits.append(interpreter.get_tensor(output_details["index"])[0])

    predicted_logits = np.array(predicted_logits, dtype=np.float32)
    predicted_probs = softmax(predicted_logits)

    argmax_mismatch_count = int(np.sum(np.argmax(predicted_logits, axis=1) != np.argmax(reference_logits, axis=1)))
    threshold_flip_count = int(
        np.sum((predicted_probs[:, 1] > THRESHOLD) != (reference_probs[:, 1] > THRESHOLD))
    )
    max_abs_logit_delta = float(np.max(np.abs(predicted_logits - reference_logits)))
    max_abs_prob_delta = float(np.max(np.abs(predicted_probs - reference_probs)))
    mean_abs_prob_delta = float(np.mean(np.abs(predicted_probs - reference_probs)))

    print(f"Samples checked: {len(inputs)}")
    print(f"Argmax mismatches: {argmax_mismatch_count}")
    print(f"Threshold flips at {THRESHOLD:.2f}: {threshold_flip_count}")
    print(f"Max abs logit delta: {max_abs_logit_delta:.6f}")
    print(f"Max abs prob delta: {max_abs_prob_delta:.6f}")
    print(f"Mean abs prob delta: {mean_abs_prob_delta:.6f}")

    if argmax_mismatch_count or threshold_flip_count or max_abs_prob_delta > 0.02:
        sys.exit(1)


if __name__ == "__main__":
    main()
