"""
Export Temporal CNN weights, normalization stats, and parity samples.

This script expects the v2 training checkpoint so exported artifacts preserve the
exact train-time z-score normalization used by 05_train_temporal_cnn_v2.py.

Run with:
    /Users/nagashayanaramamurthy/GitHub/rp-help-tools/p2/.venv_mp/bin/python 09_export_temporal_cnn_reference.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

NUM_CLASSES = 2
CHECKPOINT_PATH = "temporal_cnn_raw_v2_checkpoint.pt"
DATASET_PATH = "p1_dataset_combined_raw.csv"
OUTPUT_DIR = "android/WizardOfOzOnDevice/model_assets"
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "temporal_cnn_raw_weights.npz")
NORMALIZATION_PATH = os.path.join(OUTPUT_DIR, "temporal_cnn_raw_normalization.npz")
NORMALIZATION_JSON_PATH = os.path.join(OUTPUT_DIR, "temporal_cnn_raw_normalization.json")
SAMPLES_PATH = os.path.join(OUTPUT_DIR, "temporal_cnn_parity_samples.npz")


class TemporalCNN(nn.Module):
    def __init__(self, num_features: int, num_classes: int = NUM_CLASSES, window_size: int = 30):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, num_features, window_size)
            flattened_length = int(np.prod(self.conv_block(dummy).shape[1:]))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_length, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_block(x)
        return self.classifier(features)


class NormalizedTemporalCNN(nn.Module):
    def __init__(self, base_model: TemporalCNN, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.mean) / self.std
        return self.base_model(normalized)


def load_checkpoint(path: str) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, list[str], int, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run 05_train_temporal_cnn_v2.py first.")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a v2 checkpoint with normalization stats.")

    state_dict = checkpoint["model_state_dict"]
    feature_cols = checkpoint["feature_cols"]
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    config = checkpoint.get("config", {})
    window_size = int(config.get("window_size", 30))
    interpolation = str(config.get("interpolation", "causal"))
    return state_dict, mean, std, feature_cols, window_size, interpolation


def create_sliding_windows(df: pd.DataFrame, feature_cols: list[str], window_size: int) -> tuple[np.ndarray, np.ndarray]:
    sequences = []
    labels = []

    for _, group in df.groupby("video_name"):
        features = group[feature_cols].to_numpy(dtype=np.float32)
        label = int(group["label"].iloc[0])
        if len(features) >= window_size:
            for start in range(len(features) - window_size + 1):
                sequences.append(features[start : start + window_size])
                labels.append(label)

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_interpolated_dataset(path: str, feature_cols: list[str], interpolation: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if interpolation == "none":
        return df

    df[feature_cols] = df[feature_cols].replace(0.0, np.nan)
    if interpolation == "bidirectional":
        df[feature_cols] = df.groupby("video_name")[feature_cols].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
    elif interpolation == "causal":
        df[feature_cols] = df.groupby("video_name")[feature_cols].transform(lambda x: x.ffill())
    else:
        raise ValueError(f"Unsupported interpolation mode: {interpolation}")

    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def feature_major(window_batch: np.ndarray) -> np.ndarray:
    return np.transpose(window_batch, (0, 2, 1)).astype(np.float32)


def stratified_indices(labels: np.ndarray, per_class: int = 24) -> np.ndarray:
    selections = []
    for class_id in np.unique(labels):
        indices = np.where(labels == class_id)[0]
        if len(indices) <= per_class:
            selections.extend(indices.tolist())
            continue
        picks = np.linspace(0, len(indices) - 1, per_class, dtype=int)
        selections.extend(indices[picks].tolist())
    return np.array(sorted(selections), dtype=np.int64)


def export_weights(state_dict: dict[str, torch.Tensor]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(
        WEIGHTS_PATH,
        conv1_weight=state_dict["conv_block.0.weight"].cpu().numpy(),
        conv1_bias=state_dict["conv_block.0.bias"].cpu().numpy(),
        conv2_weight=state_dict["conv_block.3.weight"].cpu().numpy(),
        conv2_bias=state_dict["conv_block.3.bias"].cpu().numpy(),
        fc1_weight=state_dict["classifier.1.weight"].cpu().numpy(),
        fc1_bias=state_dict["classifier.1.bias"].cpu().numpy(),
        fc2_weight=state_dict["classifier.4.weight"].cpu().numpy(),
        fc2_bias=state_dict["classifier.4.bias"].cpu().numpy(),
    )


def export_normalization(mean: np.ndarray, std: np.ndarray, feature_cols: list[str]) -> None:
    np.savez(
        NORMALIZATION_PATH,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        feature_cols=np.array(feature_cols),
    )
    with open(NORMALIZATION_JSON_PATH, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "feature_cols": feature_cols,
                "mean": mean.astype(np.float32).round(8).tolist(),
                "std": std.astype(np.float32).round(8).tolist(),
            },
            output_file,
            indent=2,
        )


def export_reference_samples(
    model: NormalizedTemporalCNN,
    feature_cols: list[str],
    window_size: int,
    interpolation: str,
) -> None:
    df = load_interpolated_dataset(DATASET_PATH, feature_cols, interpolation)
    sequences, labels = create_sliding_windows(df, feature_cols, window_size)
    sample_indices = stratified_indices(labels)
    sampled_windows = sequences[sample_indices]
    sampled_labels = labels[sample_indices]
    model_inputs = feature_major(sampled_windows)

    with torch.no_grad():
        tensor_inputs = torch.tensor(model_inputs, dtype=torch.float32)
        logits = model(tensor_inputs).cpu().numpy().astype(np.float32)
        probabilities = torch.softmax(torch.tensor(logits), dim=1).cpu().numpy().astype(np.float32)

    np.savez(
        SAMPLES_PATH,
        inputs=model_inputs,
        labels=sampled_labels.astype(np.int64),
        logits=logits,
        probabilities=probabilities,
    )


def main() -> None:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Missing {DATASET_PATH}")

    state_dict, mean, std, feature_cols, window_size, interpolation = load_checkpoint(CHECKPOINT_PATH)
    base_model = TemporalCNN(num_features=len(feature_cols), window_size=window_size)
    base_model.load_state_dict(state_dict)
    base_model.eval()

    normalized_model = NormalizedTemporalCNN(base_model, mean, std)
    normalized_model.eval()

    export_weights(state_dict)
    export_normalization(mean, std, feature_cols)
    export_reference_samples(normalized_model, feature_cols, window_size, interpolation)

    print(f"Saved weight archive to {WEIGHTS_PATH}")
    print(f"Saved normalization archive to {NORMALIZATION_PATH}")
    print(f"Saved normalization JSON to {NORMALIZATION_JSON_PATH}")
    print(f"Saved parity samples to {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
