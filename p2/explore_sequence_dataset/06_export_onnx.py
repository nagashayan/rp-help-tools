"""
Export the normalized v2 Temporal CNN to ONNX for browser inference.

The exported ONNX graph includes the train-time z-score normalization, so the
browser can keep sending raw feature-major landmark windows.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

CHECKPOINT_PATH = "temporal_cnn_raw_v2_checkpoint.pt"
ONNX_PATH = "temporal_cnn_monolithic.onnx"


class TemporalCNN(nn.Module):
    def __init__(self, num_features: int = 63, num_classes: int = 2, window_size: int = 30):
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


def load_checkpoint(path: str) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, int, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run 05_train_temporal_cnn_v2.py first.")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a v2 checkpoint with normalization stats.")

    state_dict = checkpoint["model_state_dict"]
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    num_features = len(checkpoint["feature_cols"])
    window_size = int(checkpoint.get("config", {}).get("window_size", 30))
    return state_dict, mean, std, num_features, window_size


if __name__ == "__main__":
    state_dict, mean, std, num_features, window_size = load_checkpoint(CHECKPOINT_PATH)

    print("Loading PyTorch model...")
    base_model = TemporalCNN(num_features=num_features, window_size=window_size)
    base_model.load_state_dict(state_dict)
    base_model.eval()

    model = NormalizedTemporalCNN(base_model, mean, std)
    model.eval()

    dummy_input = torch.randn(1, num_features, window_size, requires_grad=True)

    print(f"Exporting normalized model to {ONNX_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    print("SUCCESS! ONNX export now includes normalization.")
