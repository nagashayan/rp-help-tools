"""
Improved Temporal CNN training for the raw landmark dataset.

What this version changes relative to 04_train_temporal_cnn.py:
1. Splits videos with label stratification at the clip level.
2. Uses separate train/validation/test sets.
3. Normalizes each feature using training-set statistics only.
4. Supports deployment-aligned causal interpolation.
5. Tracks balanced metrics and keeps the best validation checkpoint.

Run with:
    /Users/nagashayanaramamurthy/GitHub/rp-help-tools/p2/.venv_mp/bin/python 05_train_temporal_cnn_v2.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

WINDOW_SIZE = 30
NUM_FEATURES = 63
NUM_CLASSES = 2
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 35
DEFAULT_PATIENCE = 8
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
DATASET_PATH = "p1_dataset_combined_raw.csv"
CHECKPOINT_PATH = "temporal_cnn_raw_v2_checkpoint.pt"
STATE_DICT_PATH = "temporal_cnn_raw_v2.pth"
METRICS_PATH = "temporal_cnn_raw_v2_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the improved Temporal CNN on raw hand landmarks.")
    parser.add_argument("--dataset", default=DATASET_PATH, help="Path to the raw landmark CSV.")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Number of frames per training window.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience on validation macro F1.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Adam weight decay.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of videos held out for test.")
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of all videos reserved for validation.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("causal", "bidirectional", "none"),
        default="causal",
        help="Missing-landmark fill mode. 'causal' is deployment-aligned.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def landmark_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("landmark_")]


def interpolate_landmarks(df: pd.DataFrame, feature_cols: list[str], mode: str) -> pd.DataFrame:
    df = df.copy()
    if mode == "none":
        return df

    df[feature_cols] = df[feature_cols].replace(0.0, np.nan)
    if mode == "bidirectional":
        df[feature_cols] = df.groupby("video_name")[feature_cols].transform(
            lambda values: values.interpolate(method="linear", limit_direction="both")
        )
    elif mode == "causal":
        df[feature_cols] = df.groupby("video_name")[feature_cols].transform(lambda values: values.ffill())
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")

    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def split_videos(df: pd.DataFrame, test_size: float, val_size: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.")

    video_labels = df.groupby("video_name")["label"].first().reset_index()
    videos = video_labels["video_name"].to_numpy()
    labels = video_labels["label"].to_numpy()

    first_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(first_split.split(videos, labels))

    train_val_videos = videos[train_val_idx]
    train_val_labels = labels[train_val_idx]
    relative_val_size = val_size / (1.0 - test_size)

    second_split = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed + 1)
    train_idx, val_idx = next(second_split.split(train_val_videos, train_val_labels))

    return train_val_videos[train_idx], train_val_videos[val_idx], videos[test_idx]


def compute_normalization_stats(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = df[feature_cols].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def create_sliding_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences = []
    labels = []
    groups = []

    for video_name, group in df.groupby("video_name"):
        features = group[feature_cols].to_numpy(dtype=np.float32)
        features = (features - mean) / std
        label = int(group["label"].iloc[0])
        if len(features) < window_size:
            continue
        for start in range(len(features) - window_size + 1):
            sequences.append(features[start : start + window_size])
            labels.append(label)
            groups.append(video_name)

    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(groups),
    )


class HandshakeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class TemporalCNN(nn.Module):
    def __init__(self, num_features: int = NUM_FEATURES, num_classes: int = NUM_CLASSES, window_size: int = WINDOW_SIZE):
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
            conv_output = self.conv_block(dummy)
            flattened_length = conv_output.shape[1] * conv_output.shape[2]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_length, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block(x))


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(HandshakeDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def class_weight_tensor(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = len(labels) / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float | list[list[int]]]:
    model.eval()
    total_loss = 0.0
    logits_all = []
    labels_all = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    predictions = logits.argmax(dim=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.numpy(),
        predictions.numpy(),
        average="macro",
        zero_division=0,
    )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels.numpy(), predictions.numpy()),
        "balanced_accuracy": balanced_accuracy_score(labels.numpy(), predictions.numpy()),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion_matrix": confusion_matrix(labels.numpy(), predictions.numpy()).tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    logits_all = []
    labels_all = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    predictions = logits.argmax(dim=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.numpy(),
        predictions.numpy(),
        average="macro",
        zero_division=0,
    )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels.numpy(), predictions.numpy()),
        "balanced_accuracy": balanced_accuracy_score(labels.numpy(), predictions.numpy()),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def filter_videos(df: pd.DataFrame, videos: np.ndarray) -> pd.DataFrame:
    return df[df["video_name"].isin(videos)].copy()


def video_label_summary(df: pd.DataFrame) -> dict[int, int]:
    counts = df.groupby("video_name")["label"].first().value_counts().sort_index()
    return {int(key): int(value) for key, value in counts.items()}


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_raw_dataset(args.dataset)
    feature_cols = landmark_columns(df)
    df = interpolate_landmarks(df, feature_cols, args.interpolation)

    train_videos, val_videos, test_videos = split_videos(df, args.test_size, args.val_size, args.seed)
    train_df = filter_videos(df, train_videos)
    val_df = filter_videos(df, val_videos)
    test_df = filter_videos(df, test_videos)

    print(f"Video splits -> train: {len(train_videos)}, val: {len(val_videos)}, test: {len(test_videos)}")
    print(f"Train video labels: {video_label_summary(train_df)}")
    print(f"Val video labels: {video_label_summary(val_df)}")
    print(f"Test video labels: {video_label_summary(test_df)}")

    mean, std = compute_normalization_stats(train_df, feature_cols)
    X_train, y_train, _ = create_sliding_windows(train_df, feature_cols, args.window_size, mean, std)
    X_val, y_val, _ = create_sliding_windows(val_df, feature_cols, args.window_size, mean, std)
    X_test, y_test, _ = create_sliding_windows(test_df, feature_cols, args.window_size, mean, std)

    print(f"Window splits -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, args.batch_size, shuffle=False)

    model = TemporalCNN(num_features=len(feature_cols), window_size=args.window_size).to(device)
    weights = class_weight_tensor(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print(f"Class weights: {weights.detach().cpu().numpy().round(3).tolist()}")
    print("Starting training...")

    best_epoch = -1
    best_val_f1 = -np.inf
    best_state_dict = None
    best_val_metrics = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.3f} bal {train_metrics['balanced_accuracy']:.3f} f1 {train_metrics['macro_f1']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.3f} bal {val_metrics['balanced_accuracy']:.3f} f1 {val_metrics['macro_f1']:.3f}"
        )

        if val_metrics["macro_f1"] > best_val_f1 + 1e-6:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())
            best_val_metrics = deepcopy(val_metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    if best_state_dict is None or best_val_metrics is None:
        raise RuntimeError("Training completed without producing a checkpoint.")

    model.load_state_dict(best_state_dict)
    test_metrics = evaluate(model, test_loader, criterion, device)

    checkpoint = {
        "model_state_dict": best_state_dict,
        "feature_cols": feature_cols,
        "normalization_mean": mean,
        "normalization_std": std,
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "train_videos": train_videos.tolist(),
        "val_videos": val_videos.tolist(),
        "test_videos": test_videos.tolist(),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    torch.save(best_state_dict, STATE_DICT_PATH)

    serializable_report = {
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "train_videos": train_videos.tolist(),
        "val_videos": val_videos.tolist(),
        "test_videos": test_videos.tolist(),
        "history": history,
        "normalization_preview": {
            "mean_first_5": mean[:5].round(6).tolist(),
            "std_first_5": std[:5].round(6).tolist(),
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(to_builtin(serializable_report), metrics_file, indent=2)

    print("\nBest checkpoint summary")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation metrics: {json.dumps(to_builtin(best_val_metrics))}")
    print(f"Test metrics: {json.dumps(to_builtin(test_metrics))}")
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")
    print(f"Saved state dict to {STATE_DICT_PATH}")
    print(f"Saved metrics report to {METRICS_PATH}")


if __name__ == "__main__":
    run_training(parse_args())
