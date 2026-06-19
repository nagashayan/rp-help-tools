#!/usr/bin/env bash
set -euo pipefail

ASSET_DIR="android/WizardOfOzOnDevice/app/src/main/assets"
ASSET_PATH="$ASSET_DIR/hand_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

mkdir -p "$ASSET_DIR"
curl -L "$MODEL_URL" -o "$ASSET_PATH"
echo "Saved $ASSET_PATH"
