# Wizard Of Oz On Device

Native Android client for the Mac-camera pipeline.

## What it does

- Connects to the Mac snapshot server at `http://<mac-ip>:5000`
- Pulls `/snapshot` frames over LAN
- Runs MediaPipe Hand Landmarker locally on Android
- Builds a 30-frame landmark window
- Runs the Temporal CNN classifier locally with TFLite
- Exposes a debug screen plus pocket mode

## Required assets

The app expects these files in `app/src/main/assets/`:

- `hand_landmarker.task`
- `temporal_cnn_raw.tflite`

The repo-side conversion scripts generate the TFLite classifier. The MediaPipe hand model is downloaded from the official Google AI Edge model bucket.

## Refresh model assets after retraining

From the repo root:

```bash
./13_prepare_android_assets.sh
```

That script:

1. exports PyTorch weights and parity samples
2. rebuilds the TFLite classifier
3. validates parity against the PyTorch reference outputs
4. downloads the MediaPipe hand tracker asset
