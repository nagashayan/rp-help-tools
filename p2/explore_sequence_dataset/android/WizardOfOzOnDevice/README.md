# Wizard Of Oz On Device

Native Android client for phone-to-phone or Mac-to-phone egocentric inference.

## What it does

- Can run in inference mode or camera source mode from the same APK
- In inference mode, connects to a snapshot server at `http://<source-ip>:5000`
- Pulls `/snapshot` frames over LAN
- Runs MediaPipe Hand Landmarker locally on Android
- Builds a 30-frame landmark window
- Runs the Temporal CNN classifier locally with TFLite
- Exposes a debug screen plus pocket mode
- In camera source mode, uses the Android phone camera to serve `/health` and `/snapshot`

## Two-phone workflow

1. Install the same app on both phones.
2. On the mounted source phone, open `Camera Source Mode`.
3. Note the displayed local Wi-Fi IP address.
4. On the inference phone, enter that IP and press `Connect`.

## Required assets

Inference mode expects these files in `app/src/main/assets/`:

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
