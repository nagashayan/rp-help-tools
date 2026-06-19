# Plan: Build a New Native Android App for On-Phone Handshake Inference

## Summary
- Leave [`WizardOfOz2`](/Users/nagashayanaramamurthy/AndroidStudioProjects/WizardOfOz2) unchanged. It remains the current thin client that polls `08_woz_master.py` at `GET /status`.
- Treat the browser path in [`06_export_onnx.py`](/Users/nagashayanaramamurthy/GitHub/rp-help-tools/p2/explore_sequence_dataset/06_export_onnx.py), [`07_mac_server.py`](/Users/nagashayanaramamurthy/GitHub/rp-help-tools/p2/explore_sequence_dataset/07_mac_server.py), and [`index.html`](/Users/nagashayanaramamurthy/GitHub/rp-help-tools/p2/explore_sequence_dataset/index.html) as a discarded experiment. Do not reuse the web client architecture.
- Create a brand-new Android Studio project whose job is: fetch frames from the Mac, run hand tracking natively on Android, assemble the 30-frame landmark window, run the classifier locally, and trigger the handshake UX locally.
- Keep the Mac as camera-only infrastructure. The Mac should stop making inference decisions for this new app.

## Key Changes
- **New Android project**
  - Create a fresh project, separate from `WizardOfOz2`, with a new app id and package name.
  - Use a standard Views-based app for v1, since the current app is already simple and the main complexity is the ML pipeline rather than UI architecture.
  - First version UX should include two modes in the same app:
    - `Debug` screen: live preview, connection status, tracker status, buffer fill (`0..30`), confidence, and current state.
    - `Pocket` mode: blacked-out low-distraction screen with haptics and minimal text, similar to the current app’s usage pattern.

- **Mac server split**
  - Keep [`07_mac_server.py`](/Users/nagashayanaramamurthy/GitHub/rp-help-tools/p2/explore_sequence_dataset/07_mac_server.py) as the base for the new app’s server path, not [`08_woz_master.py`](/Users/nagashayanaramamurthy/GitHub/rp-help-tools/p2/explore_sequence_dataset/08_woz_master.py).
  - For the new app’s flow, the Mac server should expose only:
    - `GET /snapshot` -> latest JPEG frame
    - `GET /health` -> simple readiness response
  - `08_woz_master.py` remains only for the old app and Wizard-of-Oz testing.

- **Native Android perception pipeline**
  - Poll `/snapshot` over LAN on a background coroutine at a fixed cadence, targeting about `15 FPS` initially.
  - Decode JPEG frames off the main thread and hand them to native MediaPipe Hands on Android.
  - Use one-hand tracking only.
  - Convert the detected hand into exactly `63` floats in the same order used by training: `21 * (x, y, z)`.
  - Maintain a rolling `30`-frame buffer.
  - When the buffer is full, transpose to the classifier input shape `float32[1, 63, 30]`.
  - Run local inference and map the result to:
    - probability for class `1`
    - `HANDSHAKE` if probability `> 0.80`
    - otherwise `IDLE`
  - Hold the handshake state for about `500 ms` and add a short vibration cooldown so one gesture does not retrigger continuously.

- **Model runtime**
  - Target TensorFlow Lite for the Android classifier runtime.
  - Recreate the current Temporal CNN from [`04_train_temporal_cnn.py`](/Users/nagashayanaramamurthy/GitHub/rp-help-tools/p2/explore_sequence_dataset/04_train_temporal_cnn.py) in TensorFlow/Keras, export to `.tflite`, and ship the model as an app asset.
  - Use TFLite CPU/XNNPACK first. Do not treat GPU/NNAPI as part of v1.

- **Tracking-loss behavior**
  - For short gaps in hand detection, reuse the last valid landmarks for a brief window so the sequence does not collapse on every missed frame.
  - After a longer gap, clear the buffer and return to accumulation mode.
  - Keep this simple and deterministic; do not attempt training-time interpolation logic on-device beyond short-gap carry-forward.

- **App-facing structure**
  - `MacFrameSource`: handles IP entry, health check, frame polling, and JPEG decode.
  - `HandTracker`: wraps native MediaPipe Hands and returns landmarks or no-hand.
  - `SequenceBuffer`: stores and manages the last 30 frames.
  - `HandshakeClassifier`: loads the TFLite model and returns logits/probabilities.
  - `HandshakeStateMachine`: applies threshold, hold time, cooldown, and haptic triggers.
  - `DebugViewModel` or equivalent controller: exposes preview/debug state to the UI.

## Public Interfaces / Contracts
- **Mac server contract for the new app**
  - `GET /health` -> `200 OK` when camera is ready
  - `GET /snapshot` -> JPEG image for the latest frame
- **Classifier contract**
  - Input tensor: `float32[1, 63, 30]`
  - Feature ordering: landmark-major within each frame as produced by current training
  - Output: 2 logits or 2 probabilities, with class `1` meaning handshake
- **Android app config**
  - User enters Mac IP manually in v1, same as the current app.
  - Cleartext LAN HTTP remains allowed for local development/testing.

## Test Plan
- **Model parity**
  - Build a small offline parity script that feeds saved 30-frame landmark windows into:
    - the current PyTorch model
    - the new TFLite model
  - Accept only if predicted class matches on representative samples and score drift does not flip decisions around the `0.80` threshold.

- **Pipeline validation**
  - Confirm the new app can:
    - connect to the Mac
    - receive frames continuously
    - detect a hand locally
    - fill the 30-frame buffer
    - produce local inference without calling `/status` or any server-side inference endpoint

- **UX validation**
  - In Debug mode, verify preview renders, buffer increments, confidence updates, and state changes are understandable.
  - In Pocket mode, verify screen dimming/blackout, wake behavior, and vibration reliability.

- **Failure cases**
  - Mac server unavailable at connect time
  - Wi-Fi dropout mid-session
  - hand exits frame temporarily
  - repeated missed detections
  - startup before buffer is full
  - phone rotation / app backgrounding

## Assumptions
- We are creating a new project, not cloning or modifying [`WizardOfOz2`](/Users/nagashayanaramamurthy/AndroidStudioProjects/WizardOfOz2).
- The failed Chrome experiment was specifically the combination of browser-side MediaPipe JS plus web inference flow, so the new app should use native Android MediaPipe rather than revisit the browser client.
- `08_woz_master.py` stays intact for the current study flow, while the new app uses a camera-only Mac server path derived from `07_mac_server.py`.
- For v1, manual Mac IP entry is good enough; auto-discovery can wait until after the local ML path is stable.
