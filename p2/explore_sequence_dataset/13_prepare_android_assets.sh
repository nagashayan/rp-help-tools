#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/nagashayanaramamurthy/GitHub/rp-help-tools/p2/explore_sequence_dataset"
MP_PY="$ROOT/../.venv_mp/bin/python"
TF_PY="$ROOT/../.venv_tf/bin/python"

cd "$ROOT"
"$MP_PY" 09_export_temporal_cnn_reference.py
"$TF_PY" 10_convert_temporal_cnn_to_tflite.py
"$TF_PY" 11_validate_tflite_parity.py
./12_download_hand_landmarker.sh
