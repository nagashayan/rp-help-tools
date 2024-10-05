# Setup

python3.11 -m venv .venv
source .venv/bin/activate
pip install pip-tools
pip-compile --output-file=requirements.txt requirements.in
pip-sync requirements.txt

# Prep the dataset

Retrain the model using mediapipe model maker
Also prep the data according to COCO dataset format
https://ai.google.dev/edge/mediapipe/solutions/customization/object_detector#coco_format
