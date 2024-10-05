# rp-help-tools

# Setup
python3.11 -m venv .venv
source .venv/bin/activate
pip install pip-tools
pip-compile --output-file=requirements.txt requirements.in
pip-sync requirements.txt


# ml model download
wget -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
python download_images.py
python model.py

# development
tox
tox -e lint