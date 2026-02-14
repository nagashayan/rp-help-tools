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

Info about files:
cnn_model_trainer.py - Run this to train our model, make sure images are there in expected folders and also plots learning curve at end.
predict_cnn.py - Once the training is done, it will use the model to predict handshake
record_sequence_dataset.py - Use this to create sequence based dataset required for sequence based confusion matrix creation
sequence_based_confusion_matrix.py - Once we have sequence based dataset, run this file to create confusion matrix for CNN vs CNN+K-sccore+hand stability
hand_andmarker.task - Pretrained mobilev2net mmodel
handshake_model.keras - Our CNN based trained model
