from google.colab import files
import os
import json
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import gesture_recognizer


# !wget https://storage.googleapis.com/mediapipe-tasks/object_detector/android_figurine.zip
# !unzip android_figurine.zip
train_dataset_path = "../images/handshake/train"
validation_dataset_path =  "../images/handshake/validation"
