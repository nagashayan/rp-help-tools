# STEP 1: Import the necessary modules.
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480


def show_using_matplot_lib(image, result):
    # Convert the MediaPipe image to a NumPy array.
    image_array = np.array(image.numpy_view(), dtype=np.uint8)

    # Display the image using matplotlib (which uses RGB format).
    plt.imshow(image_array)
    # Add text information on the image.
    # (x, y) coordinates are in pixels. Set (x, y) relative to the size of the image.
    plt.text(
        50,
        50,
        f'Gesture: {result["category"]}',
        fontsize=16,
        color="red",
        weight="bold",
    )

    # Optionally add more text or details.
    plt.text(
        50,
        100,
        f'Confidence: {result["confidence"]}',
        fontsize=14,
        color="red",
        weight="bold",
    )

    plt.axis("off")  # Hide axis.
    plt.show()


def construct_show_result(result):
    if len(result) == 0:
        return {"category": "unknown", "confidence": 0}

    category_name = result[0].category_name
    confidence = result[0].score
    return {"category": category_name, "confidence": confidence}


def display_batch_of_images_with_gestures_and_hand_landmarks(images_with_results):
    # Preview the images.
    for _, details in images_with_results.items():
        image = details["image"]
        result = details["result"]
        show_result = construct_show_result(result)
        show_using_matplot_lib(image, show_result)


def get_image_extensions() -> tuple:
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
    return image_extensions


def get_image_folders() -> List:
    folder_paths = [Path("images/basic_images"), Path("images/shake")]
    return folder_paths


def get_images_filenames() -> List:
    folder_paths = get_image_folders()
    image_extensions = get_image_extensions()
    image_filenames = []
    for folder in folder_paths:
        filenames = [
            str(f) for f in folder.glob("*") if f.suffix.lower() in image_extensions
        ]
        image_filenames.extend(filenames)
    return image_filenames


IMAGE_FILENAMES = get_images_filenames()

# STEP 2: Create an GestureRecognizer object.
# base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
base_options = python.BaseOptions(model_asset_path="update_model/v1/gesture_recognizer.task")
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

images = []
results = []
images_with_result = {}
for count, image_file_name in enumerate(IMAGE_FILENAMES):
    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(image_file_name)

    # STEP 4: Recognize gestures in the input image.
    recognition_result = recognizer.recognize(image)

    # STEP 5: Process the result. In this case, visualize it.
    images.append(image)
    images_with_result[count] = {"image": image}
    if len(recognition_result.gestures) == 0:
        images_with_result[count]["result"] = []
        continue
    top_gesture = recognition_result.gestures[0][0]
    hand_landmarks = recognition_result.hand_landmarks
    results.append((top_gesture, hand_landmarks))
    images_with_result[count]["result"] = (top_gesture, hand_landmarks)

display_batch_of_images_with_gestures_and_hand_landmarks(images_with_result)
