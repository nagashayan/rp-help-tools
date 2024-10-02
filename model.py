# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import matplotlib.pyplot as plt
import numpy as np


DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480


def show_using_matplot_lib(image, result):
    # Convert the MediaPipe image to a NumPy array.
    image_array = np.array(image.numpy_view(), dtype=np.uint8)

    # Display the image using matplotlib (which uses RGB format).
    plt.imshow(image_array)
    # Add text information on the image.
    # (x, y) coordinates are in pixels. Set (x, y) relative to the size of the image.
    plt.text(50, 50, f'Gesture: {result["category"]}', fontsize=16, color='red', weight='bold')

    # Optionally add more text or details.
    plt.text(50, 100, f'Confidence: {result["confidence"]}', fontsize=14, color='blue')

    plt.axis('off')  # Hide axis.
    plt.show()


def resize_and_show(image, result):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow(f'Image Display:{result}', img)
    # Wait for a key press indefinitely or for a specific time in milliseconds.
    cv2.waitKey(0)  # 0 means wait indefinitely until a key is pressed.

    # Close all OpenCV windows.
    cv2.destroyAllWindows()


def construct_show_result(result):
    if len(result) == 0:
        return {
            'category': 'unknown',
            'confidence': 0
        }

    category_name = result[0].category_name
    confidence = result[0].score
    return {
        'category': category_name,
        'confidence': confidence
    }


def display_batch_of_images_with_gestures_and_hand_landmarks(images_with_results):
    # Preview the images.
    # images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
    # breakpoint()
    for _, details in images_with_results.items():
        image = details['image']
        result = details['result']
        show_result = construct_show_result(result)
        show_using_matplot_lib(image, show_result)
        # resize_and_show(image, category_name)


IMAGE_FILENAMES = ['images/basic_images/thumbs_down.jpg', 'images/basic_images/victory.jpg', 'images/basic_images/thumbs_up.jpg', 'images/basic_images/pointing_up.jpg']


# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
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
    images_with_result[count] = {'image': image}
    top_gesture = recognition_result.gestures[0][0]
    hand_landmarks = recognition_result.hand_landmarks
    results.append((top_gesture, hand_landmarks))
    images_with_result[count]['result'] = (top_gesture, hand_landmarks)

display_batch_of_images_with_gestures_and_hand_landmarks(images_with_result)