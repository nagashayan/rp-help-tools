import requests

IMAGE_FILENAMES = ["thumbs_down.jpg", "victory.jpg", "thumbs_up.jpg", "pointing_up.jpg"]

for name in IMAGE_FILENAMES:
    url = f"https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{name}"
    try:
        print(f"Downloading {name} from {url}")
        response = requests.get(url)
        if response.status_code == 200:
            with open(name, "wb") as file:
                file.write(response.content)
            print(f"{name} downloaded successfully.")
        else:
            print(
                f"Failed to download {name}. HTTP Status Code: {response.status_code}"
            )
    except Exception as e:
        print(f"Failed to download {name}. Error: {e}")
