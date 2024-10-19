Not yet built model but in dataset gathering mode.

# Datasets:
- ChatGPT generate from V1
- https://ci3d.imar.ro/download (needs login of edu)
    - ^ this dataset contains handshake from diff lens perspectives but its mp4 format.
    - what todo?
    - we can break down to frames

But further reading this doc: https://storage.googleapis.com/mediapipe-assets/gesture_recognizer/model_card_hand_gesture_classification_with_faireness_2022.pdf

This model is not suitable for two hands (only for single hand) and no moving expressions (like hand wave etc)
because the embedded stage, considers one hand and then inputs it to secondary stage classification.

Conclusion:
So we gotta find a model which works with two hands and accepts movement.
one potential model is "Hand Landmark Detection"
    - low light training
    - when two hands join it fails, which should be okay?

- Openpose

Comparision:
| Feature                     | **OpenPose**                                 | **MediaPipe Hand Tracking**                        |
|-----------------------------|----------------------------------------------|---------------------------------------------------|
| **Purpose**                  | Full-body pose, hand, face tracking          | Specialized hand tracking and finger gestures     |
| **Keypoints Detected**       | 21 keypoints per hand (plus body/face)       | 21 keypoints per hand (fingers, joints, palm)     |
| **Architecture**             | Bottom-up, multi-stage CNN with PAFs         | Top-down, lightweight neural network              |
| **Performance**              | Real-time on high-end devices, slower on low-end | Real-time on most devices, optimized for mobile/web|
| **Ease of Use**              | More complex setup, requires GPU             | Easy to integrate, supports mobile and web        |
| **Use Case**                 | Full-body and multi-person tracking          | Hand-centric applications (e.g., sign language)   |
| **Multi-Person Support**     | Yes (full-body)                              | Yes (multi-hand, but no body tracking)            |
| **Platforms**                | Cross-platform, but heavy                    | Mobile, web, desktop with excellent performance   |

Lets stick to gesture recognition, I have been generating lot of images using LLMs and myown dataset to train.
Also found this in model card:
https://storage.googleapis.com/mediapipe-assets/gesture_recognizer/model_card_hand_gesture_classification_with_faireness_2022.pdf


    This model was trained and evaluated from the hand
    landmarks output data from MediaPipe Hands Model.
    This classication model does not directly use any
    images or videos as inputs.

so its better if we input images to hands model and then take output from there and train this model? I think so.

The way hands model works summarized from this link:
https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

1. The palm detection model is run to extract only the hand part (identification, scaling etc)
2. The hand landmark recognizer performs precise keypoint localization of 21 3D hand-knucle coordinates inside the detected hand regions via regression, that is direct coordinate prediction.

hands model is nothing but hand landmark model here:
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

The pipeline for V2 model should be:
- Use hand_landmark.task to detect 21 3D points of a hand from our dataset
- Take the output from above step and input it to our new model
- Train the new model
