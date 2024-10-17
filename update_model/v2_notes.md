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
