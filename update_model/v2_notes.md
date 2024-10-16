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