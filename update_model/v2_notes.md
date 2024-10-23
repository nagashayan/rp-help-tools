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

One more caveat, probably the above pipeline will work but why RPS dataset training is working? 
Because RPS dataset doesn't have full human body?

just hands - is that what we are missing?
Train the new model with just hands, so crop all the images we have till now.
and Retrain for V2.

# Trial 1
Downloading https://storage.googleapis.com/mediapipe-assets/gesture_embedder.tar.gz to /var/folders/vm/s5xwrxd93_xbnhp70qvyw03w0000gq/T/model_maker/gesture_recognizer/gesture_embedder
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 hand_embedding (InputLayer  [(None, 128)]             0         
 )                                                               
                                                                 
 batch_normalization (Batch  (None, 128)               512       
 Normalization)                                                  
                                                                 
 re_lu (ReLU)                (None, 128)               0         
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 custom_gesture_recognizer_  (None, 2)                 258       
 out (Dense)                                                     
                                                                 
=================================================================
Total params: 770 (3.01 KB)
Trainable params: 514 (2.01 KB)
Non-trainable params: 256 (1.00 KB)
_________________________________________________________________
None
Epoch 1/10
15/15 [==============================] - 2s 70ms/step - loss: 0.4987 - categorical_accuracy: 0.3333 - val_loss: 0.4603 - val_categorical_accuracy: 0.7500 - lr: 0.0010
Epoch 2/10
 1/15 [=>............................] - ETA: 0s - loss: 0.3149 - categorical_accuracy: 0.000015/15 [==============================] - 0s 30ms/step - loss: 0.1277 - categorical_accuracy: 0.8000 - val_loss: 0.7719 - val_categorical_accuracy: 0.5000 - lr: 9.9000e-04
Epoch 3/10
15/15 [==============================] - 0s 33ms/step - loss: 0.1503 - categorical_accuracy: 0.7667 - val_loss: 0.7600 - val_categorical_accuracy: 0.5000 - lr: 9.8010e-04
Epoch 4/10
15/15 [==============================] - 1s 44ms/step - loss: 0.1337 - categorical_accuracy: 0.7667 - val_loss: 0.6384 - val_categorical_accuracy: 0.7500 - lr: 9.7030e-04
Epoch 5/10
15/15 [==============================] - 0s 35ms/step - loss: 0.1335 - categorical_accuracy: 0.7333 - val_loss: 0.5533 - val_categorical_accuracy: 0.7500 - lr: 9.6060e-04
Epoch 6/10
15/15 [==============================] - 0s 33ms/step - loss: 0.0895 - categorical_accuracy: 0.8000 - val_loss: 0.5457 - val_categorical_accuracy: 0.7500 - lr: 9.5099e-04
Epoch 7/10
15/15 [==============================] - 0s 33ms/step - loss: 0.0754 - categorical_accuracy: 0.9333 - val_loss: 0.5932 - val_categorical_accuracy: 0.7500 - lr: 9.4148e-04
Epoch 8/10
15/15 [==============================] - 1s 37ms/step - loss: 0.0973 - categorical_accuracy: 0.8333 - val_loss: 0.5856 - val_categorical_accuracy: 0.7500 - lr: 9.3207e-04
Epoch 9/10
15/15 [==============================] - 1s 35ms/step - loss: 0.0974 - categorical_accuracy: 0.8333 - val_loss: 0.5103 - val_categorical_accuracy: 0.7500 - lr: 9.2274e-04
Epoch 10/10
15/15 [==============================] - 0s 30ms/step - loss: 0.0648 - categorical_accuracy: 0.9333 - val_loss: 0.4804 - val_categorical_accuracy: 0.7500 - lr: 9.1352e-04
4/4 [==============================] - 0s 7ms/step - loss: 1.2402 - categorical_accuracy: 0.5000
Test loss:1.240241527557373, Test accuracy:0.5

ChatGPT remarks:
This output is from a model training process for gesture recognition using MediaPipe. Here’s a breakdown of the key components:

Model Architecture:
Input Layer: The input is a hand_embedding with 128 dimensions, suggesting it's likely derived from hand landmarks or features.
Batch Normalization Layer: Helps stabilize and speed up the training process by normalizing activations.
ReLU Activation Layer: Adds non-linearity to the model, improving its ability to capture complex patterns.
Dropout Layer: Applies regularization by randomly setting a portion of activations to zero, helping prevent overfitting.
Dense (Fully Connected) Layer: This is the output layer with 2 units, likely for binary classification (perhaps two gestures).
The model has:

770 total parameters: 514 trainable and 256 non-trainable.
Training Process:
The model was trained for 10 epochs, with categorical accuracy and loss being tracked.
Initially, the training started with a loss of 0.4987 and a low accuracy of 33.33%, but it improved rapidly.
After 10 epochs, the training accuracy reached 93.33%, but validation accuracy plateaued at 75% after epoch 2. This suggests potential overfitting: the model is performing better on the training data than on the validation data.
Learning Rate:
The learning rate (lr) starts at 0.001 and decays slightly after each epoch. This strategy is used to gradually fine-tune the model towards a better solution.
Test Performance:
Test loss: 1.2402, Test accuracy: 50%. This suggests that the model struggles to generalize to unseen data, as it only achieves random guess-level accuracy on the test set.
Key Observations:
Overfitting: The model shows high training accuracy but lower validation accuracy, with significant test loss and poor test accuracy (50%). More regularization (e.g., higher dropout or additional data augmentation) may help.
Gesture Class Complexity: If the task involves complex hand gestures, more sophisticated features or a deeper model might be needed.
Test Performance: Indicates that the model needs further tuning or more data to improve generalization.
To improve, consider:

Adding more data or diversifying the training data.
Tweaking the model architecture by adding layers or adjusting regularization techniques.
Early stopping or more epochs with proper validation tracking.
