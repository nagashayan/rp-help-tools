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

# V2.1
In this trial, I want to train using RPS (Rock paper scissor) dataset but use only rock and none.
I also kept only 30 images limited dataset to see if we can reproduce earlier problem:
- if it is limited dataset is the problem
- or is it too complicated/too many features and requires much more deep models.

we will be using rps_data_sample_experiment dataset for this experiment.

../images/rps_data_sample_experiment
Using existing files at /var/folders/vm/s5xwrxd93_xbnhp70qvyw03w0000gq/T/model_maker/gesture_recognizer/palm_detection_full.tflite
Using existing files at /var/folders/vm/s5xwrxd93_xbnhp70qvyw03w0000gq/T/model_maker/gesture_recognizer/hand_landmark_full.tflite
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1729686170.668226  677483 gl_context.cc:357] GL version: 2.1 (2.1 INTEL-22.5.12), renderer: Intel(R) UHD Graphics 617
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1729686170.731214  677979 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1729686170.771229  677978 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1729686170.847824  677979 landmark_projection_calculator.cc:186] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.11/site-packages/google/protobuf/symbol_database.py:55: UserWarning: SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead. SymbolDatabase.GetPrototype() will be removed soon.
  warnings.warn('SymbolDatabase.GetPrototype() is deprecated. Please '
Using existing files at /var/folders/vm/s5xwrxd93_xbnhp70qvyw03w0000gq/T/model_maker/gesture_recognizer/gesture_embedder
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
56/56 [==============================] - 3s 24ms/step - loss: 0.1610 - categorical_accuracy: 0.7054 - val_loss: 0.1000 - val_categorical_accuracy: 0.8571 - lr: 0.0010
Epoch 2/10
56/56 [==============================] - 1s 15ms/step - loss: 0.1347 - categorical_accuracy: 0.7768 - val_loss: 0.1338 - val_categorical_accuracy: 0.6429 - lr: 9.9000e-04
Epoch 3/10
56/56 [==============================] - 1s 17ms/step - loss: 0.1272 - categorical_accuracy: 0.8304 - val_loss: 0.1268 - val_categorical_accuracy: 0.7143 - lr: 9.8010e-04
Epoch 4/10
56/56 [==============================] - 1s 20ms/step - loss: 0.0994 - categorical_accuracy: 0.8393 - val_loss: 0.1498 - val_categorical_accuracy: 0.7143 - lr: 9.7030e-04
Epoch 5/10
56/56 [==============================] - 1s 16ms/step - loss: 0.1110 - categorical_accuracy: 0.8482 - val_loss: 0.1308 - val_categorical_accuracy: 0.7143 - lr: 9.6060e-04
Epoch 6/10
56/56 [==============================] - 1s 17ms/step - loss: 0.1039 - categorical_accuracy: 0.8393 - val_loss: 0.1257 - val_categorical_accuracy: 0.7143 - lr: 9.5099e-04
Epoch 7/10
56/56 [==============================] - 1s 17ms/step - loss: 0.1061 - categorical_accuracy: 0.8214 - val_loss: 0.1324 - val_categorical_accuracy: 0.7143 - lr: 9.4148e-04
Epoch 8/10
56/56 [==============================] - 1s 18ms/step - loss: 0.0854 - categorical_accuracy: 0.8393 - val_loss: 0.1445 - val_categorical_accuracy: 0.7143 - lr: 9.3207e-04
Epoch 9/10
56/56 [==============================] - 1s 15ms/step - loss: 0.0905 - categorical_accuracy: 0.8571 - val_loss: 0.1286 - val_categorical_accuracy: 0.7143 - lr: 9.2274e-04
Epoch 10/10
56/56 [==============================] - 1s 15ms/step - loss: 0.0893 - categorical_accuracy: 0.8571 - val_loss: 0.1156 - val_categorical_accuracy: 0.7143 - lr: 9.1352e-04
15/15 [==============================] - 1s 3ms/step - loss: 0.0472 - categorical_accuracy: 1.0000
Test loss:0.0472198948264122, Test accuracy:1.0

ChatGPT remarks:
The output you provided contains information about a training process for a Mediapipe gesture recognizer. Here's an interpretation of the key components:

Model Information:

The model includes an input layer for hand embeddings with 128 features, followed by a batch normalization layer, a ReLU activation layer, a dropout layer, and a final dense layer (classifier) with 2 output units. This indicates a simple feed-forward neural network used for classification.
The total number of parameters in the model is 770, with 514 being trainable.
Training and Validation Performance:

You trained the model for 10 epochs. The categorical_accuracy and val_categorical_accuracy refer to the accuracy on the training and validation sets, respectively.
Starting from Epoch 1, the training accuracy is relatively low at 70.5%, but it improves steadily to reach 85.7% by Epoch 10. The loss also decreases over time, suggesting the model is learning effectively.
Validation accuracy fluctuates around 71.4% after Epoch 3, indicating some potential overfitting, as the model's performance on the training set improves more than on the validation set.
Key Metrics:

Training accuracy: The model achieves 85.7% categorical accuracy by the end of training, meaning it's correctly classifying around 86% of the gestures in the training data.
Validation accuracy: It stabilizes around 71.4%, which might suggest that the model struggles to generalize to new, unseen data or that the validation set is particularly challenging.
Test accuracy: When evaluated on the test set, the model achieves 100% accuracy (categorical_accuracy: 1.0) with a very low test loss (0.0472). This suggests that the model performs extremely well on the test set.
Potential Issues:

Feedback manager warning: There is a warning about disabling support for feedback tensors due to a requirement for a single signature inference. This likely won't affect the final results but could be important depending on the specific application of the model.
Overfitting: The training accuracy significantly surpasses validation accuracy after a few epochs. The validation accuracy plateaus around 71%, while training accuracy keeps improving, suggesting overfitting.
Validation performance: The validation accuracy is not improving as much as expected, so the model might benefit from further tuning, such as regularization, early stopping, or using more data for validation.
Summary:
The model is learning effectively on the training data and performs exceptionally well on the test set.
However, the relatively low and stable validation accuracy compared to training accuracy suggests potential overfitting, and the model may struggle with generalization.

ChatGPT remarks on our hypothesis that its not low data but complex features in handhake compared to paper gesture is the
problem.

From the output of the second experiment, where you trained the model on the "paper" gesture with the same number of images, a few important observations can be made that help evaluate the initial hypothesis regarding gesture complexity:

### Key Results:
- **Test Accuracy**: The model achieves **100% accuracy** on the test data for the "paper" gesture, suggesting it is able to classify this gesture perfectly.
- **Validation Accuracy**: The model maintains a **validation accuracy around 71.43%**, which is somewhat lower than the test accuracy, but still decent.
- **Training Performance**: The model performs consistently well, reaching **85.71% categorical accuracy** by the 10th epoch, with a very low **test loss of 0.0472**.

### Comparison with "Handshake" Gesture Training:
- In the earlier experiment with the "handshake" gesture, the model achieved **50% test accuracy**, which is essentially random guessing for a binary classification task.
- On the "paper" gesture, the model achieves **100% test accuracy**, suggesting it can easily distinguish between the two classes for this gesture.
  
### Interpretation:
- **Gesture Complexity**: It seems that the "handshake" gesture is indeed more complex or harder to recognize than the "paper" gesture. The model can learn and classify simpler gestures like "paper" effectively with a shallow architecture.
- **Overfitting**: Overfitting is less likely the primary issue here since the model generalizes well for the "paper" gesture. The fact that the model struggles with the "handshake" gesture indicates that the issue is more likely related to the complexity of the gesture.

### Conclusion:
The experiment supports the hypothesis that the **"handshake" gesture likely requires a deeper or more complex neural network architecture** for better performance. You may need to:
1. **Increase model complexity** (e.g., by adding more layers, units, or using more advanced architectures like convolutional layers for better feature extraction).
2. **Improve gesture representation** by using more training data or augmenting the existing dataset to better capture the intricacies of the "handshake" gesture.

It seems that the current model can handle simpler gestures but may need to be scaled up to accommodate more nuanced hand shapes like "handshake."

# V2.2
Hypothesis:
- More none category than primary gesture might avoid overfitting? because v2.1 had the same pattern.

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
55/55 [==============================] - 3s 27ms/step - loss: 0.2069 - categorical_accuracy: 0.6545 - val_loss: 0.0801 - val_categorical_accuracy: 0.8571 - lr: 0.0010
Epoch 2/10
55/55 [==============================] - 1s 16ms/step - loss: 0.1866 - categorical_accuracy: 0.6909 - val_loss: 0.0871 - val_categorical_accuracy: 0.9286 - lr: 9.9000e-04
Epoch 3/10
55/55 [==============================] - 1s 20ms/step - loss: 0.1554 - categorical_accuracy: 0.7727 - val_loss: 0.1088 - val_categorical_accuracy: 0.7143 - lr: 9.8010e-04
Epoch 4/10
55/55 [==============================] - 1s 16ms/step - loss: 0.1464 - categorical_accuracy: 0.7727 - val_loss: 0.1107 - val_categorical_accuracy: 0.7143 - lr: 9.7030e-04
Epoch 5/10
55/55 [==============================] - 1s 16ms/step - loss: 0.1440 - categorical_accuracy: 0.7909 - val_loss: 0.1146 - val_categorical_accuracy: 0.7143 - lr: 9.6060e-04
Epoch 6/10
55/55 [==============================] - 1s 15ms/step - loss: 0.1354 - categorical_accuracy: 0.7818 - val_loss: 0.1163 - val_categorical_accuracy: 0.7143 - lr: 9.5099e-04
Epoch 7/10
55/55 [==============================] - 1s 17ms/step - loss: 0.1226 - categorical_accuracy: 0.8182 - val_loss: 0.1138 - val_categorical_accuracy: 0.7143 - lr: 9.4148e-04
Epoch 8/10
55/55 [==============================] - 1s 16ms/step - loss: 0.1412 - categorical_accuracy: 0.7636 - val_loss: 0.1015 - val_categorical_accuracy: 0.7857 - lr: 9.3207e-04
Epoch 9/10
55/55 [==============================] - 1s 18ms/step - loss: 0.1225 - categorical_accuracy: 0.7636 - val_loss: 0.1098 - val_categorical_accuracy: 0.7143 - lr: 9.2274e-04
Epoch 10/10
55/55 [==============================] - 1s 14ms/step - loss: 0.1127 - categorical_accuracy: 0.8000 - val_loss: 0.1038 - val_categorical_accuracy: 0.7143 - lr: 9.1352e-04
14/14 [==============================] - 1s 6ms/step - loss: 0.1402 - categorical_accuracy: 0.6429
Test loss:0.1402260959148407, Test accuracy:0.6428571343421936

ChatGPT remarks:
Final Test Performance:
After training, the model is evaluated on a test set, showing:
Test Loss: 0.1402
Test Accuracy: 64.29%
This accuracy is lower than the validation accuracy, indicating that the model has some issues generalizing to new data. It might still be overfitting or not fully capturing the underlying patterns of the task.
Summary:
The model starts with promising results but appears to hit a plateau, and performance slightly drops as training progresses, especially on validation and test sets. This is likely due to overfitting, which might be addressed with techniques such as more regularization, tuning hyperparameters, or collecting more data.

Compare v2 vs v2.2
The two model training outputs, version **v2** and version **v2.2**, exhibit several key differences and insights that highlight variations in performance, overfitting tendencies, and generalization capability.

### Key Similarities:
- **Model Architecture**: Both versions share the same architecture. They have an input layer (`hand_embedding`) of size 128, followed by a batch normalization layer, a ReLU activation, a dropout layer, and an output dense layer with 2 units (likely for binary classification).
  - Total Parameters: 770 (514 trainable, 256 non-trainable) for both.
  
- **Training Configuration**: Both models train for 10 epochs with similar learning rate schedules that decrease slightly across epochs.

### Key Differences:
#### 1. **Initial Model Performance (Epoch 1)**:
   - **v2.2**:
     - **Loss**: 0.2069
     - **Accuracy**: 65.45%
     - **Validation Accuracy**: 85.71%
   - **v2**:
     - **Loss**: 0.4987
     - **Accuracy**: 33.33%
     - **Validation Accuracy**: 75.00%

   **Insight**: Version **v2.2** starts with much better accuracy and a lower loss, indicating it learned faster in the initial epoch. In contrast, version **v2** struggles, beginning with poor training accuracy and a high loss, but the validation accuracy is high, suggesting the model might have learned some useful patterns.

#### 2. **Accuracy Trends**:
   - **v2.2**:
     - **Final Training Accuracy**: 80.00%
     - **Final Validation Accuracy**: 71.43% (dropped significantly after the first epoch)
   - **v2**:
     - **Final Training Accuracy**: 93.33%
     - **Final Validation Accuracy**: 75.00% (remained constant from Epoch 1)

   **Insight**: 
   - In **v2.2**, the training accuracy increases but the validation accuracy decreases from an initial high, which suggests **overfitting** (the model memorizes training data but struggles with generalization). 
   - In **v2**, the validation accuracy stays stable at 75%, but the training accuracy increases significantly (93.33%). This suggests that **v2** might still have some overfitting but generalizes slightly better based on validation performance.

#### 3. **Loss Trends**:
   - **v2.2**:
     - **Final Training Loss**: 0.1127
     - **Final Validation Loss**: 0.1038
   - **v2**:
     - **Final Training Loss**: 0.0648
     - **Final Validation Loss**: 0.4804

   **Insight**: The lower final training loss in **v2** suggests the model fit the training data better, but the validation loss remains significantly higher, indicating **v2** struggles more with generalization, as seen by the relatively high validation loss compared to **v2.2**. However, **v2.2** shows a more balanced training and validation loss, despite a plateau in validation accuracy.

#### 4. **Test Performance**:
   - **v2.2**:
     - **Test Loss**: 0.1402
     - **Test Accuracy**: 64.29%
   - **v2**:
     - **Test Loss**: 1.2402
     - **Test Accuracy**: 50.00%

   **Insight**: Version **v2.2** performs much better on the test set, achieving a test accuracy of 64.29% and a much lower loss (0.1402) compared to **v2**, which has a high test loss (1.2402) and only 50% accuracy. This indicates that **v2.2** generalizes better to unseen data, while **v2** suffers from poor generalization and overfitting, as seen in the test performance.

### Conclusion:
- **v2.2** has better overall generalization and maintains a reasonable balance between training and validation performance, though it does exhibit some overfitting after the first epoch.
- **v2**, despite achieving higher training accuracy, struggles with generalization, as indicated by its high validation and test losses, along with its poor test accuracy. This model likely overfits to the training data and fails to perform well on unseen data.

In summary, **v2.2** is the superior model for generalization, while **v2** shows signs of strong overfitting, performing poorly on new data despite good training accuracy.

How to test these models in realtime like the one google shows? lets try that.
