# v1 results:

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
6/6 [==============================] - 2s 169ms/step - loss: 0.3410 - categorical_accuracy: 0.4167 - val_loss: 0.6487 - val_categorical_accuracy: 0.0000e+00 - lr: 0.0010
Epoch 2/10
6/6 [==============================] - 0s 98ms/step - loss: 0.1853 - categorical_accuracy: 0.6667 - val_loss: 0.2834 - val_categorical_accuracy: 0.5000 - lr: 9.9000e-04
Epoch 3/10
6/6 [==============================] - 0s 98ms/step - loss: 0.1316 - categorical_accuracy: 0.7500 - val_loss: 0.1263 - val_categorical_accuracy: 0.5000 - lr: 9.8010e-04
Epoch 4/10
6/6 [==============================] - 1s 110ms/step - loss: 0.1653 - categorical_accuracy: 0.8333 - val_loss: 0.0759 - val_categorical_accuracy: 1.0000 - lr: 9.7030e-04
Epoch 5/10
6/6 [==============================] - 1s 105ms/step - loss: 0.1637 - categorical_accuracy: 0.6667 - val_loss: 0.0653 - val_categorical_accuracy: 1.0000 - lr: 9.6060e-04
Epoch 6/10
6/6 [==============================] - 1s 105ms/step - loss: 0.1289 - categorical_accuracy: 0.8333 - val_loss: 0.0645 - val_categorical_accuracy: 1.0000 - lr: 9.5099e-04
Epoch 7/10
6/6 [==============================] - 0s 86ms/step - loss: 0.1126 - categorical_accuracy: 0.8333 - val_loss: 0.0649 - val_categorical_accuracy: 1.0000 - lr: 9.4148e-04
Epoch 8/10
6/6 [==============================] - 0s 87ms/step - loss: 0.1497 - categorical_accuracy: 0.7500 - val_loss: 0.0650 - val_categorical_accuracy: 1.0000 - lr: 9.3207e-04
Epoch 9/10
6/6 [==============================] - 0s 91ms/step - loss: 0.0970 - categorical_accuracy: 0.9167 - val_loss: 0.0690 - val_categorical_accuracy: 1.0000 - lr: 9.2274e-04
Epoch 10/10
6/6 [==============================] - 0s 93ms/step - loss: 0.1315 - categorical_accuracy: 0.7500 - val_loss: 0.0700 - val_categorical_accuracy: 1.0000 - lr: 9.1352e-04
2/2 [==============================] - 0s 19ms/step - loss: 0.0990 - categorical_accuracy: 0.5000
Test loss:0.09897075593471527, Test accuracy:0.5

# Tips from chatgpt
- Overfitting problem
- Use old data with new data (combined, mixed dataset) to train
- Keep learning rate low, so that drastic changes wont be introduced
- Introduce data augmentation if the new data is limited to prevent overfitting.

Summary:
By applying these techniques, you can reduce the likelihood of catastrophic forgetting and ensure that your model retains its knowledge of previously learned gestures while adapting to new ones.
