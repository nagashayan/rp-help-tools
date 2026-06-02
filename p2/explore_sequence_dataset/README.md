This folder we are using our p1_dataset_combined to extract features and feed into GRU

python train_gru_baseline.py
X shape: (62, 30, 9)
y shape: (62,)
Class counts: [20 42]
Class weights: {0: 1.5333333333333334, 1: 0.7419354838709677}
Epoch 1/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 1s 66ms/step - accuracy: 0.3529 - loss: 0.7019 - val_accuracy: 0.2500 - val_loss: 0.7220
Epoch 2/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.3824 - loss: 0.7137 - val_accuracy: 0.2500 - val_loss: 0.7267
Epoch 3/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.3824 - loss: 0.7063 - val_accuracy: 0.2500 - val_loss: 0.7232
Epoch 4/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.3824 - loss: 0.7074 - val_accuracy: 0.2500 - val_loss: 0.7199
Epoch 5/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.4118 - loss: 0.7054 - val_accuracy: 0.2500 - val_loss: 0.7145
Epoch 6/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.4118 - loss: 0.6966 - val_accuracy: 0.2500 - val_loss: 0.7116
Epoch 7/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.4706 - loss: 0.6883 - val_accuracy: 0.2500 - val_loss: 0.7091
Epoch 8/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.4706 - loss: 0.6839 - val_accuracy: 0.2500 - val_loss: 0.7071
Epoch 9/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.4118 - loss: 0.6912 - val_accuracy: 0.1667 - val_loss: 0.7007
Epoch 10/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.5294 - loss: 0.6850 - val_accuracy: 0.2500 - val_loss: 0.6967
Epoch 11/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.5294 - loss: 0.6819 - val_accuracy: 0.2500 - val_loss: 0.6986
Epoch 12/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.4118 - loss: 0.6828 - val_accuracy: 0.2500 - val_loss: 0.7018
Epoch 13/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.5000 - loss: 0.6862 - val_accuracy: 0.2500 - val_loss: 0.7054
Epoch 14/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.5588 - loss: 0.6780 - val_accuracy: 0.2500 - val_loss: 0.7079
Epoch 15/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.4412 - loss: 0.6849 - val_accuracy: 0.2500 - val_loss: 0.7057
Epoch 16/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.4706 - loss: 0.6810 - val_accuracy: 0.3333 - val_loss: 0.7005
Epoch 17/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.5588 - loss: 0.6843 - val_accuracy: 0.4167 - val_loss: 0.6917
Epoch 18/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.5882 - loss: 0.6676 - val_accuracy: 0.4167 - val_loss: 0.6953
Epoch 19/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.5294 - loss: 0.6761 - val_accuracy: 0.4167 - val_loss: 0.6895
Epoch 20/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.4412 - loss: 0.6841 - val_accuracy: 0.4167 - val_loss: 0.6841
Epoch 21/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.5882 - loss: 0.6683 - val_accuracy: 0.4167 - val_loss: 0.6813
Epoch 22/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.6176 - loss: 0.6597 - val_accuracy: 0.5833 - val_loss: 0.6683
Epoch 23/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7647 - loss: 0.6553 - val_accuracy: 0.6667 - val_loss: 0.6435
Epoch 24/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7647 - loss: 0.6465 - val_accuracy: 0.6667 - val_loss: 0.6210
Epoch 25/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.8529 - loss: 0.6291 - val_accuracy: 0.6667 - val_loss: 0.6254
Epoch 26/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7647 - loss: 0.6303 - val_accuracy: 0.5833 - val_loss: 0.6305
Epoch 27/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7353 - loss: 0.6208 - val_accuracy: 0.5833 - val_loss: 0.6152
Epoch 28/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7353 - loss: 0.6131 - val_accuracy: 0.5833 - val_loss: 0.6147
Epoch 29/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7353 - loss: 0.5856 - val_accuracy: 0.5833 - val_loss: 0.5766
Epoch 30/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7941 - loss: 0.5503 - val_accuracy: 0.5833 - val_loss: 0.5249
Epoch 31/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.8235 - loss: 0.5002 - val_accuracy: 0.6667 - val_loss: 0.5860
Epoch 32/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7647 - loss: 0.4825 - val_accuracy: 0.6667 - val_loss: 0.5345
Epoch 33/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7941 - loss: 0.4776 - val_accuracy: 0.7500 - val_loss: 0.4978
Epoch 34/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7059 - loss: 0.4534 - val_accuracy: 0.6667 - val_loss: 0.4640
Epoch 35/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7941 - loss: 0.5161 - val_accuracy: 0.6667 - val_loss: 0.5204
Epoch 36/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.8235 - loss: 0.4239 - val_accuracy: 0.5000 - val_loss: 0.8277
Epoch 37/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.6471 - loss: 0.5283 - val_accuracy: 0.6667 - val_loss: 0.5712
Epoch 38/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7941 - loss: 0.4329 - val_accuracy: 0.6667 - val_loss: 0.4314
Epoch 39/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.7941 - loss: 0.3827 - val_accuracy: 0.7500 - val_loss: 0.3209
Epoch 40/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7647 - loss: 0.5953 - val_accuracy: 0.6667 - val_loss: 0.4241
Epoch 41/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7647 - loss: 0.4288 - val_accuracy: 0.6667 - val_loss: 0.5288
Epoch 42/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.7941 - loss: 0.4213 - val_accuracy: 0.5833 - val_loss: 0.6129
Epoch 43/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7353 - loss: 0.4397 - val_accuracy: 0.6667 - val_loss: 0.5616
Epoch 44/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7647 - loss: 0.4182 - val_accuracy: 0.6667 - val_loss: 0.5059
Epoch 45/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7941 - loss: 0.4175 - val_accuracy: 0.6667 - val_loss: 0.4755
Epoch 46/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7941 - loss: 0.4019 - val_accuracy: 0.7500 - val_loss: 0.4437
Epoch 47/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.8235 - loss: 0.3971 - val_accuracy: 0.7500 - val_loss: 0.4293
Epoch 48/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.7941 - loss: 0.4130 - val_accuracy: 0.7500 - val_loss: 0.4179
Epoch 49/100
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.8529 - loss: 0.3803 - val_accuracy: 0.7500 - val_loss: 0.4375
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 91ms/step

Confusion Matrix
[[1 4]
 [3 8]]

Classification Report
              precision    recall  f1-score   support

           0       0.25      0.20      0.22         5
           1       0.67      0.73      0.70        11

    accuracy                           0.56        16
   macro avg       0.46      0.46      0.46        16
weighted avg       0.54      0.56      0.55        16

Saved gru_handshake_sequence.keras

Accuracy was low so we will change features
