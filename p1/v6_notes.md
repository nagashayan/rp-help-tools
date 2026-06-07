- Added palm tilt as validation param under pose score (or K score)
- Converted images to greyscale and blocked faces before training CNN to avoid face and color bias

python cnn_model_trainer.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
Found 103 images belonging to 2 classes.
Found 25 images belonging to 2 classes.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
Epoch 1/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 4s 589ms/step - accuracy: 0.4949 - loss: 0.8976 - val_accuracy: 0.2800 - val_loss: 0.8918
Epoch 2/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 426ms/step - accuracy: 0.4963 - loss: 0.8657 - val_accuracy: 0.3200 - val_loss: 0.8891
Epoch 3/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 321ms/step - accuracy: 0.5724 - loss: 0.9200 - val_accuracy: 0.2000 - val_loss: 0.9101
Epoch 4/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 330ms/step - accuracy: 0.5577 - loss: 0.8248 - val_accuracy: 0.3200 - val_loss: 0.7751
Epoch 5/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 316ms/step - accuracy: 0.5960 - loss: 0.6884 - val_accuracy: 0.4800 - val_loss: 0.8300
Epoch 6/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 336ms/step - accuracy: 0.4912 - loss: 0.8900 - val_accuracy: 0.5200 - val_loss: 0.8544
Epoch 7/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 323ms/step - accuracy: 0.6589 - loss: 0.6013 - val_accuracy: 0.3600 - val_loss: 0.9240
Epoch 8/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 429ms/step - accuracy: 0.6055 - loss: 0.7775 - val_accuracy: 0.3600 - val_loss: 0.8680
Epoch 9/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 427ms/step - accuracy: 0.7173 - loss: 0.6089 - val_accuracy: 0.4800 - val_loss: 0.7815
Epoch 10/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 350ms/step - accuracy: 0.7584 - loss: 0.5373 - val_accuracy: 0.4000 - val_loss: 0.8369
Epoch 11/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 339ms/step - accuracy: 0.6701 - loss: 0.6092 - val_accuracy: 0.5200 - val_loss: 0.8433
Epoch 12/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 441ms/step - accuracy: 0.6155 - loss: 0.6638 - val_accuracy: 0.3200 - val_loss: 0.8519
Epoch 13/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 443ms/step - accuracy: 0.7187 - loss: 0.6048 - val_accuracy: 0.5200 - val_loss: 0.7317
Epoch 14/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 432ms/step - accuracy: 0.6554 - loss: 0.6353 - val_accuracy: 0.4800 - val_loss: 0.8643
Epoch 15/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 2s 460ms/step - accuracy: 0.7348 - loss: 0.5415 - val_accuracy: 0.5200 - val_loss: 0.7650
Epoch 1/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 16s 911ms/step - accuracy: 0.6705 - loss: 0.6407 - val_accuracy: 0.5600 - val_loss: 0.7522
Epoch 2/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 643ms/step - accuracy: 0.6814 - loss: 0.6934 - val_accuracy: 0.3600 - val_loss: 0.8484
Epoch 3/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 650ms/step - accuracy: 0.6630 - loss: 0.5968 - val_accuracy: 0.4800 - val_loss: 0.7864
Epoch 4/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 704ms/step - accuracy: 0.6268 - loss: 0.6753 - val_accuracy: 0.4400 - val_loss: 0.7861
Epoch 5/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 699ms/step - accuracy: 0.5455 - loss: 0.7601 - val_accuracy: 0.5200 - val_loss: 0.7946
Epoch 6/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 878ms/step - accuracy: 0.6639 - loss: 0.6980 - val_accuracy: 0.4800 - val_loss: 0.7809
Epoch 7/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 649ms/step - accuracy: 0.6516 - loss: 0.6325 - val_accuracy: 0.5200 - val_loss: 0.7839
Epoch 8/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 718ms/step - accuracy: 0.6212 - loss: 0.6619 - val_accuracy: 0.5200 - val_loss: 0.7374
Epoch 9/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 875ms/step - accuracy: 0.6023 - loss: 0.6344 - val_accuracy: 0.3600 - val_loss: 0.9236
Epoch 10/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 886ms/step - accuracy: 0.6349 - loss: 0.6689 - val_accuracy: 0.4800 - val_loss: 0.8191
Epoch 11/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 893ms/step - accuracy: 0.7387 - loss: 0.5405 - val_accuracy: 0.3600 - val_loss: 0.8286
Epoch 12/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 705ms/step - accuracy: 0.6424 - loss: 0.6212 - val_accuracy: 0.3600 - val_loss: 0.8841
Epoch 13/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 754ms/step - accuracy: 0.6671 - loss: 0.6029 - val_accuracy: 0.6000 - val_loss: 0.7618
Epoch 14/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 760ms/step - accuracy: 0.6838 - loss: 0.5720 - val_accuracy: 0.4400 - val_loss: 0.7936
Epoch 15/15
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 718ms/step - accuracy: 0.7572 - loss: 0.5601 - val_accuracy: 0.6000 - val_loss: 0.7802
Model training complete!
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step
Precision: 0.4615
Recall: 0.4615
F1-Score: 0.4615
Specificity: 0.4167

plot12.png

- Convert tensorflow to tensorflow lite
- Deployed on raspberry pi 4 to get all edge deployment params.

- calling base_model.trainable = True unfreezes all 2.2 million weights. I have updated this section to only unfreeze the top 20 layers.

- In Keras, changing the .shuffle attribute after the generator has been initialized is notoriously buggy and often ignored by the backend. As a result, your y_pred (predictions) and your validation_data.classes (actual labels) get completely misaligned.

-  tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dropout(0.5),
Stacking two 50% dropouts doesn't make it better; it mathematically cripples the network by aggressively zeroing out roughly 75% of the neurons right before the final decision. I removed the duplicate layer so your model can actually retain what it learns.

- We need more photos

  To fix this, spend 15 minutes taking about ~370 more photos with your phone to reach the 500 total mark.

  250 Handshake Images: Do not just take photos of your hand in the exact same spot. Reach from the left, reach from the right, stand under bright lights, and stand in dark rooms.

  250 "No Handshake" Images: This is where your model is currently failing. You need to teach it what isn't a handshake. Take photos of:

  An empty room / blank walls.

  Your hand making a fist, a flat "stop" gesture, or a thumbs up.

  Your hand holding a phone, a cup, or a pen.


Added new images of mine (~180 images of both categories) and we see model precision, recall, f1 score, specificity improved but model is still overfitting


python cnn_model_trainer.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
Found 278 images belonging to 2 classes.
Found 69 images belonging to 2 classes.

--- Starting Phase 1: Feature Extraction ---
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
Epoch 1/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 7s 577ms/step - accuracy: 0.5437 - loss: 0.7912 - val_accuracy: 0.4203 - val_loss: 0.8338
Epoch 2/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 364ms/step - accuracy: 0.5761 - loss: 0.6997 - val_accuracy: 0.4348 - val_loss: 0.8719
Epoch 3/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 352ms/step - accuracy: 0.7104 - loss: 0.5324 - val_accuracy: 0.3768 - val_loss: 0.9541
Epoch 4/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 362ms/step - accuracy: 0.7413 - loss: 0.5136 - val_accuracy: 0.4493 - val_loss: 1.0140
Epoch 5/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 354ms/step - accuracy: 0.7960 - loss: 0.4554 - val_accuracy: 0.4638 - val_loss: 0.9879
Epoch 6/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 360ms/step - accuracy: 0.8032 - loss: 0.4223 - val_accuracy: 0.4058 - val_loss: 1.0768
Epoch 7/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 350ms/step - accuracy: 0.8613 - loss: 0.3850 - val_accuracy: 0.4058 - val_loss: 1.1200
Epoch 8/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 356ms/step - accuracy: 0.8748 - loss: 0.4010 - val_accuracy: 0.3913 - val_loss: 1.1721
Epoch 9/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 351ms/step - accuracy: 0.8165 - loss: 0.4259 - val_accuracy: 0.4203 - val_loss: 1.0674
Epoch 10/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 350ms/step - accuracy: 0.8856 - loss: 0.3364 - val_accuracy: 0.4783 - val_loss: 1.0015
Epoch 11/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 360ms/step - accuracy: 0.8283 - loss: 0.3555 - val_accuracy: 0.4058 - val_loss: 1.1821
Epoch 12/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 352ms/step - accuracy: 0.8543 - loss: 0.3214 - val_accuracy: 0.4348 - val_loss: 1.0668
Epoch 13/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 366ms/step - accuracy: 0.8461 - loss: 0.3330 - val_accuracy: 0.4493 - val_loss: 1.0139
Epoch 14/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 354ms/step - accuracy: 0.8778 - loss: 0.3497 - val_accuracy: 0.4348 - val_loss: 1.1885
Epoch 15/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 354ms/step - accuracy: 0.8741 - loss: 0.3167 - val_accuracy: 0.4058 - val_loss: 1.0799

--- Starting Phase 2: Fine-Tuning ---
Epoch 1/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 8s 487ms/step - accuracy: 0.8252 - loss: 0.4692 - val_accuracy: 0.4348 - val_loss: 1.2385
Epoch 2/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 359ms/step - accuracy: 0.8577 - loss: 0.3686 - val_accuracy: 0.4638 - val_loss: 1.2466
Epoch 3/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 360ms/step - accuracy: 0.8407 - loss: 0.3670 - val_accuracy: 0.4203 - val_loss: 1.3127
Epoch 4/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 360ms/step - accuracy: 0.8320 - loss: 0.3750 - val_accuracy: 0.4203 - val_loss: 1.4079
Epoch 5/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 374ms/step - accuracy: 0.8632 - loss: 0.3548 - val_accuracy: 0.4348 - val_loss: 1.4060
Epoch 6/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 375ms/step - accuracy: 0.8649 - loss: 0.3428 - val_accuracy: 0.3768 - val_loss: 1.4373
Epoch 7/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 365ms/step - accuracy: 0.8828 - loss: 0.3041 - val_accuracy: 0.4348 - val_loss: 1.3005
Epoch 8/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 367ms/step - accuracy: 0.8453 - loss: 0.3536 - val_accuracy: 0.4348 - val_loss: 1.3207
Epoch 9/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 369ms/step - accuracy: 0.8694 - loss: 0.3255 - val_accuracy: 0.4638 - val_loss: 1.3748
Epoch 10/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 368ms/step - accuracy: 0.8843 - loss: 0.2912 - val_accuracy: 0.4783 - val_loss: 1.3116
Epoch 11/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 364ms/step - accuracy: 0.8767 - loss: 0.3073 - val_accuracy: 0.4348 - val_loss: 1.4223
Epoch 12/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 367ms/step - accuracy: 0.8280 - loss: 0.3449 - val_accuracy: 0.4783 - val_loss: 1.3228
Epoch 13/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 374ms/step - accuracy: 0.8857 - loss: 0.3513 - val_accuracy: 0.4493 - val_loss: 1.3117
Epoch 14/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 360ms/step - accuracy: 0.8856 - loss: 0.3442 - val_accuracy: 0.4203 - val_loss: 1.3788
Epoch 15/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 361ms/step - accuracy: 0.8952 - loss: 0.2948 - val_accuracy: 0.4058 - val_loss: 1.2560
Model training complete and saved as handshake_model.keras!
Saved learning_curves.png

--- Running Final Evaluation ---
Found 347 images belonging to 2 classes.
11/11 ━━━━━━━━━━━━━━━━━━━━ 3s 191ms/step
Precision:   0.8910
Recall:      0.7277
F1-Score:    0.8012
Specificity: 0.8910
Saved confusion_matrix.png

To solve model overfitting we introduced early stopping

python cnn_model_trainer.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
Found 278 images belonging to 2 classes.
Found 69 images belonging to 2 classes.

--- Starting Phase 1: Feature Extraction ---
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
Epoch 1/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 8s 588ms/step - accuracy: 0.5209 - loss: 0.7819 - val_accuracy: 0.3188 - val_loss: 0.8364
Epoch 2/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 4s 385ms/step - accuracy: 0.6233 - loss: 0.6483 - val_accuracy: 0.3913 - val_loss: 0.9508
Epoch 3/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 358ms/step - accuracy: 0.7427 - loss: 0.5323 - val_accuracy: 0.4058 - val_loss: 0.9198
Epoch 4/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 358ms/step - accuracy: 0.7659 - loss: 0.4855 - val_accuracy: 0.4348 - val_loss: 1.1610
Epoch 5/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 380ms/step - accuracy: 0.8069 - loss: 0.4455 - val_accuracy: 0.4638 - val_loss: 0.9485
Epoch 6/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 361ms/step - accuracy: 0.8185 - loss: 0.4127 - val_accuracy: 0.4493 - val_loss: 1.0039

--- Starting Phase 2: Fine-Tuning ---
Epoch 1/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 8s 497ms/step - accuracy: 0.6428 - loss: 0.6520 - val_accuracy: 0.4783 - val_loss: 0.7735
Epoch 2/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 361ms/step - accuracy: 0.6968 - loss: 0.5894 - val_accuracy: 0.4203 - val_loss: 0.7916
Epoch 3/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 362ms/step - accuracy: 0.6733 - loss: 0.5829 - val_accuracy: 0.4058 - val_loss: 0.7920
Epoch 4/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 367ms/step - accuracy: 0.7095 - loss: 0.5836 - val_accuracy: 0.3768 - val_loss: 0.8614
Epoch 5/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 4s 405ms/step - accuracy: 0.7092 - loss: 0.5873 - val_accuracy: 0.3913 - val_loss: 0.8540
Epoch 6/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 4s 393ms/step - accuracy: 0.7569 - loss: 0.5062 - val_accuracy: 0.4638 - val_loss: 0.8100
Epoch 7/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 382ms/step - accuracy: 0.7270 - loss: 0.5407 - val_accuracy: 0.4203 - val_loss: 0.8801
Epoch 8/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 374ms/step - accuracy: 0.7235 - loss: 0.5477 - val_accuracy: 0.4058 - val_loss: 0.8511
Epoch 9/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 382ms/step - accuracy: 0.7749 - loss: 0.5140 - val_accuracy: 0.3768 - val_loss: 0.8924
Epoch 10/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 364ms/step - accuracy: 0.7934 - loss: 0.4999 - val_accuracy: 0.3913 - val_loss: 0.8845
Epoch 11/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 394ms/step - accuracy: 0.8258 - loss: 0.4266 - val_accuracy: 0.4493 - val_loss: 0.8805
Epoch 12/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 363ms/step - accuracy: 0.8293 - loss: 0.4563 - val_accuracy: 0.4203 - val_loss: 0.8960
Epoch 13/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 362ms/step - accuracy: 0.8151 - loss: 0.4336 - val_accuracy: 0.4783 - val_loss: 0.8972
Epoch 14/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 4s 384ms/step - accuracy: 0.8206 - loss: 0.4171 - val_accuracy: 0.3913 - val_loss: 0.8556
Epoch 15/15
9/9 ━━━━━━━━━━━━━━━━━━━━ 3s 367ms/step - accuracy: 0.7812 - loss: 0.4182 - val_accuracy: 0.3913 - val_loss: 0.8941
Model training complete and saved as handshake_model.keras!
Saved learning_curves.png

--- Running Final Evaluation ---
Found 347 images belonging to 2 classes.
11/11 ━━━━━━━━━━━━━━━━━━━━ 3s 190ms/step
Precision:   0.7831
Recall:      0.7749
F1-Score:    0.7789
Specificity: 0.7372

We need my friend (Shyla) images, I plan to take ~180 images today (90 handshake, 90 none), which hopefully will improve validation accuracy.

Added 200 images


python cnn_model_trainer.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
Found 501 images belonging to 2 classes.
Found 125 images belonging to 2 classes.

--- Starting Phase 1: Feature Extraction ---
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
Epoch 1/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 10s 459ms/step - accuracy: 0.5583 - loss: 0.7440 - val_accuracy: 0.5600 - val_loss: 0.7067
Epoch 2/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 377ms/step - accuracy: 0.6967 - loss: 0.6429 - val_accuracy: 0.6240 - val_loss: 0.7275
Epoch 3/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 378ms/step - accuracy: 0.6985 - loss: 0.5397 - val_accuracy: 0.6000 - val_loss: 0.7178
Epoch 4/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 372ms/step - accuracy: 0.7487 - loss: 0.5260 - val_accuracy: 0.6000 - val_loss: 0.7293
Epoch 5/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 375ms/step - accuracy: 0.7782 - loss: 0.4786 - val_accuracy: 0.5520 - val_loss: 0.8044
Epoch 6/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 374ms/step - accuracy: 0.7965 - loss: 0.4346 - val_accuracy: 0.5360 - val_loss: 0.8380

--- Starting Phase 2: Fine-Tuning ---
Epoch 1/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 11s 467ms/step - accuracy: 0.6252 - loss: 0.6526 - val_accuracy: 0.6320 - val_loss: 0.6669
Epoch 2/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 384ms/step - accuracy: 0.6865 - loss: 0.5948 - val_accuracy: 0.6160 - val_loss: 0.6753
Epoch 3/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 385ms/step - accuracy: 0.6606 - loss: 0.6155 - val_accuracy: 0.5840 - val_loss: 0.7134
Epoch 4/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 381ms/step - accuracy: 0.7008 - loss: 0.5680 - val_accuracy: 0.5840 - val_loss: 0.7161
Epoch 5/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 382ms/step - accuracy: 0.6942 - loss: 0.5859 - val_accuracy: 0.5920 - val_loss: 0.6841
Epoch 6/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 382ms/step - accuracy: 0.7109 - loss: 0.5732 - val_accuracy: 0.5840 - val_loss: 0.7335
Epoch 7/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 383ms/step - accuracy: 0.7403 - loss: 0.5163 - val_accuracy: 0.6320 - val_loss: 0.7072
Epoch 8/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 384ms/step - accuracy: 0.7724 - loss: 0.5029 - val_accuracy: 0.6320 - val_loss: 0.7234
Epoch 9/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 381ms/step - accuracy: 0.7385 - loss: 0.5131 - val_accuracy: 0.6080 - val_loss: 0.7637
Epoch 10/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 385ms/step - accuracy: 0.7780 - loss: 0.4461 - val_accuracy: 0.5840 - val_loss: 0.7856
Epoch 11/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 380ms/step - accuracy: 0.7881 - loss: 0.4613 - val_accuracy: 0.6080 - val_loss: 0.7486
Epoch 12/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 385ms/step - accuracy: 0.7983 - loss: 0.4351 - val_accuracy: 0.6000 - val_loss: 0.7397
Epoch 13/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 384ms/step - accuracy: 0.8028 - loss: 0.4254 - val_accuracy: 0.5760 - val_loss: 0.7752
Epoch 14/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 378ms/step - accuracy: 0.7994 - loss: 0.4485 - val_accuracy: 0.5600 - val_loss: 0.7869
Epoch 15/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 6s 387ms/step - accuracy: 0.8203 - loss: 0.4220 - val_accuracy: 0.6240 - val_loss: 0.7796
Model training complete and saved as handshake_model.keras!
Saved learning_curves.png

--- Running Final Evaluation ---
Found 626 images belonging to 2 classes.
20/20 ━━━━━━━━━━━━━━━━━━━━ 4s 169ms/step 
Precision:   0.8509
Recall:      0.7312
F1-Score:    0.7866
Specificity: 0.8660
Saved confusion_matrix.png


First benchmarking results on 28 clips

(.venv) nagashayanaramamurthy@nagapi:~/handshake-detection $ python benchmarking_script.py 
Error in cpuinfo: prctl(PR_SVE_GET_VL) failed
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1772197069.231706    2548 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1772197069.314496    2548 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in "/home/nagashayanaramamurthy/handshake-detection/.venv/lib/python3.11/site-packages/cv2/qt/plugins"
==================================================
🚀 STARTING NESTED BATCH BENCHMARK ON RASPBERRY PI...
==================================================
Processing handshake/clip_10 (43 frames)...
W0000 00:00:1772197069.938039    2545 landmark_projection_calculator.cc:186] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Processing handshake/clip_11 (48 frames)...
Processing handshake/clip_12 (43 frames)...
Processing handshake/clip_13 (59 frames)...
Processing handshake/clip_14 (49 frames)...
Processing handshake/clip_15 (39 frames)...
Processing handshake/clip_16 (46 frames)...
Processing handshake/clip_17 (53 frames)...
Processing handshake/clip_3 (43 frames)...
Processing handshake/clip_4 (42 frames)...
Processing handshake/clip_5 (44 frames)...
Processing handshake/clip_6 (47 frames)...
Processing handshake/clip_7 (40 frames)...
Processing handshake/clip_8 (44 frames)...
Processing handshake/clip_9 (35 frames)...
Processing none/clip_18 (35 frames)...
Processing none/clip_19 (38 frames)...
Processing none/clip_20 (41 frames)...
Processing none/clip_21 (41 frames)...
Processing none/clip_22 (40 frames)...
Processing none/clip_23 (40 frames)...
Processing none/clip_24 (42 frames)...
Processing none/clip_25 (38 frames)...
Processing none/clip_26 (42 frames)...
Processing none/clip_27 (40 frames)...
Processing none/clip_28 (36 frames)...
Processing none/clip_29 (37 frames)...
Processing none/clip_30 (53 frames)...

==================================================
📊 FINAL HARDWARE INFERENCE LATENCY (Averaged)
==================================================
Total Frames Benchmarked : 1193
MediaPipe Tracking (ms)  : 133.20 ms
Spatial SBF Logic (ms)   : 0.40 ms
MobileNetV2 CNN (ms)     : 86.13 ms
--------------------------------------------------
Total Pipeline Latency   : 219.73 ms
Estimated Real-Time FPS  : 4.6 FPS

==================================================
🎯 DYNAMIC SEQUENCE ACCURACY (Clip-Level)
==================================================
True Positives (Hit)     : 14
False Positives (Miss)   : 0
True Negatives (Correct) : 13
False Negatives (Miss)   : 1


Improvements?

One Important Technical Suggestion

If you want to improve FPS:
	•	Use TensorFlow Lite model
	•	Quantize MobileNet to INT8
	•	Reduce MediaPipe model complexity
	•	Process every 2nd frame
	•	Use asynchronous threading

But that’s for later.

For now, 4.6 FPS is acceptable for assistive interaction detection.

Converting to INT8.
convert_to_int8.py

FPS didn't increase


 python benchmarking_script.py 
Error in cpuinfo: prctl(PR_SVE_GET_VL) failed
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1772208036.006690    2964 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1772208036.095661    2965 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in "/home/nagashayanaramamurthy/handshake-detection/.venv/lib/python3.11/site-packages/cv2/qt/plugins"
==================================================
🚀 STARTING NESTED BATCH BENCHMARK ON RASPBERRY PI...
==================================================
Processing handshake/clip_10 (43 frames)...
W0000 00:00:1772208037.205060    2963 landmark_projection_calculator.cc:186] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Processing handshake/clip_11 (48 frames)...
Processing handshake/clip_12 (43 frames)...
Processing handshake/clip_13 (59 frames)...
Processing handshake/clip_14 (49 frames)...
Processing handshake/clip_15 (39 frames)...
Processing handshake/clip_16 (46 frames)...
Processing handshake/clip_17 (53 frames)...
Processing handshake/clip_3 (43 frames)...
Processing handshake/clip_4 (42 frames)...
Processing handshake/clip_5 (44 frames)...
Processing handshake/clip_6 (47 frames)...
Processing handshake/clip_7 (40 frames)...
Processing handshake/clip_8 (44 frames)...
Processing handshake/clip_9 (35 frames)...
Processing none/clip_18 (35 frames)...
Processing none/clip_19 (38 frames)...
Processing none/clip_20 (41 frames)...
Processing none/clip_21 (41 frames)...
Processing none/clip_22 (40 frames)...
Processing none/clip_23 (40 frames)...
Processing none/clip_24 (42 frames)...
Processing none/clip_25 (38 frames)...
Processing none/clip_26 (42 frames)...
Processing none/clip_27 (40 frames)...
Processing none/clip_28 (36 frames)...
Processing none/clip_29 (37 frames)...
Processing none/clip_30 (53 frames)...

==================================================
📊 FINAL HARDWARE INFERENCE LATENCY (Averaged)
==================================================
Total Frames Benchmarked : 1193
MediaPipe Tracking (ms)  : 135.59 ms
Spatial SBF Logic (ms)   : 0.39 ms
MobileNetV2 CNN (ms)     : 93.49 ms
--------------------------------------------------
Total Pipeline Latency   : 229.48 ms
Estimated Real-Time FPS  : 4.4 FPS

==================================================
🎯 DYNAMIC SEQUENCE ACCURACY (Clip-Level)
==================================================
True Positives (Hit)     : 14
False Positives (Miss)   : 0
True Negatives (Correct) : 13
False Negatives (Miss)   : 1

Few more tweeks we can try:

Start simple:
	1.	Remove Gaussian blur in inference.
	2.	Reduce input to 160×160.
	3.	Run MediaPipe every 6 frames.
	4.	Only run CNN if hand detected.

along with previous tips

	•	Use TensorFlow Lite model
	•	Quantize MobileNet to INT8
	•	Reduce MediaPipe model complexity
	•	Process every 2nd frame
	•	Use asynchronous threading

Then re-benchmark.

reduce img size to 160 from 224

Epoch 1/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 8s 340ms/step - accuracy: 0.4333 - loss: 0.9544 - val_accuracy: 0.5440 - val_loss: 0.6654
Epoch 2/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 275ms/step - accuracy: 0.5840 - loss: 0.7445 - val_accuracy: 0.5840 - val_loss: 0.6838
Epoch 3/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 274ms/step - accuracy: 0.6891 - loss: 0.5978 - val_accuracy: 0.5280 - val_loss: 0.7203
Epoch 4/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 5s 281ms/step - accuracy: 0.6938 - loss: 0.6034 - val_accuracy: 0.5280 - val_loss: 0.7725
Epoch 5/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 268ms/step - accuracy: 0.7850 - loss: 0.5097 - val_accuracy: 0.6000 - val_loss: 0.7180
Epoch 6/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 267ms/step - accuracy: 0.7710 - loss: 0.4730 - val_accuracy: 0.5280 - val_loss: 0.7319

--- Starting Phase 2: Fine-Tuning ---
Epoch 1/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 9s 340ms/step - accuracy: 0.5412 - loss: 0.7165 - val_accuracy: 0.6160 - val_loss: 0.6826
Epoch 2/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 271ms/step - accuracy: 0.5824 - loss: 0.7064 - val_accuracy: 0.5920 - val_loss: 0.6617
Epoch 3/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 5s 283ms/step - accuracy: 0.6714 - loss: 0.6526 - val_accuracy: 0.5520 - val_loss: 0.7111
Epoch 4/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 277ms/step - accuracy: 0.6950 - loss: 0.5831 - val_accuracy: 0.5840 - val_loss: 0.7065
Epoch 5/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 274ms/step - accuracy: 0.6808 - loss: 0.5880 - val_accuracy: 0.5920 - val_loss: 0.7107
Epoch 6/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 272ms/step - accuracy: 0.6309 - loss: 0.6130 - val_accuracy: 0.5760 - val_loss: 0.7217
Epoch 7/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 275ms/step - accuracy: 0.6768 - loss: 0.6080 - val_accuracy: 0.5680 - val_loss: 0.7152
Epoch 8/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 274ms/step - accuracy: 0.7301 - loss: 0.5245 - val_accuracy: 0.5520 - val_loss: 0.7455
Epoch 9/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 276ms/step - accuracy: 0.7672 - loss: 0.4961 - val_accuracy: 0.5840 - val_loss: 0.7509
Epoch 10/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 277ms/step - accuracy: 0.7349 - loss: 0.5375 - val_accuracy: 0.6080 - val_loss: 0.7327
Epoch 11/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 272ms/step - accuracy: 0.7451 - loss: 0.5067 - val_accuracy: 0.5760 - val_loss: 0.7538
Epoch 12/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 270ms/step - accuracy: 0.7789 - loss: 0.4760 - val_accuracy: 0.5440 - val_loss: 0.7629
Epoch 13/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 271ms/step - accuracy: 0.7559 - loss: 0.5114 - val_accuracy: 0.5600 - val_loss: 0.7502
Epoch 14/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 272ms/step - accuracy: 0.7932 - loss: 0.4682 - val_accuracy: 0.5840 - val_loss: 0.7739
Epoch 15/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 275ms/step - accuracy: 0.7973 - loss: 0.4631 - val_accuracy: 0.5920 - val_loss: 0.7759
Model training complete and saved as handshake_model.keras!
Saved learning_curves.png

--- Running Final Evaluation ---
Found 626 images belonging to 2 classes.
20/20 ━━━━━━━━━━━━━━━━━━━━ 4s 153ms/step 
Precision:   0.8179
Recall:      0.7156
F1-Score:    0.7633
Specificity: 0.8333
Saved confusion_matrix.png


After reducing input to 160*160 size

==================================================
🚀 STARTING NESTED BATCH BENCHMARK ON RASPBERRY PI...
==================================================
Processing handshake/clip_10 (43 frames)...
W0000 00:00:1772211111.264186    3138 landmark_projection_calculator.cc:186] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Processing handshake/clip_11 (48 frames)...
Processing handshake/clip_12 (43 frames)...
Processing handshake/clip_13 (59 frames)...
Processing handshake/clip_14 (49 frames)...
Processing handshake/clip_15 (39 frames)...
Processing handshake/clip_16 (46 frames)...
Processing handshake/clip_17 (53 frames)...
Processing handshake/clip_3 (43 frames)...
Processing handshake/clip_4 (42 frames)...
Processing handshake/clip_5 (44 frames)...
Processing handshake/clip_6 (47 frames)...
Processing handshake/clip_7 (40 frames)...
Processing handshake/clip_8 (44 frames)...
Processing handshake/clip_9 (35 frames)...
Processing none/clip_18 (35 frames)...
Processing none/clip_19 (38 frames)...
Processing none/clip_20 (41 frames)...
Processing none/clip_21 (41 frames)...
Processing none/clip_22 (40 frames)...
Processing none/clip_23 (40 frames)...
Processing none/clip_24 (42 frames)...
Processing none/clip_25 (38 frames)...
Processing none/clip_26 (42 frames)...
Processing none/clip_27 (40 frames)...
Processing none/clip_28 (36 frames)...
Processing none/clip_29 (37 frames)...
Processing none/clip_30 (53 frames)...

==================================================
📊 FINAL HARDWARE INFERENCE LATENCY (Averaged)
==================================================
Total Frames Benchmarked : 1193
MediaPipe Tracking (ms)  : 133.70 ms
Spatial SBF Logic (ms)   : 0.39 ms
MobileNetV2 CNN (ms)     : 46.60 ms
--------------------------------------------------
Total Pipeline Latency   : 180.69 ms
Estimated Real-Time FPS  : 5.5 FPS

==================================================
🎯 DYNAMIC SEQUENCE ACCURACY (Clip-Level)
==================================================
True Positives (Hit)     : 13
False Positives (Miss)   : 0
True Negatives (Correct) : 13
False Negatives (Miss)   : 2


thats gold

We gated CNN, we run CNN only when hand is detected and reach & pose is validated.
We also used landmark lite mode which earlier increased FPS to 5.6.

==================================================
📊 FINAL HARDWARE INFERENCE LATENCY (Averaged)
==================================================
Total Frames Benchmarked : 1193
MediaPipe Tracking (ms)  : 133.87 ms
Spatial SBF Logic (ms)   : 0.39 ms
MobileNetV2 CNN (ms)     : 11.41 ms
--------------------------------------------------
Total Pipeline Latency   : 145.67 ms
Estimated Real-Time FPS  : 6.9 FPS

==================================================
🎯 DYNAMIC SEQUENCE ACCURACY (Clip-Level)
==================================================
True Positives (Hit)     : 13
False Positives (Miss)   : 0
True Negatives (Correct) : 13
False Negatives (Miss)   : 2

Highest and best so far.
stopping here for now.

"While the system achieved high sensitivity, failure case analysis revealed that extreme vertical offsets (e.g., handshakes initiated significantly above or below the camera's horizon line) induce severe 2D perspective distortion. This distortion temporarily invalidates the SBF's palm-tilt estimations. Future iterations could address this by leveraging the full 3D rotation matrix provided by MediaPipe, rather than relying on 2D planar projections."

clip_11 (low handshake) and clip_12 (high handshake) in sequence_dataset_v3

"The pipeline achieved an 85.7% True Positive rate with strict zero False Positives. Failure case analysis revealed two primary limitations: extreme perspective collapse during 'low' handshakes (clip_11, clip_12), and temporal misalignment where peak neural confidence and peak spatial stability failed to synchronize within the same frame during rapid gestures (clip_10)."

For ICHORA deadline perspective, I will go back to last 6.9 freeze but generate new 640*480 dataset for 2 people.
dataset will be generated to minimize multiple hands, the model is not trained and SBF is not mature enough to handle different hands so benchmark drops.

DATASET info:
- p1_dataset: is the final official dataset for paper 1 ICHORA
  TODO:
    - Add my none category images
    - Sort through complex unwanted images of handshake/none and move to p2_dataset folder
- p2_dataset: This is for future paper


Ablation study1: with out palm tilt
python benchmarking_script_v4.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
I0000 00:00:1772831163.479944 5732913 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1772831163.495704 5732915 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1772831163.512122 5732915 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
==================================================
🚀 STARTING ABLATION BATCH BENCHMARK (NO TILT)...
==================================================
Processing handshake/clip_0 (90 frames)...
W0000 00:00:1772831163.779068 5732921 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Processing handshake/clip_1 (90 frames)...
Processing handshake/clip_10 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_10
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.199
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.940 (Needs >0.6)
   -> Max CNN Raw Score     : 0.351
   -> Max Fused Avg Conf    : 0.540 (Needs >0.5)

Processing handshake/clip_11 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_11
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.091
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.988 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.198 (Needs >0.5)

Processing handshake/clip_12 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_12
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.061
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.957 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.191 (Needs >0.5)

Processing handshake/clip_13 (90 frames)...
Processing handshake/clip_14 (90 frames)...
Processing handshake/clip_15 (90 frames)...
Processing handshake/clip_16 (90 frames)...
Processing handshake/clip_17 (90 frames)...
Processing handshake/clip_18 (90 frames)...
Processing handshake/clip_19 (90 frames)...
Processing handshake/clip_2 (90 frames)...
Processing handshake/clip_20 (90 frames)...
Processing handshake/clip_21 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_21
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.060
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.804 (Needs >0.6)
   -> Max CNN Raw Score     : 0.186
   -> Max Fused Avg Conf    : 0.447 (Needs >0.5)

Processing handshake/clip_22 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_22
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.093
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.960 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.192 (Needs >0.5)

Processing handshake/clip_23 (90 frames)...
Processing handshake/clip_24 (90 frames)...
Processing handshake/clip_25 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_25
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.307
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.531 (Needs >0.6)
   -> Max CNN Raw Score     : 0.210
   -> Max Fused Avg Conf    : 0.581 (Needs >0.5)

Processing handshake/clip_26 (90 frames)...
Processing handshake/clip_27 (90 frames)...
Processing handshake/clip_28 (90 frames)...
Processing handshake/clip_29 (90 frames)...
Processing handshake/clip_3 (90 frames)...
Processing handshake/clip_30 (90 frames)...
Processing handshake/clip_31 (90 frames)...
Processing handshake/clip_32 (90 frames)...
Processing handshake/clip_33 (90 frames)...
Processing handshake/clip_34 (90 frames)...
Processing handshake/clip_35 (90 frames)...
Processing handshake/clip_36 (90 frames)...
Processing handshake/clip_37 (90 frames)...
Processing handshake/clip_38 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_38
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.373
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.586 (Needs >0.6)
   -> Max CNN Raw Score     : 0.780
   -> Max Fused Avg Conf    : 0.730 (Needs >0.5)

Processing handshake/clip_39 (90 frames)...
Processing handshake/clip_4 (90 frames)...
Processing handshake/clip_40 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_40
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.037
   -> Thumb Ever Open?      : False
   -> Max Stability Score   : 0.974 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.195 (Needs >0.5)

Processing handshake/clip_41 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_41
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.116
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.969 (Needs >0.6)
   -> Max CNN Raw Score     : 0.332
   -> Max Fused Avg Conf    : 0.494 (Needs >0.5)

Processing handshake/clip_5 (90 frames)...
Processing handshake/clip_6 (90 frames)...
Processing handshake/clip_7 (90 frames)...
Processing handshake/clip_8 (90 frames)...
Processing handshake/clip_9 (90 frames)...
Processing none/clip_21 (90 frames)...
Processing none/clip_22 (90 frames)...
Processing none/clip_23 (90 frames)...
Processing none/clip_24 (90 frames)...
Processing none/clip_25 (90 frames)...
Processing none/clip_26 (90 frames)...
Processing none/clip_27 (90 frames)...
Processing none/clip_28 (90 frames)...
Processing none/clip_29 (90 frames)...
Processing none/clip_30 (90 frames)...
Processing none/clip_31 (90 frames)...
Processing none/clip_32 (90 frames)...
Processing none/clip_33 (90 frames)...
Processing none/clip_34 (90 frames)...
Processing none/clip_35 (90 frames)...
Processing none/clip_36 (90 frames)...
Processing none/clip_37 (90 frames)...
Processing none/clip_38 (90 frames)...
Processing none/clip_39 (90 frames)...
Processing none/clip_40 (90 frames)...

==================================================
📊 FINAL HARDWARE INFERENCE LATENCY (Averaged)
==================================================
Total Frames Benchmarked : 5575
MediaPipe Tracking (ms)  : 25.72 ms
Spatial SBF Logic (ms)   : 0.10 ms
MobileNetV2 CNN (ms)     : 1.60 ms
--------------------------------------------------
Total Pipeline Latency   : 27.42 ms
Estimated Real-Time FPS  : 36.5 FPS

==================================================
🎯 DYNAMIC SEQUENCE ACCURACY (Clip-Level)
==================================================
True Positives (Hit)     : 33
False Positives (Miss)   : 17
True Negatives (Correct) : 3
False Negatives (Miss)   : 9

Ablation study2: with palm tilt of 20-120
(.venv) nagashayanaramamurthy@Nagas-MacBook-Pro update_model % python benchmarking_script_v3.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
I0000 00:00:1772831512.303914 5737266 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1772831512.319229 5737267 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1772831512.340330 5737267 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
==================================================
🚀 STARTING NESTED BATCH BENCHMARK ON MAC (WITH AUTOPSY)...
==================================================
Processing handshake/clip_0 (90 frames)...
W0000 00:00:1772831512.582883 5737268 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Processing handshake/clip_1 (90 frames)...
Processing handshake/clip_10 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_10
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.199
   -> Palm Tilt (Needs 20-160): 0.2 to 179.5
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.940 (Needs >0.6)
   -> Max CNN Raw Score     : 0.351
   -> Max Fused Avg Conf    : 0.540 (Needs >0.5)

Processing handshake/clip_11 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_11
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.091
   -> Palm Tilt (Needs 20-160): 0.6 to 179.2
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.988 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.198 (Needs >0.5)

Processing handshake/clip_12 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_12
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.061
   -> Palm Tilt (Needs 20-160): 0.0 to 179.2
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.957 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.191 (Needs >0.5)

Processing handshake/clip_13 (90 frames)...
Processing handshake/clip_14 (90 frames)...
Processing handshake/clip_15 (90 frames)...
Processing handshake/clip_16 (90 frames)...
Processing handshake/clip_17 (90 frames)...
Processing handshake/clip_18 (90 frames)...
Processing handshake/clip_19 (90 frames)...
Processing handshake/clip_2 (90 frames)...
Processing handshake/clip_20 (90 frames)...
Processing handshake/clip_21 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_21
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.060
   -> Palm Tilt (Needs 20-160): 2.6 to 178.6
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.804 (Needs >0.6)
   -> Max CNN Raw Score     : 0.186
   -> Max Fused Avg Conf    : 0.447 (Needs >0.5)

Processing handshake/clip_22 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_22
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.093
   -> Palm Tilt (Needs 20-160): 38.2 to 178.9
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.960 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.192 (Needs >0.5)

Processing handshake/clip_23 (90 frames)...
Processing handshake/clip_24 (90 frames)...
Processing handshake/clip_25 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_25
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.307
   -> Palm Tilt (Needs 20-160): 59.4 to 93.3
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.531 (Needs >0.6)
   -> Max CNN Raw Score     : 0.210
   -> Max Fused Avg Conf    : 0.581 (Needs >0.5)

Processing handshake/clip_26 (90 frames)...
Processing handshake/clip_27 (90 frames)...
Processing handshake/clip_28 (90 frames)...
Processing handshake/clip_29 (90 frames)...
Processing handshake/clip_3 (90 frames)...
Processing handshake/clip_30 (90 frames)...
Processing handshake/clip_31 (90 frames)...
Processing handshake/clip_32 (90 frames)...
Processing handshake/clip_33 (90 frames)...
Processing handshake/clip_34 (90 frames)...
Processing handshake/clip_35 (90 frames)...
Processing handshake/clip_36 (90 frames)...
Processing handshake/clip_37 (90 frames)...
Processing handshake/clip_38 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_38
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.373
   -> Palm Tilt (Needs 20-160): 22.6 to 176.8
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.586 (Needs >0.6)
   -> Max CNN Raw Score     : 0.780
   -> Max Fused Avg Conf    : 0.730 (Needs >0.5)

Processing handshake/clip_39 (90 frames)...
Processing handshake/clip_4 (90 frames)...
Processing handshake/clip_40 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_40
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.037
   -> Palm Tilt (Needs 20-160): 25.7 to 179.8
   -> Thumb Ever Open?      : False
   -> Max Stability Score   : 0.974 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.195 (Needs >0.5)

Processing handshake/clip_41 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_41
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.116
   -> Palm Tilt (Needs 20-160): 0.0 to 177.1
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.969 (Needs >0.6)
   -> Max CNN Raw Score     : 0.332
   -> Max Fused Avg Conf    : 0.494 (Needs >0.5)

Processing handshake/clip_5 (90 frames)...
Processing handshake/clip_6 (90 frames)...
Processing handshake/clip_7 (90 frames)...
Processing handshake/clip_8 (90 frames)...
Processing handshake/clip_9 (90 frames)...
Processing none/clip_21 (90 frames)...
Processing none/clip_22 (90 frames)...
Processing none/clip_23 (90 frames)...
Processing none/clip_24 (90 frames)...
Processing none/clip_25 (90 frames)...
Processing none/clip_26 (90 frames)...
Processing none/clip_27 (90 frames)...
Processing none/clip_28 (90 frames)...
Processing none/clip_29 (90 frames)...
Processing none/clip_30 (90 frames)...
Processing none/clip_31 (90 frames)...
Processing none/clip_32 (90 frames)...
Processing none/clip_33 (90 frames)...
Processing none/clip_34 (90 frames)...
Processing none/clip_35 (90 frames)...
Processing none/clip_36 (90 frames)...
Processing none/clip_37 (90 frames)...
Processing none/clip_38 (90 frames)...
Processing none/clip_39 (90 frames)...
Processing none/clip_40 (90 frames)...

==================================================
📊 FINAL HARDWARE INFERENCE LATENCY (Averaged)
==================================================
Total Frames Benchmarked : 5575
MediaPipe Tracking (ms)  : 25.76 ms
Spatial SBF Logic (ms)   : 0.10 ms
MobileNetV2 CNN (ms)     : 1.56 ms
--------------------------------------------------
Total Pipeline Latency   : 27.42 ms
Estimated Real-Time FPS  : 36.5 FPS

==================================================
🎯 DYNAMIC SEQUENCE ACCURACY (Clip-Level)
==================================================
True Positives (Hit)     : 33
False Positives (Miss)   : 15
True Negatives (Correct) : 5
False Negatives (Miss)   : 9

Ablation study3: with palm tilt 45-120

python benchmarking_script_v3.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
I0000 00:00:1772885451.893953 6302554 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1772885451.907179 6302558 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1772885451.922509 6302558 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
==================================================
🚀 STARTING NESTED BATCH BENCHMARK ON MAC (WITH AUTOPSY)...
==================================================
Processing handshake/clip_0 (90 frames)...
W0000 00:00:1772885452.242436 6302562 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Processing handshake/clip_1 (90 frames)...
Processing handshake/clip_10 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_10
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.199
   -> Palm Tilt (Needs 45-135): 0.2 to 179.5
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.940 (Needs >0.6)
   -> Max CNN Raw Score     : 0.351
   -> Max Fused Avg Conf    : 0.540 (Needs >0.5)

Processing handshake/clip_11 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_11
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.091
   -> Palm Tilt (Needs 45-135): 0.6 to 179.2
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.988 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.198 (Needs >0.5)

Processing handshake/clip_12 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_12
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.061
   -> Palm Tilt (Needs 45-135): 0.0 to 179.2
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.957 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.191 (Needs >0.5)

Processing handshake/clip_13 (90 frames)...
Processing handshake/clip_14 (90 frames)...
Processing handshake/clip_15 (90 frames)...
Processing handshake/clip_16 (90 frames)...
Processing handshake/clip_17 (90 frames)...
Processing handshake/clip_18 (90 frames)...
Processing handshake/clip_19 (90 frames)...
Processing handshake/clip_2 (90 frames)...
Processing handshake/clip_20 (90 frames)...
Processing handshake/clip_21 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_21
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.060
   -> Palm Tilt (Needs 45-135): 2.6 to 178.6
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.804 (Needs >0.6)
   -> Max CNN Raw Score     : 0.186
   -> Max Fused Avg Conf    : 0.447 (Needs >0.5)

Processing handshake/clip_22 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_22
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.093
   -> Palm Tilt (Needs 45-135): 38.2 to 178.9
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.960 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.192 (Needs >0.5)

Processing handshake/clip_23 (90 frames)...
Processing handshake/clip_24 (90 frames)...
Processing handshake/clip_25 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_25
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.307
   -> Palm Tilt (Needs 45-135): 59.4 to 93.3
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.531 (Needs >0.6)
   -> Max CNN Raw Score     : 0.210
   -> Max Fused Avg Conf    : 0.581 (Needs >0.5)

Processing handshake/clip_26 (90 frames)...
Processing handshake/clip_27 (90 frames)...
Processing handshake/clip_28 (90 frames)...
Processing handshake/clip_29 (90 frames)...
Processing handshake/clip_3 (90 frames)...
Processing handshake/clip_30 (90 frames)...
Processing handshake/clip_31 (90 frames)...
Processing handshake/clip_32 (90 frames)...
Processing handshake/clip_33 (90 frames)...
Processing handshake/clip_34 (90 frames)...
Processing handshake/clip_35 (90 frames)...
Processing handshake/clip_36 (90 frames)...
Processing handshake/clip_37 (90 frames)...
Processing handshake/clip_38 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_38
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.373
   -> Palm Tilt (Needs 45-135): 22.6 to 176.8
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.586 (Needs >0.6)
   -> Max CNN Raw Score     : 0.780
   -> Max Fused Avg Conf    : 0.730 (Needs >0.5)

Processing handshake/clip_39 (90 frames)...
Processing handshake/clip_4 (90 frames)...
Processing handshake/clip_40 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_40
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.037
   -> Palm Tilt (Needs 45-135): 25.7 to 179.8
   -> Thumb Ever Open?      : False
   -> Max Stability Score   : 0.974 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.195 (Needs >0.5)

Processing handshake/clip_41 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_41
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.116
   -> Palm Tilt (Needs 45-135): 0.0 to 177.1
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.969 (Needs >0.6)
   -> Max CNN Raw Score     : 0.332
   -> Max Fused Avg Conf    : 0.494 (Needs >0.5)

Processing handshake/clip_5 (90 frames)...
Processing handshake/clip_6 (90 frames)...
Processing handshake/clip_7 (90 frames)...
Processing handshake/clip_8 (90 frames)...
Processing handshake/clip_9 (90 frames)...
Processing none/clip_21 (90 frames)...
Processing none/clip_22 (90 frames)...
Processing none/clip_23 (90 frames)...
Processing none/clip_24 (90 frames)...
Processing none/clip_25 (90 frames)...
Processing none/clip_26 (90 frames)...
Processing none/clip_27 (90 frames)...
Processing none/clip_28 (90 frames)...
Processing none/clip_29 (90 frames)...
Processing none/clip_30 (90 frames)...
Processing none/clip_31 (90 frames)...
Processing none/clip_32 (90 frames)...
Processing none/clip_33 (90 frames)...
Processing none/clip_34 (90 frames)...
Processing none/clip_35 (90 frames)...
Processing none/clip_36 (90 frames)...
Processing none/clip_37 (90 frames)...
Processing none/clip_38 (90 frames)...
Processing none/clip_39 (90 frames)...
Processing none/clip_40 (90 frames)...

==================================================
📊 FINAL HARDWARE INFERENCE LATENCY (Averaged)
==================================================
Total Frames Benchmarked : 5575
MediaPipe Tracking (ms)  : 26.11 ms
Spatial SBF Logic (ms)   : 0.10 ms
MobileNetV2 CNN (ms)     : 1.47 ms
--------------------------------------------------
Total Pipeline Latency   : 27.68 ms
Estimated Real-Time FPS  : 36.1 FPS

==================================================
🎯 DYNAMIC SEQUENCE ACCURACY (Clip-Level)
==================================================
True Positives (Hit)     : 33
False Positives (Miss)   : 13
True Negatives (Correct) : 7
False Negatives (Miss)   : 9


why we have to solve the problem
how are we solving
our results are they supporting it

how did we capture data? how much frame rate? 10 frames in time? how many seconds of temporal stability.
mention vertical peripheral and talk about that degrees.

my review:
- we should also have a similar confusion matrix for adversary datasset and also mention in caption 
this is a standard dataset
- The pipeline or any text doesn't explain how SBF and temporal stability acts + temporal queue and diff components easily


--- After professor feedback ---
1. Run normal benchmarking
python benchmarking_script_v3.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
I0000 00:00:1775303418.062446 3207941 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1775303418.079995 3207942 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1775303418.107183 3207944 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
==================================================
🚀 STARTING NESTED BATCH BENCHMARK ON MAC (WITH AUTOPSY)...
==================================================
Processing handshake/clip_0 (90 frames)...
W0000 00:00:1775303418.503574 3207945 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Processing handshake/clip_1 (90 frames)...
Processing handshake/clip_10 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_10
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.199
   -> Palm Tilt (Needs 45-135): 0.2 to 179.5
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.940 (Needs >0.6)
   -> Max CNN Raw Score     : 0.351
   -> Max Fused Avg Conf    : 0.540 (Needs >0.5)

Processing handshake/clip_13 (90 frames)...
Processing handshake/clip_14 (90 frames)...
Processing handshake/clip_15 (90 frames)...
Processing handshake/clip_16 (90 frames)...
Processing handshake/clip_17 (90 frames)...
Processing handshake/clip_18 (90 frames)...
Processing handshake/clip_19 (90 frames)...
Processing handshake/clip_2 (90 frames)...
Processing handshake/clip_20 (90 frames)...
Processing handshake/clip_23 (89 frames)...
Processing handshake/clip_24 (90 frames)...
Processing handshake/clip_26 (90 frames)...
Processing handshake/clip_27 (90 frames)...
Processing handshake/clip_28 (90 frames)...
Processing handshake/clip_29 (90 frames)...
Processing handshake/clip_3 (90 frames)...
Processing handshake/clip_30 (90 frames)...
Processing handshake/clip_31 (90 frames)...
Processing handshake/clip_32 (90 frames)...
Processing handshake/clip_33 (90 frames)...
Processing handshake/clip_34 (90 frames)...
Processing handshake/clip_35 (90 frames)...
Processing handshake/clip_36 (90 frames)...
Processing handshake/clip_37 (90 frames)...
Processing handshake/clip_4 (90 frames)...
Processing handshake/clip_41 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_41
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.116
   -> Palm Tilt (Needs 45-135): 0.0 to 177.1
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.969 (Needs >0.6)
   -> Max CNN Raw Score     : 0.332
   -> Max Fused Avg Conf    : 0.494 (Needs >0.5)

Processing handshake/clip_5 (90 frames)...
Processing handshake/clip_6 (90 frames)...
Processing handshake/clip_7 (90 frames)...
Processing handshake/clip_8 (90 frames)...
Processing handshake/clip_9 (90 frames)...
Processing none/clip_23 (89 frames)...
Processing none/clip_28 (90 frames)...
Processing none/clip_29 (90 frames)...
Processing none/clip_32 (90 frames)...
Processing none/clip_33 (90 frames)...
Processing none/clip_35 (89 frames)...
Processing none/clip_39 (90 frames)...

==================================================
📊 FINAL HARDWARE INFERENCE LATENCY (Averaged)
==================================================
Total Frames Benchmarked : 3682
MediaPipe Tracking (ms)  : 26.78 ms
Spatial SBF Logic (ms)   : 0.11 ms
MobileNetV2 CNN (ms)     : 1.61 ms
--------------------------------------------------
Total Pipeline Latency   : 28.49 ms
Estimated Real-Time FPS  : 35.1 FPS

==================================================
🎯 DYNAMIC SEQUENCE ACCURACY (Clip-Level)
==================================================
True Positives (Hit)     : 32
False Positives (Miss)   : 0
True Negatives (Correct) : 7
False Negatives (Miss)   : 2

2. Now run with             fused_pred = (0 * cnn_raw) + (0.6 * pose_score) + (0.4 * stability_score)
so CNN is cancelled

python benchmarking_script_v3.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
I0000 00:00:1775304101.469635 3216484 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1775304101.484038 3216487 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1775304101.498984 3216487 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
==================================================
🚀 STARTING NESTED BATCH BENCHMARK ON MAC (WITH AUTOPSY)...
==================================================
Processing handshake/clip_0 (90 frames)...
W0000 00:00:1775304101.795461 3216488 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Processing handshake/clip_1 (90 frames)...
Processing handshake/clip_10 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_10
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.199
   -> Palm Tilt (Needs 45-135): 0.2 to 179.5
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.940 (Needs >0.6)
   -> Max CNN Raw Score     : 0.351
   -> Max Fused Avg Conf    : 0.600 (Needs >0.5)

Processing handshake/clip_13 (90 frames)...
Processing handshake/clip_14 (90 frames)...
Processing handshake/clip_15 (90 frames)...
Processing handshake/clip_16 (90 frames)...
Processing handshake/clip_17 (90 frames)...
Processing handshake/clip_18 (90 frames)...
Processing handshake/clip_19 (90 frames)...
Processing handshake/clip_2 (90 frames)...
Processing handshake/clip_20 (90 frames)...
Processing handshake/clip_23 (89 frames)...
Processing handshake/clip_24 (90 frames)...
Processing handshake/clip_26 (90 frames)...
Processing handshake/clip_27 (90 frames)...
Processing handshake/clip_28 (90 frames)...
Processing handshake/clip_29 (90 frames)...
Processing handshake/clip_3 (90 frames)...
Processing handshake/clip_30 (90 frames)...
Processing handshake/clip_31 (90 frames)...
Processing handshake/clip_32 (90 frames)...
Processing handshake/clip_33 (90 frames)...
Processing handshake/clip_34 (90 frames)...
Processing handshake/clip_35 (90 frames)...
Processing handshake/clip_36 (90 frames)...
Processing handshake/clip_37 (90 frames)...
Processing handshake/clip_4 (90 frames)...
Processing handshake/clip_41 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_41
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.116
   -> Palm Tilt (Needs 45-135): 0.0 to 177.1
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.969 (Needs >0.6)
   -> Max CNN Raw Score     : 0.332
   -> Max Fused Avg Conf    : 0.600 (Needs >0.5)

Processing handshake/clip_5 (90 frames)...
Processing handshake/clip_6 (90 frames)...
Processing handshake/clip_7 (90 frames)...
Processing handshake/clip_8 (90 frames)...
Processing handshake/clip_9 (90 frames)...
Processing none/clip_23 (89 frames)...
Processing none/clip_28 (90 frames)...
Processing none/clip_29 (90 frames)...
Processing none/clip_32 (90 frames)...
Processing none/clip_33 (90 frames)...
Processing none/clip_35 (89 frames)...
Processing none/clip_39 (90 frames)...

==================================================
📊 FINAL HARDWARE INFERENCE LATENCY (Averaged)
==================================================
Total Frames Benchmarked : 3682
MediaPipe Tracking (ms)  : 27.00 ms
Spatial SBF Logic (ms)   : 0.11 ms
MobileNetV2 CNN (ms)     : 1.62 ms
--------------------------------------------------
Total Pipeline Latency   : 28.72 ms
Estimated Real-Time FPS  : 34.8 FPS

==================================================
🎯 DYNAMIC SEQUENCE ACCURACY (Clip-Level)
==================================================
True Positives (Hit)     : 32
False Positives (Miss)   : 0
True Negatives (Correct) : 7
False Negatives (Miss)   : 2

We got the exact same score.

True Positives: 32

False Positives: 0

True Negatives: 7

False Negatives: 2

Your professor's challenge ("What if the CNN is eliminated?") just yielded a profound result: on your Standard Activities of Daily Living (ADL) dataset, the CNN provides zero additional benefit. The geometric rules alone correctly rejected all 7 background clips (walking, typing, etc.) and caught the exact same 32 handshakes.

Why did the 2 False Negatives still happen?
Look at your autopsy for clip_10 and clip_41:

Max Fused Avg Conf : 0.600 (Needs >0.5)

Max Stability Score : 0.940 (Needs >0.6)

Wait a minute! If the fused confidence was 0.600, and the stability was 0.940, both of those are above your thresholds! So why did it register as a False Negative?

The reason is the temporal averaging window (the prediction_queue). While the maximum instantaneous score hit 0.600, your script calculates an avg_conf over the last 10 frames. Because the handshake motion was too fast, or the hand jittered in and out of the perfect geometric angle, the rolling average over those 10 frames never quite crossed the 0.50 threshold.


3. what if we run **without CNN** on adversary dataset?
python benchmarking_script_v3.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
I0000 00:00:1775304577.564247 3223129 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1775304577.581654 3223130 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1775304577.596580 3223130 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
ERROR: Please create the folder structure '../images/p1_dataset_adversarial/handshake' and '../images/p1_dataset_adversarial/none'.
(.venv) nagashayanaramamurthy@Nagas-MacBook-Pro update_model % clear
(.venv) nagashayanaramamurthy@Nagas-MacBook-Pro update_model % python benchmarking_script_v3.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
I0000 00:00:1775304655.089443 3224525 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1775304655.106357 3224527 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1775304655.121265 3224527 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
==================================================
🚀 STARTING NESTED BATCH BENCHMARK ON MAC (WITH AUTOPSY)...
==================================================
Processing handshake/clip_11 (90 frames)...
W0000 00:00:1775304655.426333 3224527 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_11
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.091
   -> Palm Tilt (Needs 45-135): 0.6 to 179.2
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.988 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.395 (Needs >0.5)

Processing handshake/clip_12 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_12
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.061
   -> Palm Tilt (Needs 45-135): 0.0 to 179.2
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.957 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.383 (Needs >0.5)

Processing handshake/clip_21 (90 frames)...
Processing handshake/clip_22 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_22
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.093
   -> Palm Tilt (Needs 45-135): 38.2 to 178.9
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.960 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.384 (Needs >0.5)

Processing handshake/clip_25 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_25
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.307
   -> Palm Tilt (Needs 45-135): 59.4 to 93.3
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.531 (Needs >0.6)
   -> Max CNN Raw Score     : 0.210
   -> Max Fused Avg Conf    : 0.803 (Needs >0.5)

Processing handshake/clip_38 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_38
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.373
   -> Palm Tilt (Needs 45-135): 22.6 to 176.8
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.586 (Needs >0.6)
   -> Max CNN Raw Score     : 0.780
   -> Max Fused Avg Conf    : 0.834 (Needs >0.5)

Processing handshake/clip_39 (90 frames)...
Processing handshake/clip_40 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_40
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.037
   -> Palm Tilt (Needs 45-135): 25.7 to 179.8
   -> Thumb Ever Open?      : False
   -> Max Stability Score   : 0.974 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.390 (Needs >0.5)

Processing none/clip_21 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_21
   -> Max CNN Raw Score : 0.353
   -> Max Fused Avg Conf: 0.969 (Crossed 0.5)
Processing none/clip_22 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_22
   -> Max CNN Raw Score : 0.289
   -> Max Fused Avg Conf: 0.967 (Crossed 0.5)
Processing none/clip_24 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_24
   -> Max CNN Raw Score : 0.384
   -> Max Fused Avg Conf: 0.950 (Crossed 0.5)
Processing none/clip_25 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_25
   -> Max CNN Raw Score : 0.222
   -> Max Fused Avg Conf: 0.921 (Crossed 0.5)
Processing none/clip_26 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_26
   -> Max CNN Raw Score : 0.236
   -> Max Fused Avg Conf: 0.976 (Crossed 0.5)
Processing none/clip_27 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_27
   -> Max CNN Raw Score : 0.349
   -> Max Fused Avg Conf: 0.977 (Crossed 0.5)
Processing none/clip_30 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_30
   -> Max CNN Raw Score : 0.595
   -> Max Fused Avg Conf: 0.966 (Crossed 0.5)
Processing none/clip_31 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_31
   -> Max CNN Raw Score : 0.797
   -> Max Fused Avg Conf: 0.983 (Crossed 0.5)
Processing none/clip_34 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_34
   -> Max CNN Raw Score : 0.507
   -> Max Fused Avg Conf: 0.984 (Crossed 0.5)
Processing none/clip_36 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_36
   -> Max CNN Raw Score : 0.736
   -> Max Fused Avg Conf: 0.600 (Crossed 0.5)
Processing none/clip_37 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_37
   -> Max CNN Raw Score : 0.752
   -> Max Fused Avg Conf: 0.976 (Crossed 0.5)
Processing none/clip_38 (89 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_38
   -> Max CNN Raw Score : 0.845
   -> Max Fused Avg Conf: 0.957 (Crossed 0.5)
Processing none/clip_40 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_40
   -> Max CNN Raw Score : 0.605
   -> Max Fused Avg Conf: 0.981 (Crossed 0.5)

==================================================
📊 FINAL HARDWARE INFERENCE LATENCY (Averaged)
==================================================
Total Frames Benchmarked : 1884
MediaPipe Tracking (ms)  : 25.83 ms
Spatial SBF Logic (ms)   : 0.09 ms
MobileNetV2 CNN (ms)     : 1.17 ms
--------------------------------------------------
Total Pipeline Latency   : 27.09 ms
Estimated Real-Time FPS  : 36.9 FPS

==================================================
🎯 DYNAMIC SEQUENCE ACCURACY (Clip-Level)
==================================================
True Positives (Hit)     : 2
False Positives (Miss)   : 13
True Negatives (Correct) : 0
False Negatives (Miss)   : 6

4. Lets try with CNN same old weight now

            fused_pred = (0.4 * cnn_raw) + (0.4 * pose_score) + (0.2 * stability_score)


python benchmarking_script_v3.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
I0000 00:00:1775304925.915471 3229260 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1775304925.932446 3229262 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1775304925.947478 3229262 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
==================================================
🚀 STARTING NESTED BATCH BENCHMARK ON MAC (WITH AUTOPSY)...
==================================================
Processing handshake/clip_11 (90 frames)...
W0000 00:00:1775304926.258078 3229266 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_11
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.091
   -> Palm Tilt (Needs 45-135): 0.6 to 179.2
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.988 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.198 (Needs >0.5)

Processing handshake/clip_12 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_12
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.061
   -> Palm Tilt (Needs 45-135): 0.0 to 179.2
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.957 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.191 (Needs >0.5)

Processing handshake/clip_21 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_21
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.060
   -> Palm Tilt (Needs 45-135): 2.6 to 178.6
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.804 (Needs >0.6)
   -> Max CNN Raw Score     : 0.186
   -> Max Fused Avg Conf    : 0.447 (Needs >0.5)

Processing handshake/clip_22 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_22
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.093
   -> Palm Tilt (Needs 45-135): 38.2 to 178.9
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.960 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.192 (Needs >0.5)

Processing handshake/clip_25 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_25
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.307
   -> Palm Tilt (Needs 45-135): 59.4 to 93.3
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.531 (Needs >0.6)
   -> Max CNN Raw Score     : 0.210
   -> Max Fused Avg Conf    : 0.581 (Needs >0.5)

Processing handshake/clip_38 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_38
   -> SBF Gate Ever Opened? : True
   -> Max Reach (Needs >0.05): 0.373
   -> Palm Tilt (Needs 45-135): 22.6 to 176.8
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.586 (Needs >0.6)
   -> Max CNN Raw Score     : 0.780
   -> Max Fused Avg Conf    : 0.730 (Needs >0.5)

Processing handshake/clip_39 (90 frames)...
Processing handshake/clip_40 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_40
   -> SBF Gate Ever Opened? : False
   -> Max Reach (Needs >0.05): 0.037
   -> Palm Tilt (Needs 45-135): 25.7 to 179.8
   -> Thumb Ever Open?      : False
   -> Max Stability Score   : 0.974 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.195 (Needs >0.5)

Processing none/clip_21 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_21
   -> Max CNN Raw Score : 0.353
   -> Max Fused Avg Conf: 0.718 (Crossed 0.5)
Processing none/clip_22 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_22
   -> Max CNN Raw Score : 0.289
   -> Max Fused Avg Conf: 0.657 (Crossed 0.5)
Processing none/clip_24 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_24
   -> Max CNN Raw Score : 0.384
   -> Max Fused Avg Conf: 0.660 (Crossed 0.5)
Processing none/clip_25 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_25
   -> Max CNN Raw Score : 0.222
   -> Max Fused Avg Conf: 0.606 (Crossed 0.5)
Processing none/clip_26 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_26
   -> Max CNN Raw Score : 0.236
   -> Max Fused Avg Conf: 0.624 (Crossed 0.5)
Processing none/clip_27 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_27
   -> Max CNN Raw Score : 0.349
   -> Max Fused Avg Conf: 0.671 (Crossed 0.5)
Processing none/clip_30 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_30
   -> Max CNN Raw Score : 0.595
   -> Max Fused Avg Conf: 0.812 (Crossed 0.5)
Processing none/clip_31 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_31
   -> Max CNN Raw Score : 0.797
   -> Max Fused Avg Conf: 0.731 (Crossed 0.5)
Processing none/clip_34 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_34
   -> Max CNN Raw Score : 0.507
   -> Max Fused Avg Conf: 0.728 (Crossed 0.5)
Processing none/clip_36 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_36
   -> Max CNN Raw Score : 0.736
   -> Max Fused Avg Conf: 0.654 (Crossed 0.5)
Processing none/clip_37 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_37
   -> Max CNN Raw Score : 0.752
   -> Max Fused Avg Conf: 0.812 (Crossed 0.5)
Processing none/clip_38 (89 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_38
   -> Max CNN Raw Score : 0.845
   -> Max Fused Avg Conf: 0.863 (Crossed 0.5)
Processing none/clip_40 (90 frames)...

⚠️ FALSE POSITIVE DETECTED: none/clip_40
   -> Max CNN Raw Score : 0.605
   -> Max Fused Avg Conf: 0.645 (Crossed 0.5)

==================================================
📊 FINAL HARDWARE INFERENCE LATENCY (Averaged)
==================================================
Total Frames Benchmarked : 1884
MediaPipe Tracking (ms)  : 25.08 ms
Spatial SBF Logic (ms)   : 0.10 ms
MobileNetV2 CNN (ms)     : 1.17 ms
--------------------------------------------------
Total Pipeline Latency   : 26.35 ms
Estimated Real-Time FPS  : 38.0 FPS

==================================================
🎯 DYNAMIC SEQUENCE ACCURACY (Clip-Level)
==================================================
True Positives (Hit)     : 1
False Positives (Miss)   : 13
True Negatives (Correct) : 0
False Negatives (Miss)   : 7

so the difference is negligible. CNN is not adding any value for now.

GM Ashwin,
You are right, CNN currently not adding any value. 

standard dataset consists of normal handshakes by 2 people in diff in height, clothes, light settings, backgrounds, 1 person in a frame at a time.
none clips consists of high fives, hellos etc

adversary dataset (complicated env than standard dataset)  consists of handshakes by 2 people in diff in height, clothes, light settings, backgrounds, multiple hands, multiple people with multiple hands
none clips consists of near pattern of handshakes like holding a cup, showing numbers in hand - basically trying to satisfy geometry with hand gestures but not handshake

Adversarial Clip,Without CNN (CNN Raw),Without CNN (Fusion),With CNN (CNN Raw),With CNN (Fusion),Outcome Shift
Valid Handshakes,,,,,
clip_11,0.000,0.395 (FN),0.000,0.198 (FN),Missed in both
clip_12,0.000,0.383 (FN),0.000,0.191 (FN),Missed in both
clip_21,(Not Logged),> 0.500 (Hit/TP),0.186,0.447 (FN),❌ Worse with CNN
clip_22,0.000,0.384 (FN),0.000,0.192 (FN),Missed in both
clip_25,0.210,0.803 (FN)*,0.210,0.581 (FN)*,Missed in both
clip_38,0.780,0.834 (FN)*,0.780,0.730 (FN)*,Missed in both
clip_39,(Not Logged),> 0.500 (Hit/TP),(Not Logged),> 0.500 (Hit/TP),Perfect in both
clip_40,0.000,0.390 (FN),0.000,0.195 (FN),Missed in both
,,,,,
"Negative Gestures (High-Fives, Waves, Objects)",,,,,
none/clip_21,0.353,0.969 (FP),0.353,0.718 (FP),False Positive in both
none/clip_22,0.289,0.967 (FP),0.289,0.657 (FP),False Positive in both
none/clip_24,0.384,0.950 (FP),0.384,0.660 (FP),False Positive in both
none/clip_25,0.222,0.921 (FP),0.222,0.606 (FP),False Positive in both
none/clip_26,0.236,0.976 (FP),0.236,0.624 (FP),False Positive in both
none/clip_27,0.349,0.977 (FP),0.349,0.671 (FP),False Positive in both
none/clip_30,0.595,0.966 (FP),0.595,0.812 (FP),False Positive in both
none/clip_31,0.797,0.983 (FP),0.797,0.731 (FP),False Positive in both
none/clip_34,0.507,0.984 (FP),0.507,0.728 (FP),False Positive in both
none/clip_36,0.736,0.600 (FP),0.736,0.654 (FP),False Positive in both
none/clip_37,0.752,0.976 (FP),0.752,0.812 (FP),False Positive in both
none/clip_38,0.845,0.957 (FP),0.845,0.863 (FP),False Positive in both
none/clip_40,0.605,0.981 (FP),0.605,0.645 (FP),False Positive in both


--- if we reduce stability from 0.6 to 0.5


INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
==================================================
🚀 STARTING NESTED BATCH BENCHMARK ON MAC (WITH AUTOPSY)...
📁 DATASET: p1_dataset_adversaries
==================================================
Processing handshake/clip_11 (90 frames)...
W0000 00:00:1775418480.554574 3521337 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_11
   -> SBF Gate Ever Opened? : False
   -> Handedness Tracked    : Right, Left (Passed SBF: None)
   -> Wrist Y-Axis Range    : 0.451 to 0.704 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.091
   -> Palm Tilt (Needs 45-135): 0.6 to 179.2
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.988 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.395 (Needs >0.5)

Processing handshake/clip_12 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_12
   -> SBF Gate Ever Opened? : False
   -> Handedness Tracked    : Right, Left (Passed SBF: None)
   -> Wrist Y-Axis Range    : 0.441 to 0.686 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.061
   -> Palm Tilt (Needs 45-135): 0.0 to 179.2
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.957 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.383 (Needs >0.5)

Processing handshake/clip_21 (90 frames)...

✅ TRUE POSITIVE AUTOPSY: handshake/clip_21
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Left)
   -> Wrist Y-Axis Range    : 0.423 to 0.776 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.060
   -> Palm Tilt (Needs 45-135): 2.6 to 178.6
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.804 (Needs >0.6)
   -> Max CNN Raw Score     : 0.186
   -> Max Fused Avg Conf    : 0.600 (Needs >0.5)

Processing handshake/clip_22 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_22
   -> SBF Gate Ever Opened? : False
   -> Handedness Tracked    : Right, Left (Passed SBF: None)
   -> Wrist Y-Axis Range    : 0.584 to 0.916 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.093
   -> Palm Tilt (Needs 45-135): 38.2 to 178.9
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.960 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.384 (Needs >0.5)

Processing handshake/clip_25 (90 frames)...

✅ TRUE POSITIVE AUTOPSY: handshake/clip_25
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Left (Passed SBF: Left)
   -> Wrist Y-Axis Range    : 0.571 to 0.952 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.307
   -> Palm Tilt (Needs 45-135): 59.4 to 93.3
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.531 (Needs >0.6)
   -> Max CNN Raw Score     : 0.210
   -> Max Fused Avg Conf    : 0.803 (Needs >0.5)

Processing handshake/clip_38 (90 frames)...

✅ TRUE POSITIVE AUTOPSY: handshake/clip_38
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Right)
   -> Wrist Y-Axis Range    : 0.458 to 0.891 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.373
   -> Palm Tilt (Needs 45-135): 22.6 to 176.8
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.586 (Needs >0.6)
   -> Max CNN Raw Score     : 0.780
   -> Max Fused Avg Conf    : 0.834 (Needs >0.5)

Processing handshake/clip_39 (90 frames)...

✅ TRUE POSITIVE AUTOPSY: handshake/clip_39
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Right)
   -> Wrist Y-Axis Range    : 0.405 to 0.808 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.055
   -> Palm Tilt (Needs 45-135): 0.1 to 176.9
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.985 (Needs >0.6)
   -> Max CNN Raw Score     : 0.325
   -> Max Fused Avg Conf    : 0.887 (Needs >0.5)

Processing handshake/clip_40 (90 frames)...

❌ FALSE NEGATIVE AUTOPSY: handshake/clip_40
   -> SBF Gate Ever Opened? : False
   -> Handedness Tracked    : Right, Left (Passed SBF: None)
   -> Wrist Y-Axis Range    : 0.441 to 0.784 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.037
   -> Palm Tilt (Needs 45-135): 25.7 to 179.8
   -> Thumb Ever Open?      : False
   -> Max Stability Score   : 0.974 (Needs >0.6)
   -> Max CNN Raw Score     : 0.000
   -> Max Fused Avg Conf    : 0.390 (Needs >0.5)

Processing none/clip_21 (90 frames)...

⚠️ FALSE POSITIVE AUTOPSY: none/clip_21
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Right)
   -> Wrist Y-Axis Range    : 0.348 to 0.706 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.112
   -> Palm Tilt (Needs 45-135): 39.7 to 167.1
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.923 (Needs >0.6)
   -> Max CNN Raw Score     : 0.353
   -> Max Fused Avg Conf    : 0.969 (Needs >0.5)

Processing none/clip_22 (90 frames)...

⚠️ FALSE POSITIVE AUTOPSY: none/clip_22
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Right)
   -> Wrist Y-Axis Range    : 0.241 to 0.706 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.136
   -> Palm Tilt (Needs 45-135): 39.7 to 110.3
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.917 (Needs >0.6)
   -> Max CNN Raw Score     : 0.289
   -> Max Fused Avg Conf    : 0.967 (Needs >0.5)

Processing none/clip_24 (90 frames)...

⚠️ FALSE POSITIVE AUTOPSY: none/clip_24
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Right)
   -> Wrist Y-Axis Range    : 0.309 to 0.692 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.319
   -> Palm Tilt (Needs 45-135): 59.8 to 178.8
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.876 (Needs >0.6)
   -> Max CNN Raw Score     : 0.384
   -> Max Fused Avg Conf    : 0.950 (Needs >0.5)

Processing none/clip_25 (90 frames)...

⚠️ FALSE POSITIVE AUTOPSY: none/clip_25
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Right)
   -> Wrist Y-Axis Range    : 0.304 to 0.685 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.209
   -> Palm Tilt (Needs 45-135): 9.9 to 171.3
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.904 (Needs >0.6)
   -> Max CNN Raw Score     : 0.222
   -> Max Fused Avg Conf    : 0.921 (Needs >0.5)

Processing none/clip_26 (90 frames)...

⚠️ FALSE POSITIVE AUTOPSY: none/clip_26
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Left)
   -> Wrist Y-Axis Range    : 0.389 to 0.713 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.118
   -> Palm Tilt (Needs 45-135): 3.7 to 134.9
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.940 (Needs >0.6)
   -> Max CNN Raw Score     : 0.236
   -> Max Fused Avg Conf    : 0.976 (Needs >0.5)

Processing none/clip_27 (90 frames)...

⚠️ FALSE POSITIVE AUTOPSY: none/clip_27
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Left)
   -> Wrist Y-Axis Range    : 0.320 to 0.615 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.133
   -> Palm Tilt (Needs 45-135): 36.7 to 137.8
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.942 (Needs >0.6)
   -> Max CNN Raw Score     : 0.349
   -> Max Fused Avg Conf    : 0.977 (Needs >0.5)

Processing none/clip_30 (90 frames)...

⚠️ FALSE POSITIVE AUTOPSY: none/clip_30
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Left)
   -> Wrist Y-Axis Range    : 0.351 to 0.706 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.297
   -> Palm Tilt (Needs 45-135): 9.6 to 172.1
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.916 (Needs >0.6)
   -> Max CNN Raw Score     : 0.595
   -> Max Fused Avg Conf    : 0.966 (Needs >0.5)

Processing none/clip_31 (90 frames)...

⚠️ FALSE POSITIVE AUTOPSY: none/clip_31
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Right)
   -> Wrist Y-Axis Range    : 0.574 to 0.741 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.162
   -> Palm Tilt (Needs 45-135): 88.0 to 129.5
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.958 (Needs >0.6)
   -> Max CNN Raw Score     : 0.797
   -> Max Fused Avg Conf    : 0.983 (Needs >0.5)

Processing none/clip_34 (90 frames)...

⚠️ FALSE POSITIVE AUTOPSY: none/clip_34
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Right)
   -> Wrist Y-Axis Range    : 0.315 to 0.644 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.186
   -> Palm Tilt (Needs 45-135): 8.1 to 156.4
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.959 (Needs >0.6)
   -> Max CNN Raw Score     : 0.507
   -> Max Fused Avg Conf    : 0.984 (Needs >0.5)

Processing none/clip_36 (90 frames)...

⚠️ FALSE POSITIVE AUTOPSY: none/clip_36
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Left (Passed SBF: Left)
   -> Wrist Y-Axis Range    : 0.452 to 0.864 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.055
   -> Palm Tilt (Needs 45-135): 13.3 to 70.4
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.969 (Needs >0.6)
   -> Max CNN Raw Score     : 0.736
   -> Max Fused Avg Conf    : 0.600 (Needs >0.5)

Processing none/clip_37 (90 frames)...

⚠️ FALSE POSITIVE AUTOPSY: none/clip_37
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Right)
   -> Wrist Y-Axis Range    : 0.371 to 0.751 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.245
   -> Palm Tilt (Needs 45-135): 51.6 to 140.4
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.941 (Needs >0.6)
   -> Max CNN Raw Score     : 0.752
   -> Max Fused Avg Conf    : 0.976 (Needs >0.5)

Processing none/clip_38 (89 frames)...

⚠️ FALSE POSITIVE AUTOPSY: none/clip_38
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Left)
   -> Wrist Y-Axis Range    : 0.449 to 0.670 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.174
   -> Palm Tilt (Needs 45-135): 2.8 to 176.3
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.893 (Needs >0.6)
   -> Max CNN Raw Score     : 0.845
   -> Max Fused Avg Conf    : 0.957 (Needs >0.5)

Processing none/clip_40 (90 frames)...

⚠️ FALSE POSITIVE AUTOPSY: none/clip_40
   -> SBF Gate Ever Opened? : True
   -> Handedness Tracked    : Right, Left (Passed SBF: Left)
   -> Wrist Y-Axis Range    : 0.176 to 0.855 (0=Top, 1=Bottom)
   -> Max Reach (Needs >0.05): 0.119
   -> Palm Tilt (Needs 45-135): 75.1 to 171.3
   -> Thumb Ever Open?      : True
   -> Max Stability Score   : 0.951 (Needs >0.6)
   -> Max CNN Raw Score     : 0.605
   -> Max Fused Avg Conf    : 0.981 (Needs >0.5)


==================================================
📊 FINAL HARDWARE INFERENCE LATENCY (Averaged)
==================================================
Total Frames Benchmarked : 1884
MediaPipe Tracking (ms)  : 27.37 ms
Spatial SBF Logic (ms)   : 0.10 ms
MobileNetV2 CNN (ms)     : 1.19 ms
--------------------------------------------------
Total Pipeline Latency   : 28.66 ms
Estimated Real-Time FPS  : 34.9 FPS

==================================================
🎯 DYNAMIC SEQUENCE ACCURACY (Clip-Level)
==================================================
True Positives (Hit)     : 4
False Positives (Miss)   : 13
True Negatives (Correct) : 0
False Negatives (Miss)   : 4

--- should we consider y axis?

so my dataset is like this, I have 626 images of handshake(306) and diff action(326) which are non handshakes and then sequence dataset of 41 videos frames extracted (90 per video approx) handshake 42 and non handshake 20.



previously I used 626 static images to train cnn and then sequence dataset to evaluated cnn. when cnn failed badly on 8 handshakes and 13 non handshakes I called them adversary dataset

Go ahead and run the train_decision_tree.py script I provided earlier! Make sure the BENCHMARK_DIR in the script points exactly to that adversarial dataset folder.

Paste the output text of the tree here. It’s going to literally print out the if/else rules it learned, and we can see exactly what math it used to solve the high-five problem!


4. what if I use latest mediapipe and try to tune geometry to see both hands and pick one near by? can that solve adversal dataset problem?

we picked one hand using geometry and fed the data to decision tree to see the conditions it might generate


 python train_decision_tree.py 
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🌳 EXTRACTING FEATURES FOR DECISION TREE...
==================================================
I0000 00:00:1775474679.163484 3605747 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1775474679.181911 3605749 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1775474679.198008 3605754 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1775474679.253089 3605752 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Extraction Complete! Extracted 855 valid hand frames.

==================================================
🤖 DECISION TREE RULES LEARNED:
==================================================
|--- Thumb_Distance <= 0.06
|   |--- Reach_Z <= 0.09
|   |   |--- Reach_Z <= 0.02
|   |   |   |--- Reach_Z <= -0.01
|   |   |   |   |--- class: 0
|   |   |   |--- Reach_Z >  -0.01
|   |   |   |   |--- class: 1
|   |   |--- Reach_Z >  0.02
|   |   |   |--- Palm_Tilt <= 100.56
|   |   |   |   |--- class: 0
|   |   |   |--- Palm_Tilt >  100.56
|   |   |   |   |--- class: 1
|   |--- Reach_Z >  0.09
|   |   |--- Palm_Tilt <= 95.27
|   |   |   |--- Thumb_Distance <= 0.03
|   |   |   |   |--- class: 0
|   |   |   |--- Thumb_Distance >  0.03
|   |   |   |   |--- class: 0
|   |   |--- Palm_Tilt >  95.27
|   |   |   |--- Reach_Z <= 0.15
|   |   |   |   |--- class: 0
|   |   |   |--- Reach_Z >  0.15
|   |   |   |   |--- class: 1
|--- Thumb_Distance >  0.06
|   |--- Palm_Tilt <= 40.20
|   |   |--- Reach_Z <= 0.04
|   |   |   |--- class: 1
|   |   |--- Reach_Z >  0.04
|   |   |   |--- Palm_Tilt <= 25.90
|   |   |   |   |--- class: 0
|   |   |   |--- Palm_Tilt >  25.90
|   |   |   |   |--- class: 1
|   |--- Palm_Tilt >  40.20
|   |   |--- Reach_Z <= 0.25
|   |   |   |--- Wrist_Altitude_Y <= 0.68
|   |   |   |   |--- class: 0
|   |   |   |--- Wrist_Altitude_Y >  0.68
|   |   |   |   |--- class: 0
|   |   |--- Reach_Z >  0.25
|   |   |   |--- Palm_Tilt <= 122.38
|   |   |   |   |--- class: 1
|   |   |   |--- Palm_Tilt >  122.38
|   |   |   |   |--- class: 0


==================================================
📊 ML ACCURACY ON INDIVIDUAL FRAMES:
==================================================
              precision    recall  f1-score   support

No Handshake       0.89      0.97      0.93       503
   Handshake       0.95      0.83      0.89       352

    accuracy                           0.91       855
   macro avg       0.92      0.90      0.91       855
weighted avg       0.92      0.91      0.91       855



so without knowing we are mirroing the google mediapipe gesture recognizer architecture

- Take 21 raw coordinates and feed to classfication model

what we are doing
- Take 4 raw coordinates and feed to decision tree.

lets analyze the results of decision tree trained on static images + adversary dataset

 python train_decision_tree_combined.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🌳 EXTRACTING FEATURES FROM COMBINED DATASETS...
==================================================
I0000 00:00:1775475966.408308 3621337 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1775475966.424234 3621340 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1775475966.438363 3621344 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
Scanning Static Images in ../images/train_dataset_v2...
W0000 00:00:1775475966.626993 3621340 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Scanning Adversary Sequences in ../images/p1_dataset_adversaries...

Extraction Complete! Total valid hand frames: 1244

==================================================
🤖 DECISION TREE RULES LEARNED:
==================================================
|--- Reach_Z <= 0.02
|   |--- Wrist_Altitude_Y <= 0.44
|   |   |--- Palm_Tilt <= 80.74
|   |   |   |--- Reach_Z <= 0.02
|   |   |   |   |--- class: 0
|   |   |   |--- Reach_Z >  0.02
|   |   |   |   |--- class: 0
|   |   |--- Palm_Tilt >  80.74
|   |   |   |--- Thumb_Distance <= 0.05
|   |   |   |   |--- class: 0
|   |   |   |--- Thumb_Distance >  0.05
|   |   |   |   |--- class: 1
|   |--- Wrist_Altitude_Y >  0.44
|   |   |--- Thumb_Distance <= 0.09
|   |   |   |--- Reach_Z <= -0.01
|   |   |   |   |--- class: 0
|   |   |   |--- Reach_Z >  -0.01
|   |   |   |   |--- class: 1
|   |   |--- Thumb_Distance >  0.09
|   |   |   |--- Wrist_Altitude_Y <= 0.48
|   |   |   |   |--- class: 1
|   |   |   |--- Wrist_Altitude_Y >  0.48
|   |   |   |   |--- class: 0
|--- Reach_Z >  0.02
|   |--- Reach_Z <= 0.25
|   |   |--- Thumb_Distance <= 0.07
|   |   |   |--- Palm_Tilt <= 95.18
|   |   |   |   |--- class: 0
|   |   |   |--- Palm_Tilt >  95.18
|   |   |   |   |--- class: 1
|   |   |--- Thumb_Distance >  0.07
|   |   |   |--- Wrist_Altitude_Y <= 0.45
|   |   |   |   |--- class: 0
|   |   |   |--- Wrist_Altitude_Y >  0.45
|   |   |   |   |--- class: 0
|   |--- Reach_Z >  0.25
|   |   |--- Palm_Tilt <= 143.55
|   |   |   |--- Palm_Tilt <= 45.19
|   |   |   |   |--- class: 0
|   |   |   |--- Palm_Tilt >  45.19
|   |   |   |   |--- class: 1
|   |   |--- Palm_Tilt >  143.55
|   |   |   |--- Thumb_Distance <= 0.16
|   |   |   |   |--- class: 0
|   |   |   |--- Thumb_Distance >  0.16
|   |   |   |   |--- class: 1


==================================================
📊 ML ACCURACY ON COMBINED DATASET:
==================================================
              precision    recall  f1-score   support

No Handshake       0.79      0.87      0.83       698
   Handshake       0.81      0.71      0.75       546

    accuracy                           0.80      1244
   macro avg       0.80      0.79      0.79      1244
weighted avg       0.80      0.80      0.80      1244

then lets try both feeding 4 raw coordinates and 21 coordinates to classificaation model

but google did action recognition on static image but we are doing on sequential datset, here few frames
can make major difference so it's much harder problem?

nobody attempted it so far?

There are few things ppl do, usually it's 3rd person camera only meta ego4d or egoblind dataset is close?
ppl used classically these things

Dynamic Time Warping (DTW): This algorithm measures the similarity between two temporal sequences that might vary in speed. If one person shakes hands fast and another shakes hands slow, DTW stretches the time axis to see if the core "shape" of the movement matches.

Hidden Markov Models (HMMs): Researchers used HMMs to calculate the probability of moving from one state to another (e.g., the probability that an "arm raised" state will be followed by an "arm dropping" state).

2. The Neural Network Era (What Google and Meta do)
When you ask how the big tech companies solve sequential video datasets, they use massive architectures that look at both space (X/Y pixels) and time (Z sequences) simultaneously:

LSTMs (Long Short-Term Memory): Researchers take the exact 3D skeletal coordinates you are extracting (Reach, Tilt, Altitude) and feed them into an LSTM. LSTMs are networks with "memory." They remember that 5 frames ago the hand was up high, so when the hand drops low, the LSTM knows it's a wave, not a handshake.

Two-Stream CNNs: This is a famous architecture. Stream 1 looks at the static RGB image. Stream 2 looks at "Optical Flow" (a heat map of which direction pixels are moving).

3D CNNs (like I3D): Instead of a 2D image (Height × Width), they feed the network a 3D block of video (Time × Height × Width). The CNN learns the physical shape of motion.

The Ultimate Hybrid Pipeline (Your Final System)
This is what your final system should be:

The SBF Gate (Geometry): We use very relaxed, broad rules (e.g., Reach > 0.05, Tilt between 15-165). It will let the high-fives pass through, but it guarantees it will never miss a real handshake. Its only job is to wake the system up.

The Temporal Latch: We wait for the hand to be stable for a few frames.

The Custom CNN: We train a tiny, 3-layer Convolutional Neural Network from scratch on your 626 static images. Because it only runs when the SBF gate opens, it saves massive battery life. And because it looks at texture/pixels instead of geometry, it will instantly realize a high-five looks different than a handshake.

train on static image:

python train_custom_cnn.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🧠 TRAINING CUSTOM 3-LAYER CNN FOR EDGE DEVICES
==================================================
Loading dataset...
Found 626 files belonging to 2 classes.
Using 501 files for training.
Found 626 files belonging to 2 classes.
Using 125 files for validation.
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ sequential (Sequential)              │ (None, 160, 160, 1)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ rescaling (Rescaling)                │ (None, 160, 160, 1)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d (Conv2D)                      │ (None, 160, 160, 16)        │             160 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 80, 80, 16)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 80, 80, 32)          │           4,640 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 40, 40, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 40, 40, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (MaxPooling2D)       │ (None, 20, 20, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 20, 20, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 25600)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 64)                  │       1,638,464 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 1)                   │              65 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 1,661,825 (6.34 MB)
 Trainable params: 1,661,825 (6.34 MB)
 Non-trainable params: 0 (0.00 B)

🚀 Starting Training...
Epoch 1/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 3s 114ms/step - accuracy: 0.4824 - loss: 0.8555 - val_accuracy: 0.4560 - val_loss: 0.6959
Epoch 2/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 109ms/step - accuracy: 0.5027 - loss: 0.6894 - val_accuracy: 0.4560 - val_loss: 0.6927
Epoch 3/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 113ms/step - accuracy: 0.5167 - loss: 0.6859 - val_accuracy: 0.4560 - val_loss: 0.6874
Epoch 4/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 125ms/step - accuracy: 0.5121 - loss: 0.6849 - val_accuracy: 0.4560 - val_loss: 0.6913
Epoch 5/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 4s 228ms/step - accuracy: 0.5621 - loss: 0.6518 - val_accuracy: 0.6640 - val_loss: 0.6622
Epoch 6/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 3s 147ms/step - accuracy: 0.6390 - loss: 0.6607 - val_accuracy: 0.6960 - val_loss: 0.6491
Epoch 7/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 109ms/step - accuracy: 0.6560 - loss: 0.6627 - val_accuracy: 0.6400 - val_loss: 0.6509
Epoch 8/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 106ms/step - accuracy: 0.6428 - loss: 0.6386 - val_accuracy: 0.6960 - val_loss: 0.6268
Epoch 9/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 113ms/step - accuracy: 0.7176 - loss: 0.6163 - val_accuracy: 0.7360 - val_loss: 0.6176
Epoch 10/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 100ms/step - accuracy: 0.7004 - loss: 0.6228 - val_accuracy: 0.6880 - val_loss: 0.6231
Epoch 11/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 1s 93ms/step - accuracy: 0.7004 - loss: 0.5914 - val_accuracy: 0.7360 - val_loss: 0.6225
Epoch 12/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 102ms/step - accuracy: 0.6947 - loss: 0.6116 - val_accuracy: 0.7360 - val_loss: 0.6030
Epoch 13/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 106ms/step - accuracy: 0.6971 - loss: 0.6117 - val_accuracy: 0.6960 - val_loss: 0.6002
Epoch 14/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 98ms/step - accuracy: 0.7592 - loss: 0.5937 - val_accuracy: 0.7200 - val_loss: 0.5954
Epoch 15/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 1s 85ms/step - accuracy: 0.7204 - loss: 0.5933 - val_accuracy: 0.7680 - val_loss: 0.5856

==================================================
📦 EXPORTING TFLITE MODEL...
==================================================
Saved artifact at '/var/folders/vm/s5xwrxd93_xbnhp70qvyw03w0000gq/T/tmpogfe9abl'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 160, 160, 1), dtype=tf.float32, name='keras_tensor')
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  5891714736: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5891713680: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5891746096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5891745920: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5891747680: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5891777808: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5891778864: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5891778688: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5891780448: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5891780096: TensorSpec(shape=(), dtype=tf.resource, name=None)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1775477132.990602 3633733 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1775477132.990630 3633733 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
I0000 00:00:1775477132.994946 3633733 mlir_graph_optimization_pass.cc:437] MLIR V1 optimization pass is not enabled
✅ Success! Custom edge model saved as: custom_handshake_cnn.tflite
File Size: 1630.30 KB

"Why did you bother building the SBF Geometry gate at all? Just run the CNN!" But look at the data we have gathered today:

Pure Geometry (SBF): Maxes out at ~80% because it gets confused by the downward sweep of a high-five.

Pure Texture (Custom CNN): Maxes out at ~77% because a tiny 3-layer network on a small dataset can still get confused by weird lighting or background shapes.

The Conclusion: Neither system can survive the real world alone. The only way to achieve state-of-the-art accuracy on an edge device is to fuse them together. The SBF catches the 3D intent, and the CNN verifies the 2D texture.

Feeding mediapipe data to custom CNN

 python benchmark_custom_cnn.py 
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🚀 INITIALIZING HYBRID GATED PIPELINE (IEEE FINAL)
==================================================
I0000 00:00:1775477911.567189 3645860 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1775477911.578589 3645863 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1775477911.600687 3645863 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Evaluating Category: NONE
W0000 00:00:1775477911.711684 3645862 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT] Clip: clip_28 | Triggered: False | Max CNN Score: 0.531
[✅ CORRECT] Clip: clip_29 | Triggered: False | Max CNN Score: 0.536
[✅ CORRECT] Clip: clip_33 | Triggered: False | Max CNN Score: 0.258
[✅ CORRECT] Clip: clip_35 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_32 | Triggered: False | Max CNN Score: 0.213
[✅ CORRECT] Clip: clip_23 | Triggered: False | Max CNN Score: 0.536
[✅ CORRECT] Clip: clip_39 | Triggered: False | Max CNN Score: 0.236

Evaluating Category: HANDSHAKE
[❌ FAIL] Clip: clip_10 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_28 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_17 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_19 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_26 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_18 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_27 | Triggered: False | Max CNN Score: 0.378
[❌ FAIL] Clip: clip_20 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_29 | Triggered: False | Max CNN Score: 0.522
[❌ FAIL] Clip: clip_16 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_4 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_3 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_2 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_5 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_33 | Triggered: False | Max CNN Score: 0.208
[❌ FAIL] Clip: clip_34 | Triggered: False | Max CNN Score: 0.183
[❌ FAIL] Clip: clip_35 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_32 | Triggered: False | Max CNN Score: 0.518
[❌ FAIL] Clip: clip_14 | Triggered: False | Max CNN Score: 0.531
[❌ FAIL] Clip: clip_13 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_23 | Triggered: False | Max CNN Score: 0.514
[❌ FAIL] Clip: clip_24 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_15 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_41 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_9 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_0 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_7 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_6 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_1 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_8 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_37 | Triggered: False | Max CNN Score: 0.413
[❌ FAIL] Clip: clip_30 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_31 | Triggered: False | Max CNN Score: 0.536
[❌ FAIL] Clip: clip_36 | Triggered: False | Max CNN Score: 0.173

==================================================
🏁 BENCHMARK COMPLETE
==================================================

we removed the tflite optimization and then retrained the custom cnn

python train_custom_cnn.py    
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🧠 TRAINING CUSTOM CNN V2 (BATCH NORMALIZED)
==================================================
Loading dataset...
Found 613 files belonging to 2 classes.
Using 491 files for training.
Found 613 files belonging to 2 classes.
Using 122 files for validation.

🚀 Starting Training...
Epoch 1/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 9s 204ms/step - accuracy: 0.5771 - loss: 0.8200 - val_accuracy: 0.5164 - val_loss: 0.6793
Epoch 2/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 121ms/step - accuracy: 0.7057 - loss: 0.5486 - val_accuracy: 0.5164 - val_loss: 0.6817
Epoch 3/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 118ms/step - accuracy: 0.7416 - loss: 0.5446 - val_accuracy: 0.6230 - val_loss: 0.6613
Epoch 4/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 115ms/step - accuracy: 0.7854 - loss: 0.4876 - val_accuracy: 0.6311 - val_loss: 0.6671
Epoch 5/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 116ms/step - accuracy: 0.7970 - loss: 0.4390 - val_accuracy: 0.6066 - val_loss: 0.6674
Epoch 6/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 114ms/step - accuracy: 0.8245 - loss: 0.3962 - val_accuracy: 0.5656 - val_loss: 0.6804
Epoch 7/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 115ms/step - accuracy: 0.8208 - loss: 0.4013 - val_accuracy: 0.6148 - val_loss: 0.6782
Epoch 8/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 115ms/step - accuracy: 0.8329 - loss: 0.3917 - val_accuracy: 0.5984 - val_loss: 0.6825
Epoch 9/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 116ms/step - accuracy: 0.8234 - loss: 0.3832 - val_accuracy: 0.6803 - val_loss: 0.6295
Epoch 10/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 117ms/step - accuracy: 0.8081 - loss: 0.4260 - val_accuracy: 0.6230 - val_loss: 0.6532
Epoch 11/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 120ms/step - accuracy: 0.8787 - loss: 0.3234 - val_accuracy: 0.6230 - val_loss: 0.6762
Epoch 12/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 121ms/step - accuracy: 0.8569 - loss: 0.3526 - val_accuracy: 0.6230 - val_loss: 0.6908
Epoch 13/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 128ms/step - accuracy: 0.8704 - loss: 0.3144 - val_accuracy: 0.6393 - val_loss: 0.6799
Epoch 14/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 114ms/step - accuracy: 0.8505 - loss: 0.3436 - val_accuracy: 0.6230 - val_loss: 0.7585
Epoch 15/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 117ms/step - accuracy: 0.8910 - loss: 0.2998 - val_accuracy: 0.6721 - val_loss: 0.6767

==================================================
📦 EXPORTING PURE FLOAT32 TFLITE MODEL...
==================================================
Saved artifact at '/var/folders/vm/s5xwrxd93_xbnhp70qvyw03w0000gq/T/tmpeq30iab9'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 160, 160, 1), dtype=tf.float32, name='keras_tensor')
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:

I0000 00:00:1775562451.676897 4169949 mlir_graph_optimization_pass.cc:437] MLIR V1 optimization pass is not enabled
✅ Success! V2 Edge model saved as: custom_handshake_cnn.tflite
File Size: 6.34 MB

benchmarking result:

python benchmark_custom_cnn.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🚀 INITIALIZING HYBRID GATED PIPELINE (IEEE FINAL)
==================================================
I0000 00:00:1775562827.001536 4176241 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1775562827.016757 4176246 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1775562827.031465 4176247 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Evaluating Category: NONE
W0000 00:00:1775562827.106302 4176248 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[❌ FAIL] Clip: clip_28 | Triggered: True | Max CNN Score: 0.915
[❌ FAIL] Clip: clip_29 | Triggered: True | Max CNN Score: 0.751
[✅ CORRECT] Clip: clip_33 | Triggered: False | Max CNN Score: 0.203
[✅ CORRECT] Clip: clip_35 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_32 | Triggered: False | Max CNN Score: 0.268
[❌ FAIL] Clip: clip_23 | Triggered: True | Max CNN Score: 0.782
[❌ FAIL] Clip: clip_39 | Triggered: True | Max CNN Score: 0.633

Evaluating Category: HANDSHAKE
[✅ CORRECT] Clip: clip_10 | Triggered: True | Max CNN Score: 0.847
[✅ CORRECT] Clip: clip_28 | Triggered: True | Max CNN Score: 0.682
[✅ CORRECT] Clip: clip_17 | Triggered: True | Max CNN Score: 0.908
[✅ CORRECT] Clip: clip_19 | Triggered: True | Max CNN Score: 0.788
[✅ CORRECT] Clip: clip_26 | Triggered: True | Max CNN Score: 0.667
[✅ CORRECT] Clip: clip_18 | Triggered: True | Max CNN Score: 0.811
[✅ CORRECT] Clip: clip_27 | Triggered: True | Max CNN Score: 0.821
[✅ CORRECT] Clip: clip_20 | Triggered: True | Max CNN Score: 0.865
[✅ CORRECT] Clip: clip_29 | Triggered: True | Max CNN Score: 0.840
[✅ CORRECT] Clip: clip_16 | Triggered: True | Max CNN Score: 0.955
[✅ CORRECT] Clip: clip_4 | Triggered: True | Max CNN Score: 0.917
[✅ CORRECT] Clip: clip_3 | Triggered: True | Max CNN Score: 0.886
[✅ CORRECT] Clip: clip_2 | Triggered: True | Max CNN Score: 0.920
[✅ CORRECT] Clip: clip_5 | Triggered: True | Max CNN Score: 0.899
[✅ CORRECT] Clip: clip_33 | Triggered: True | Max CNN Score: 0.688
[✅ CORRECT] Clip: clip_34 | Triggered: True | Max CNN Score: 0.797
[✅ CORRECT] Clip: clip_35 | Triggered: True | Max CNN Score: 0.834
[❌ FAIL] Clip: clip_32 | Triggered: False | Max CNN Score: 0.352
[✅ CORRECT] Clip: clip_14 | Triggered: True | Max CNN Score: 0.925
[❌ FAIL] Clip: clip_13 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_23 | Triggered: False | Max CNN Score: 0.112
[✅ CORRECT] Clip: clip_24 | Triggered: True | Max CNN Score: 0.627
[✅ CORRECT] Clip: clip_15 | Triggered: True | Max CNN Score: 0.911
[❌ FAIL] Clip: clip_41 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_9 | Triggered: True | Max CNN Score: 0.851
[✅ CORRECT] Clip: clip_0 | Triggered: True | Max CNN Score: 0.850
[✅ CORRECT] Clip: clip_7 | Triggered: True | Max CNN Score: 0.892
[✅ CORRECT] Clip: clip_6 | Triggered: True | Max CNN Score: 0.925
[✅ CORRECT] Clip: clip_1 | Triggered: True | Max CNN Score: 0.870
[✅ CORRECT] Clip: clip_8 | Triggered: True | Max CNN Score: 0.883
[❌ FAIL] Clip: clip_37 | Triggered: False | Max CNN Score: 0.391
[✅ CORRECT] Clip: clip_30 | Triggered: True | Max CNN Score: 0.766
[✅ CORRECT] Clip: clip_31 | Triggered: True | Max CNN Score: 0.865
[✅ CORRECT] Clip: clip_36 | Triggered: True | Max CNN Score: 0.762

==================================================
🏁 BENCHMARK COMPLETE
==================================================

we see now, it is not just printing 0.536 as cnn score as previous result, looks more natural

analysis:
- bumping threshold of CNN to 0.80 will reduce 4 false positives but will also reduce 1 true positive
- currentlly the recall is 85%


lets try on adversarial dataset, then we can combine both adversarial and standard to see final recall rate

[❌ FAIL] Clip: clip_21 | Triggered: True | Max CNN Score: 0.725
[❌ FAIL] Clip: clip_26 | Triggered: True | Max CNN Score: 0.858
[❌ FAIL] Clip: clip_27 | Triggered: True | Max CNN Score: 0.866
[✅ CORRECT] Clip: clip_34 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_25 | Triggered: True | Max CNN Score: 0.812
[❌ FAIL] Clip: clip_22 | Triggered: True | Max CNN Score: 0.664
[✅ CORRECT] Clip: clip_40 | Triggered: False | Max CNN Score: 0.593
[❌ FAIL] Clip: clip_24 | Triggered: True | Max CNN Score: 0.655
[❌ FAIL] Clip: clip_37 | Triggered: True | Max CNN Score: 0.696
[❌ FAIL] Clip: clip_30 | Triggered: True | Max CNN Score: 0.739
[❌ FAIL] Clip: clip_38 | Triggered: True | Max CNN Score: 0.624
[✅ CORRECT] Clip: clip_31 | Triggered: False | Max CNN Score: 0.214
[❌ FAIL] Clip: clip_36 | Triggered: True | Max CNN Score: 0.801

Evaluating Category: HANDSHAKE
[❌ FAIL] Clip: clip_21 | Triggered: False | Max CNN Score: 0.081
[✅ CORRECT] Clip: clip_11 | Triggered: True | Max CNN Score: 0.906
[❌ FAIL] Clip: clip_25 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_22 | Triggered: False | Max CNN Score: 0.276
[❌ FAIL] Clip: clip_40 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_12 | Triggered: True | Max CNN Score: 0.924
[❌ FAIL] Clip: clip_39 | Triggered: False | Max CNN Score: 0.146
[❌ FAIL] Clip: clip_38 | Triggered: False | Max CNN Score: 0.342

==================================================
🏁 BENCHMARK COMPLETE
==================================================

let me train in on train_dataset_v2 instead of train_dataset_v2/unbiased because I see when we try to block the faces it is also blocking the hands/fingers sometimes like in salute
I moved unbiased from train_dataset_v2/unbiased to train_dataset_v2_unbiased (it contains black out faces)

==================================================
🧠 TRAINING CUSTOM CNN V2 (BATCH NORMALIZED)
==================================================
Loading dataset...
Found 613 files belonging to 2 classes.
Using 491 files for training.
Found 613 files belonging to 2 classes.
Using 122 files for validation.

🚀 Starting Training...
Epoch 1/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 10s 230ms/step - accuracy: 0.5641 - loss: 0.8838 - val_accuracy: 0.6393 - val_loss: 0.6422
Epoch 2/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 127ms/step - accuracy: 0.7463 - loss: 0.5579 - val_accuracy: 0.5902 - val_loss: 0.6757
Epoch 3/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 125ms/step - accuracy: 0.7706 - loss: 0.5172 - val_accuracy: 0.6475 - val_loss: 0.6512
Epoch 4/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 121ms/step - accuracy: 0.8190 - loss: 0.4372 - val_accuracy: 0.6967 - val_loss: 0.6331
Epoch 5/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 124ms/step - accuracy: 0.8149 - loss: 0.4524 - val_accuracy: 0.7213 - val_loss: 0.6152
Epoch 6/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 123ms/step - accuracy: 0.8410 - loss: 0.3823 - val_accuracy: 0.6967 - val_loss: 0.6043
Epoch 7/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 120ms/step - accuracy: 0.8142 - loss: 0.4229 - val_accuracy: 0.7049 - val_loss: 0.6147
Epoch 8/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 126ms/step - accuracy: 0.7896 - loss: 0.4357 - val_accuracy: 0.7541 - val_loss: 0.6066
Epoch 9/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 126ms/step - accuracy: 0.8466 - loss: 0.3798 - val_accuracy: 0.7295 - val_loss: 0.6091
Epoch 10/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 116ms/step - accuracy: 0.8623 - loss: 0.3646 - val_accuracy: 0.5656 - val_loss: 0.6485
Epoch 11/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 119ms/step - accuracy: 0.8364 - loss: 0.3602 - val_accuracy: 0.7377 - val_loss: 0.5820
Epoch 12/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 118ms/step - accuracy: 0.8375 - loss: 0.3716 - val_accuracy: 0.7705 - val_loss: 0.5433
Epoch 13/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 120ms/step - accuracy: 0.8699 - loss: 0.3333 - val_accuracy: 0.7049 - val_loss: 0.5415
Epoch 14/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 121ms/step - accuracy: 0.8309 - loss: 0.3497 - val_accuracy: 0.6721 - val_loss: 0.5546
Epoch 15/15
16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 123ms/step - accuracy: 0.8737 - loss: 0.3172 - val_accuracy: 0.6475 - val_loss: 0.6217

now lets benchmark on combined dataset (standard+adversary)

[❌ FAIL] Clip: clip_28 | Triggered: True | Max CNN Score: 0.889
[❌ FAIL] Clip: clip_21 | Triggered: True | Max CNN Score: 0.674
[❌ FAIL] Clip: clip_26 | Triggered: True | Max CNN Score: 0.905
[❌ FAIL] Clip: clip_27 | Triggered: True | Max CNN Score: 0.881
[❌ FAIL] Clip: clip_29 | Triggered: True | Max CNN Score: 0.833
[✅ CORRECT] Clip: clip_33 | Triggered: False | Max CNN Score: 0.560
[✅ CORRECT] Clip: clip_34 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_35 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_32 | Triggered: True | Max CNN Score: 0.693
[❌ FAIL] Clip: clip_25 | Triggered: True | Max CNN Score: 0.826
[❌ FAIL] Clip: clip_22 | Triggered: True | Max CNN Score: 0.657
[❌ FAIL] Clip: clip_40 | Triggered: True | Max CNN Score: 0.856
[❌ FAIL] Clip: clip_23 | Triggered: True | Max CNN Score: 0.627
[❌ FAIL] Clip: clip_24 | Triggered: True | Max CNN Score: 0.650
[❌ FAIL] Clip: clip_37 | Triggered: True | Max CNN Score: 0.923
[❌ FAIL] Clip: clip_30 | Triggered: True | Max CNN Score: 0.830
[❌ FAIL] Clip: clip_39 | Triggered: True | Max CNN Score: 0.836
[❌ FAIL] Clip: clip_38 | Triggered: True | Max CNN Score: 0.866
[❌ FAIL] Clip: clip_31 | Triggered: True | Max CNN Score: 0.695
[❌ FAIL] Clip: clip_36 | Triggered: True | Max CNN Score: 0.896

Evaluating Category: HANDSHAKE
[✅ CORRECT] Clip: clip_10 | Triggered: True | Max CNN Score: 0.867
[✅ CORRECT] Clip: clip_28 | Triggered: True | Max CNN Score: 0.861
[✅ CORRECT] Clip: clip_17 | Triggered: True | Max CNN Score: 0.921
[❌ FAIL] Clip: clip_21 | Triggered: False | Max CNN Score: 0.282
[✅ CORRECT] Clip: clip_19 | Triggered: True | Max CNN Score: 0.819
[✅ CORRECT] Clip: clip_26 | Triggered: True | Max CNN Score: 0.881
[✅ CORRECT] Clip: clip_18 | Triggered: True | Max CNN Score: 0.829
[✅ CORRECT] Clip: clip_27 | Triggered: True | Max CNN Score: 0.964
[✅ CORRECT] Clip: clip_20 | Triggered: True | Max CNN Score: 0.902
[✅ CORRECT] Clip: clip_29 | Triggered: True | Max CNN Score: 0.953
[✅ CORRECT] Clip: clip_16 | Triggered: True | Max CNN Score: 0.940
[✅ CORRECT] Clip: clip_11 | Triggered: True | Max CNN Score: 0.907
[✅ CORRECT] Clip: clip_4 | Triggered: True | Max CNN Score: 0.904
[✅ CORRECT] Clip: clip_3 | Triggered: True | Max CNN Score: 0.901
[✅ CORRECT] Clip: clip_2 | Triggered: True | Max CNN Score: 0.934
[✅ CORRECT] Clip: clip_5 | Triggered: True | Max CNN Score: 0.924
[✅ CORRECT] Clip: clip_33 | Triggered: True | Max CNN Score: 0.925
[✅ CORRECT] Clip: clip_34 | Triggered: True | Max CNN Score: 0.963
[✅ CORRECT] Clip: clip_35 | Triggered: True | Max CNN Score: 0.948
[✅ CORRECT] Clip: clip_32 | Triggered: True | Max CNN Score: 0.705
[✅ CORRECT] Clip: clip_14 | Triggered: True | Max CNN Score: 0.934
[❌ FAIL] Clip: clip_13 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_25 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_22 | Triggered: True | Max CNN Score: 0.657
[❌ FAIL] Clip: clip_40 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_23 | Triggered: True | Max CNN Score: 0.605
[✅ CORRECT] Clip: clip_24 | Triggered: True | Max CNN Score: 0.924
[✅ CORRECT] Clip: clip_12 | Triggered: True | Max CNN Score: 0.930
[✅ CORRECT] Clip: clip_15 | Triggered: True | Max CNN Score: 0.924
[❌ FAIL] Clip: clip_41 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_9 | Triggered: True | Max CNN Score: 0.883
[✅ CORRECT] Clip: clip_0 | Triggered: True | Max CNN Score: 0.897
[✅ CORRECT] Clip: clip_7 | Triggered: True | Max CNN Score: 0.901
[✅ CORRECT] Clip: clip_6 | Triggered: True | Max CNN Score: 0.907
[✅ CORRECT] Clip: clip_1 | Triggered: True | Max CNN Score: 0.899
[✅ CORRECT] Clip: clip_8 | Triggered: True | Max CNN Score: 0.898
[✅ CORRECT] Clip: clip_37 | Triggered: True | Max CNN Score: 0.762
[✅ CORRECT] Clip: clip_30 | Triggered: True | Max CNN Score: 0.866
[❌ FAIL] Clip: clip_39 | Triggered: False | Max CNN Score: 0.259
[✅ CORRECT] Clip: clip_38 | Triggered: True | Max CNN Score: 0.662
[✅ CORRECT] Clip: clip_31 | Triggered: True | Max CNN Score: 0.974
[✅ CORRECT] Clip: clip_36 | Triggered: True | Max CNN Score: 0.867

should we try cropping just hands to train and test? raw coordinates to CNN seems not a good idea, because CNN like grid of pixels and we tried this on decision tree and it kinda couldn't figure out the 
math.

algoirthm:
- we keep checking hands which satisfies our geometry and put them in left/right queues to track the stability once it passes then we crop and feed itto CNN
- the cropped dataaset train_dataset_v2_cropped is not good
- cropped dataset consist of only 194 and 195 images, where was original had 300+ in each. also it has cropped both left and right, I don't like this approach I think better is blacking out the face. if needed I can manually fix few problamatic ones

- Wrote this script to convert to gray scale first from original train_dataset_v2
- now we will convert train_dataset_v2_gray to face blockout. idea is to avoid some problems with train_dataset_v2_unbiased where some black patches were on hands and stuff
- the earlier dataset_prep.py used old cv algo called haar cascade, this time we will use mediapipe DL face detector (blazeface) and confident threshold is 70%.(build_masked_dataset.py)
at 79% confidence it masked only 45 faces.
at 40% confidence it masked 99 faces, still there are faces which are not blurred.

Benchmarking on masked dataset (masked by DL model)

 python train_custom_cnn.py    
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🧠 TRAINING CUSTOM CNN V2 (BATCH NORMALIZED)
==================================================
Loading dataset...
Found 545 files belonging to 2 classes.
Using 436 files for training.
Found 545 files belonging to 2 classes.
Using 109 files for validation.

🚀 Starting Training...
Epoch 1/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 10s 285ms/step - accuracy: 0.6226 - loss: 0.7145 - val_accuracy: 0.6789 - val_loss: 0.6460
Epoch 2/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 165ms/step - accuracy: 0.7861 - loss: 0.5150 - val_accuracy: 0.5321 - val_loss: 0.7139
Epoch 3/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 3s 202ms/step - accuracy: 0.8036 - loss: 0.4927 - val_accuracy: 0.6514 - val_loss: 0.6812
Epoch 4/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 136ms/step - accuracy: 0.8324 - loss: 0.4546 - val_accuracy: 0.5963 - val_loss: 0.6840
Epoch 5/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 119ms/step - accuracy: 0.8337 - loss: 0.3982 - val_accuracy: 0.6422 - val_loss: 0.6886
Epoch 6/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 120ms/step - accuracy: 0.8085 - loss: 0.4484 - val_accuracy: 0.5963 - val_loss: 0.6715
Epoch 7/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 125ms/step - accuracy: 0.8378 - loss: 0.3815 - val_accuracy: 0.6422 - val_loss: 0.6776
Epoch 8/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 118ms/step - accuracy: 0.8767 - loss: 0.3396 - val_accuracy: 0.6239 - val_loss: 0.6821
Epoch 9/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 120ms/step - accuracy: 0.8346 - loss: 0.3630 - val_accuracy: 0.6055 - val_loss: 0.6465
Epoch 10/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 122ms/step - accuracy: 0.8332 - loss: 0.3704 - val_accuracy: 0.6239 - val_loss: 0.7056
Epoch 11/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 127ms/step - accuracy: 0.8731 - loss: 0.3304 - val_accuracy: 0.5963 - val_loss: 0.6865
Epoch 12/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 121ms/step - accuracy: 0.8497 - loss: 0.3498 - val_accuracy: 0.6514 - val_loss: 0.6477
Epoch 13/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 121ms/step - accuracy: 0.8523 - loss: 0.3358 - val_accuracy: 0.6514 - val_loss: 0.6316
Epoch 14/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 122ms/step - accuracy: 0.8962 - loss: 0.2878 - val_accuracy: 0.6422 - val_loss: 0.6085
Epoch 15/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 129ms/step - accuracy: 0.8666 - loss: 0.3065 - val_accuracy: 0.6514 - val_loss: 0.6294

==================================================
📦 EXPORTING PURE FLOAT32 TFLITE MODEL...
==================================================
Saved artifact at '/var/folders/vm/s5xwrxd93_xbnhp70qvyw03w0000gq/T/tmptuj8dfl_'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 160, 160, 1), dtype=tf.float32, name='keras_tensor')
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  5889515184: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889514128: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889542624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889543680: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889541568: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889542448: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889586272: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889586096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889605872: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889606048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889587856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889588384: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889607632: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889607456: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889626528: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889627584: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889609392: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889626352: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889629520: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889629168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889648416: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889649472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889647536: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889648240: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889650528: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5889667312: TensorSpec(shape=(), dtype=tf.resource, name=None)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1776257929.387293 9112761 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1776257929.387318 9112761 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
I0000 00:00:1776257929.394033 9112761 mlir_graph_optimization_pass.cc:437] MLIR V1 optimization pass is not enabled
✅ Success! V2 Edge model saved as: custom_handshake_cnn.tflite
File Size: 6.34 MB
(.venv) nagashayanaramamurthy@Nagas-MacBook-Pro update_model % python benchmark_custom_cnn.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🚀 INITIALIZING HYBRID GATED PIPELINE (IEEE FINAL)
==================================================
I0000 00:00:1776258041.753026 9116959 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1776258041.767883 9116960 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1776258041.782759 9116965 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Evaluating Category: NONE
W0000 00:00:1776258041.887631 9116966 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[❌ FAIL] Clip: clip_28 | Triggered: True | Max CNN Score: 0.872
[❌ FAIL] Clip: clip_21 | Triggered: True | Max CNN Score: 0.755
[❌ FAIL] Clip: clip_26 | Triggered: True | Max CNN Score: 0.852
[❌ FAIL] Clip: clip_27 | Triggered: True | Max CNN Score: 0.859
[❌ FAIL] Clip: clip_29 | Triggered: True | Max CNN Score: 0.764
[❌ FAIL] Clip: clip_33 | Triggered: True | Max CNN Score: 0.621
[✅ CORRECT] Clip: clip_34 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_35 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_32 | Triggered: True | Max CNN Score: 0.733
[❌ FAIL] Clip: clip_25 | Triggered: True | Max CNN Score: 0.816
[✅ CORRECT] Clip: clip_22 | Triggered: False | Max CNN Score: 0.589
[❌ FAIL] Clip: clip_40 | Triggered: True | Max CNN Score: 0.702
[❌ FAIL] Clip: clip_23 | Triggered: True | Max CNN Score: 0.638
[✅ CORRECT] Clip: clip_24 | Triggered: False | Max CNN Score: 0.589
[❌ FAIL] Clip: clip_37 | Triggered: True | Max CNN Score: 0.972
[❌ FAIL] Clip: clip_30 | Triggered: True | Max CNN Score: 0.888
[❌ FAIL] Clip: clip_39 | Triggered: True | Max CNN Score: 0.791
[❌ FAIL] Clip: clip_38 | Triggered: True | Max CNN Score: 0.818
[❌ FAIL] Clip: clip_31 | Triggered: True | Max CNN Score: 0.639
[❌ FAIL] Clip: clip_36 | Triggered: True | Max CNN Score: 0.950

Evaluating Category: HANDSHAKE
[✅ CORRECT] Clip: clip_10 | Triggered: True | Max CNN Score: 0.822
[✅ CORRECT] Clip: clip_28 | Triggered: True | Max CNN Score: 0.765
[✅ CORRECT] Clip: clip_17 | Triggered: True | Max CNN Score: 0.901
[❌ FAIL] Clip: clip_21 | Triggered: False | Max CNN Score: 0.237
[✅ CORRECT] Clip: clip_19 | Triggered: True | Max CNN Score: 0.781
[✅ CORRECT] Clip: clip_26 | Triggered: True | Max CNN Score: 0.856
[✅ CORRECT] Clip: clip_18 | Triggered: True | Max CNN Score: 0.768
[✅ CORRECT] Clip: clip_27 | Triggered: True | Max CNN Score: 0.895
[✅ CORRECT] Clip: clip_20 | Triggered: True | Max CNN Score: 0.751
[✅ CORRECT] Clip: clip_29 | Triggered: True | Max CNN Score: 0.935
[✅ CORRECT] Clip: clip_16 | Triggered: True | Max CNN Score: 0.947
[✅ CORRECT] Clip: clip_11 | Triggered: True | Max CNN Score: 0.873
[✅ CORRECT] Clip: clip_4 | Triggered: True | Max CNN Score: 0.924
[✅ CORRECT] Clip: clip_3 | Triggered: True | Max CNN Score: 0.889
[✅ CORRECT] Clip: clip_2 | Triggered: True | Max CNN Score: 0.894
[✅ CORRECT] Clip: clip_5 | Triggered: True | Max CNN Score: 0.886
[✅ CORRECT] Clip: clip_33 | Triggered: True | Max CNN Score: 0.914
[✅ CORRECT] Clip: clip_34 | Triggered: True | Max CNN Score: 0.921
[✅ CORRECT] Clip: clip_35 | Triggered: True | Max CNN Score: 0.939
[✅ CORRECT] Clip: clip_32 | Triggered: True | Max CNN Score: 0.603
[✅ CORRECT] Clip: clip_14 | Triggered: True | Max CNN Score: 0.918
[❌ FAIL] Clip: clip_13 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_25 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_22 | Triggered: False | Max CNN Score: 0.470
[❌ FAIL] Clip: clip_40 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_23 | Triggered: False | Max CNN Score: 0.220
[✅ CORRECT] Clip: clip_24 | Triggered: True | Max CNN Score: 0.899
[✅ CORRECT] Clip: clip_12 | Triggered: True | Max CNN Score: 0.905
[✅ CORRECT] Clip: clip_15 | Triggered: True | Max CNN Score: 0.908
[❌ FAIL] Clip: clip_41 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_9 | Triggered: True | Max CNN Score: 0.909
[✅ CORRECT] Clip: clip_0 | Triggered: True | Max CNN Score: 0.892
[✅ CORRECT] Clip: clip_7 | Triggered: True | Max CNN Score: 0.912
[✅ CORRECT] Clip: clip_6 | Triggered: True | Max CNN Score: 0.910
[✅ CORRECT] Clip: clip_1 | Triggered: True | Max CNN Score: 0.877
[✅ CORRECT] Clip: clip_8 | Triggered: True | Max CNN Score: 0.841
[✅ CORRECT] Clip: clip_37 | Triggered: True | Max CNN Score: 0.738
[✅ CORRECT] Clip: clip_30 | Triggered: True | Max CNN Score: 0.882
[❌ FAIL] Clip: clip_39 | Triggered: False | Max CNN Score: 0.269
[✅ CORRECT] Clip: clip_38 | Triggered: True | Max CNN Score: 0.772
[✅ CORRECT] Clip: clip_31 | Triggered: True | Max CNN Score: 0.931
[✅ CORRECT] Clip: clip_36 | Triggered: True | Max CNN Score: 0.846

==================================================
🏁 BENCHMARK COMPLETE
==================================================
(.venv) nagashayanaramamurthy@Nagas-MacBook-Pro update_model %       

,Predicted: Handshake,Predicted: None (Adversary)
Actual: Handshake,34 (True Positive),8 (False Negative)
Actual: None (Adversary),16 (False Positive),4 (True Negative)

Lets try to run haar on DL masked dataset

instead of haar I ran mtccn which took a while and applied masking to 78% of the images (428 images) out of 545.
I won't do manual masking

training:

==================================================
🧠 TRAINING CUSTOM CNN V2 (BATCH NORMALIZED)
==================================================
Loading dataset...
Found 545 files belonging to 2 classes.
Using 436 files for training.
Found 545 files belonging to 2 classes.
Using 109 files for validation.

🚀 Starting Training...
Epoch 1/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 12s 329ms/step - accuracy: 0.6022 - loss: 0.7886 - val_accuracy: 0.6606 - val_loss: 0.6310
Epoch 2/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 127ms/step - accuracy: 0.7205 - loss: 0.5559 - val_accuracy: 0.5688 - val_loss: 0.7160
Epoch 3/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 124ms/step - accuracy: 0.7671 - loss: 0.5061 - val_accuracy: 0.5596 - val_loss: 0.7241
Epoch 4/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 134ms/step - accuracy: 0.8106 - loss: 0.4244 - val_accuracy: 0.5780 - val_loss: 0.7061
Epoch 5/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 134ms/step - accuracy: 0.8097 - loss: 0.4195 - val_accuracy: 0.5780 - val_loss: 0.7108
Epoch 6/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 139ms/step - accuracy: 0.8015 - loss: 0.4542 - val_accuracy: 0.5780 - val_loss: 0.7038
Epoch 7/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 136ms/step - accuracy: 0.8761 - loss: 0.3753 - val_accuracy: 0.6055 - val_loss: 0.7275
Epoch 8/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 117ms/step - accuracy: 0.8392 - loss: 0.3475 - val_accuracy: 0.5688 - val_loss: 0.7568
Epoch 9/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 131ms/step - accuracy: 0.8084 - loss: 0.4299 - val_accuracy: 0.5872 - val_loss: 0.7442
Epoch 10/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 123ms/step - accuracy: 0.8715 - loss: 0.3613 - val_accuracy: 0.5963 - val_loss: 0.7506
Epoch 11/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 121ms/step - accuracy: 0.8626 - loss: 0.3313 - val_accuracy: 0.5963 - val_loss: 0.7714
Epoch 12/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 124ms/step - accuracy: 0.8331 - loss: 0.3837 - val_accuracy: 0.5872 - val_loss: 0.7895
Epoch 13/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 118ms/step - accuracy: 0.8403 - loss: 0.4110 - val_accuracy: 0.5963 - val_loss: 0.7479
Epoch 14/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 116ms/step - accuracy: 0.8620 - loss: 0.3280 - val_accuracy: 0.6147 - val_loss: 0.7424
Epoch 15/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 118ms/step - accuracy: 0.8772 - loss: 0.3149 - val_accuracy: 0.6422 - val_loss: 0.7499

Evaluating Category: NONE
W0000 00:00:1777059744.851070 11955435 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[❌ FAIL] Clip: clip_28 | Triggered: True | Max CNN Score: 0.931
[❌ FAIL] Clip: clip_21 | Triggered: True | Max CNN Score: 0.790
[❌ FAIL] Clip: clip_26 | Triggered: True | Max CNN Score: 0.882
[❌ FAIL] Clip: clip_27 | Triggered: True | Max CNN Score: 0.889
[❌ FAIL] Clip: clip_29 | Triggered: True | Max CNN Score: 0.888
[✅ CORRECT] Clip: clip_33 | Triggered: False | Max CNN Score: 0.209
[✅ CORRECT] Clip: clip_34 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_35 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_32 | Triggered: False | Max CNN Score: 0.309
[❌ FAIL] Clip: clip_25 | Triggered: True | Max CNN Score: 0.869
[❌ FAIL] Clip: clip_22 | Triggered: True | Max CNN Score: 0.657
[✅ CORRECT] Clip: clip_40 | Triggered: False | Max CNN Score: 0.589
[❌ FAIL] Clip: clip_23 | Triggered: True | Max CNN Score: 0.715
[✅ CORRECT] Clip: clip_24 | Triggered: False | Max CNN Score: 0.581
[❌ FAIL] Clip: clip_37 | Triggered: True | Max CNN Score: 0.905
[❌ FAIL] Clip: clip_30 | Triggered: True | Max CNN Score: 0.612
[✅ CORRECT] Clip: clip_39 | Triggered: False | Max CNN Score: 0.588
[❌ FAIL] Clip: clip_38 | Triggered: True | Max CNN Score: 0.700
[✅ CORRECT] Clip: clip_31 | Triggered: False | Max CNN Score: 0.181
[❌ FAIL] Clip: clip_36 | Triggered: True | Max CNN Score: 0.847

Evaluating Category: HANDSHAKE
[✅ CORRECT] Clip: clip_10 | Triggered: True | Max CNN Score: 0.885
[✅ CORRECT] Clip: clip_28 | Triggered: True | Max CNN Score: 0.612
[✅ CORRECT] Clip: clip_17 | Triggered: True | Max CNN Score: 0.897
[❌ FAIL] Clip: clip_21 | Triggered: False | Max CNN Score: 0.122
[✅ CORRECT] Clip: clip_19 | Triggered: True | Max CNN Score: 0.714
[✅ CORRECT] Clip: clip_26 | Triggered: True | Max CNN Score: 0.663
[✅ CORRECT] Clip: clip_18 | Triggered: True | Max CNN Score: 0.734
[✅ CORRECT] Clip: clip_27 | Triggered: True | Max CNN Score: 0.605
[✅ CORRECT] Clip: clip_20 | Triggered: True | Max CNN Score: 0.791
[✅ CORRECT] Clip: clip_29 | Triggered: True | Max CNN Score: 0.816
[✅ CORRECT] Clip: clip_16 | Triggered: True | Max CNN Score: 0.907
[✅ CORRECT] Clip: clip_11 | Triggered: True | Max CNN Score: 0.860
[✅ CORRECT] Clip: clip_4 | Triggered: True | Max CNN Score: 0.948
[✅ CORRECT] Clip: clip_3 | Triggered: True | Max CNN Score: 0.874
[✅ CORRECT] Clip: clip_2 | Triggered: True | Max CNN Score: 0.915
[✅ CORRECT] Clip: clip_5 | Triggered: True | Max CNN Score: 0.899
[✅ CORRECT] Clip: clip_33 | Triggered: True | Max CNN Score: 0.621
[✅ CORRECT] Clip: clip_34 | Triggered: True | Max CNN Score: 0.715
[✅ CORRECT] Clip: clip_35 | Triggered: True | Max CNN Score: 0.733
[❌ FAIL] Clip: clip_32 | Triggered: False | Max CNN Score: 0.249
[✅ CORRECT] Clip: clip_14 | Triggered: True | Max CNN Score: 0.899
[❌ FAIL] Clip: clip_13 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_25 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_22 | Triggered: False | Max CNN Score: 0.414
[❌ FAIL] Clip: clip_40 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_23 | Triggered: False | Max CNN Score: 0.086
[❌ FAIL] Clip: clip_24 | Triggered: False | Max CNN Score: 0.553
[✅ CORRECT] Clip: clip_12 | Triggered: True | Max CNN Score: 0.911
[✅ CORRECT] Clip: clip_15 | Triggered: True | Max CNN Score: 0.903
[❌ FAIL] Clip: clip_41 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_9 | Triggered: True | Max CNN Score: 0.870
[✅ CORRECT] Clip: clip_0 | Triggered: True | Max CNN Score: 0.891
[✅ CORRECT] Clip: clip_7 | Triggered: True | Max CNN Score: 0.940
[✅ CORRECT] Clip: clip_6 | Triggered: True | Max CNN Score: 0.923
[✅ CORRECT] Clip: clip_1 | Triggered: True | Max CNN Score: 0.862
[✅ CORRECT] Clip: clip_8 | Triggered: True | Max CNN Score: 0.904
[❌ FAIL] Clip: clip_37 | Triggered: False | Max CNN Score: 0.294
[✅ CORRECT] Clip: clip_30 | Triggered: True | Max CNN Score: 0.745
[❌ FAIL] Clip: clip_39 | Triggered: False | Max CNN Score: 0.123
[❌ FAIL] Clip: clip_38 | Triggered: False | Max CNN Score: 0.331
[✅ CORRECT] Clip: clip_31 | Triggered: True | Max CNN Score: 0.779
[✅ CORRECT] Clip: clip_36 | Triggered: True | Max CNN Score: 0.604

==================================================
🏁 BENCHMARK COMPLETE

,Predicted: Handshake,Predicted: None (Adversary)
Actual: Handshake,30 (True Positive),12 (False Negative)
Actual: None (Adversary),12 (False Positive),8 (True Negative)

erformance Shift Analysis
Here are the key metrics and how they changed from your last run:

Overall Accuracy: 61.3% (Exactly the same as before! 38 out of 62 correct).

Precision (TP / TP + FP): 71.4% (Up from 68.0%).

The Victory: Because the model became stricter, it successfully rejected twice as many adversarial high-fives (True Negatives jumped from 4 to 8). When this model says "Handshake", you can trust it more.

Recall (TP / TP + FN): 71.4% (Down from 81.0%).

The Cost: That same strictness caused the model to hesitate on real handshakes. It missed 4 more real handshakes than the previous iteration.

lets compare this custom cnn to mobilenetv2 as base for combined dataset

[❌ FAIL] Clip: clip_28 | Triggered: True | Max CNN Score: 0.684
[❌ FAIL] Clip: clip_21 | Triggered: True | Max CNN Score: 0.801
[❌ FAIL] Clip: clip_26 | Triggered: True | Max CNN Score: 0.698
[❌ FAIL] Clip: clip_27 | Triggered: True | Max CNN Score: 0.616
[❌ FAIL] Clip: clip_29 | Triggered: True | Max CNN Score: 0.626
[❌ FAIL] Clip: clip_33 | Triggered: True | Max CNN Score: 0.729
[✅ CORRECT] Clip: clip_34 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_35 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_32 | Triggered: True | Max CNN Score: 0.719
[❌ FAIL] Clip: clip_25 | Triggered: True | Max CNN Score: 0.685
[❌ FAIL] Clip: clip_22 | Triggered: True | Max CNN Score: 0.766
[❌ FAIL] Clip: clip_40 | Triggered: True | Max CNN Score: 0.613
[❌ FAIL] Clip: clip_23 | Triggered: True | Max CNN Score: 0.820
[❌ FAIL] Clip: clip_24 | Triggered: True | Max CNN Score: 0.801
[❌ FAIL] Clip: clip_37 | Triggered: True | Max CNN Score: 0.694
[❌ FAIL] Clip: clip_30 | Triggered: True | Max CNN Score: 0.701
[✅ CORRECT] Clip: clip_39 | Triggered: False | Max CNN Score: 0.561
[❌ FAIL] Clip: clip_38 | Triggered: True | Max CNN Score: 0.627
[❌ FAIL] Clip: clip_31 | Triggered: True | Max CNN Score: 0.738
[❌ FAIL] Clip: clip_36 | Triggered: True | Max CNN Score: 0.681

Evaluating Category: HANDSHAKE
[✅ CORRECT] Clip: clip_10 | Triggered: True | Max CNN Score: 0.743
[❌ FAIL] Clip: clip_28 | Triggered: False | Max CNN Score: 0.546
[✅ CORRECT] Clip: clip_17 | Triggered: True | Max CNN Score: 0.819
[✅ CORRECT] Clip: clip_21 | Triggered: True | Max CNN Score: 0.723
[✅ CORRECT] Clip: clip_19 | Triggered: True | Max CNN Score: 0.742
[❌ FAIL] Clip: clip_26 | Triggered: False | Max CNN Score: 0.561
[✅ CORRECT] Clip: clip_18 | Triggered: True | Max CNN Score: 0.764
[❌ FAIL] Clip: clip_27 | Triggered: False | Max CNN Score: 0.485
[✅ CORRECT] Clip: clip_20 | Triggered: True | Max CNN Score: 0.821
[✅ CORRECT] Clip: clip_29 | Triggered: True | Max CNN Score: 0.638
[✅ CORRECT] Clip: clip_16 | Triggered: True | Max CNN Score: 0.767
[✅ CORRECT] Clip: clip_11 | Triggered: True | Max CNN Score: 0.736
[✅ CORRECT] Clip: clip_4 | Triggered: True | Max CNN Score: 0.639
[✅ CORRECT] Clip: clip_3 | Triggered: True | Max CNN Score: 0.828
[✅ CORRECT] Clip: clip_2 | Triggered: True | Max CNN Score: 0.853
[✅ CORRECT] Clip: clip_5 | Triggered: True | Max CNN Score: 0.804
[❌ FAIL] Clip: clip_33 | Triggered: False | Max CNN Score: 0.574
[❌ FAIL] Clip: clip_34 | Triggered: False | Max CNN Score: 0.510
[✅ CORRECT] Clip: clip_35 | Triggered: True | Max CNN Score: 0.644
[✅ CORRECT] Clip: clip_32 | Triggered: True | Max CNN Score: 0.717
[✅ CORRECT] Clip: clip_14 | Triggered: True | Max CNN Score: 0.786
[❌ FAIL] Clip: clip_13 | Triggered: False | Max CNN Score: 0.000
[❌ FAIL] Clip: clip_25 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_22 | Triggered: True | Max CNN Score: 0.727
[❌ FAIL] Clip: clip_40 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_23 | Triggered: True | Max CNN Score: 0.633
[❌ FAIL] Clip: clip_24 | Triggered: False | Max CNN Score: 0.543
[✅ CORRECT] Clip: clip_12 | Triggered: True | Max CNN Score: 0.657
[✅ CORRECT] Clip: clip_15 | Triggered: True | Max CNN Score: 0.797
[❌ FAIL] Clip: clip_41 | Triggered: False | Max CNN Score: 0.000
[✅ CORRECT] Clip: clip_9 | Triggered: True | Max CNN Score: 0.618
[✅ CORRECT] Clip: clip_0 | Triggered: True | Max CNN Score: 0.867
[✅ CORRECT] Clip: clip_7 | Triggered: True | Max CNN Score: 0.785
[✅ CORRECT] Clip: clip_6 | Triggered: True | Max CNN Score: 0.854
[✅ CORRECT] Clip: clip_1 | Triggered: True | Max CNN Score: 0.874
[✅ CORRECT] Clip: clip_8 | Triggered: True | Max CNN Score: 0.785
[✅ CORRECT] Clip: clip_37 | Triggered: True | Max CNN Score: 0.806
[❌ FAIL] Clip: clip_30 | Triggered: False | Max CNN Score: 0.307
[❌ FAIL] Clip: clip_39 | Triggered: False | Max CNN Score: 0.600
[✅ CORRECT] Clip: clip_38 | Triggered: True | Max CNN Score: 0.700
[❌ FAIL] Clip: clip_31 | Triggered: False | Max CNN Score: 0.466
[✅ CORRECT] Clip: clip_36 | Triggered: True | Max CNN Score: 0.667

==================================================
🏁 BENCHMARK COMPLETE
==================================================

,Predicted: Handshake,Predicted: None (Adversary)
Actual: Handshake,29 (True Positive),13 (False Negative)
Actual: None (Adversary),17 (False Positive),3 (True Negative)


Performance Metrics: Custom CNN vs. MobileNetV2
Here is the side-by-side comparison proving why your custom architecture wins:

Overall Accuracy: 51.6% (MobileNetV2) vs. 61.3% (Custom CNN)

MobileNetV2 is basically performing at random chance (coin-flip) accuracy.

Precision: 63.0% (MobileNetV2) vs. 71.4% (Custom CNN)

MobileNetV2 suffers from catastrophic false positives.

Recall: 69.0% (MobileNetV2) vs. 71.4% (Custom CNN)

IDEA: The extracted landmarks from mediapipe + image can be fed to CNN? instead of  just using the mediapipe landmarks as gatekeeper?

Decision tree on combined dataset:

python train_decision_tree.py 
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🌳 EXTRACTING FEATURES FOR DECISION TREE...
==================================================
I0000 00:00:1777718680.560456 13676115 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1777718680.571322 13676117 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777718680.591217 13676117 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777718680.630600 13676123 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Extraction Complete! Extracted 2847 valid hand frames.

==================================================
🤖 DECISION TREE RULES LEARNED:
==================================================
|--- Thumb_Distance <= 0.13
|   |--- Reach_Z <= -0.01
|   |   |--- Wrist_Altitude_Y <= 0.60
|   |   |   |--- class: 0
|   |   |--- Wrist_Altitude_Y >  0.60
|   |   |   |--- Wrist_Altitude_Y <= 0.83
|   |   |   |   |--- class: 1
|   |   |   |--- Wrist_Altitude_Y >  0.83
|   |   |   |   |--- class: 0
|   |--- Reach_Z >  -0.01
|   |   |--- Reach_Z <= 0.03
|   |   |   |--- Wrist_Altitude_Y <= 0.37
|   |   |   |   |--- class: 0
|   |   |   |--- Wrist_Altitude_Y >  0.37
|   |   |   |   |--- class: 1
|   |   |--- Reach_Z >  0.03
|   |   |   |--- Wrist_Altitude_Y <= 0.40
|   |   |   |   |--- class: 1
|   |   |   |--- Wrist_Altitude_Y >  0.40
|   |   |   |   |--- class: 1
|--- Thumb_Distance >  0.13
|   |--- Reach_Z <= 0.18
|   |   |--- Wrist_Altitude_Y <= 0.43
|   |   |   |--- Thumb_Distance <= 0.16
|   |   |   |   |--- class: 1
|   |   |   |--- Thumb_Distance >  0.16
|   |   |   |   |--- class: 0
|   |   |--- Wrist_Altitude_Y >  0.43
|   |   |   |--- Reach_Z <= 0.06
|   |   |   |   |--- class: 0
|   |   |   |--- Reach_Z >  0.06
|   |   |   |   |--- class: 0
|   |--- Reach_Z >  0.18
|   |   |--- Palm_Tilt <= 53.96
|   |   |   |--- class: 0
|   |   |--- Palm_Tilt >  53.96
|   |   |   |--- Palm_Tilt <= 140.80
|   |   |   |   |--- class: 1
|   |   |   |--- Palm_Tilt >  140.80
|   |   |   |   |--- class: 0


==================================================
📊 ML ACCURACY ON INDIVIDUAL FRAMES:
==================================================
              precision    recall  f1-score   support

No Handshake       0.80      0.48      0.60       815
   Handshake       0.82      0.95      0.88      2032

    accuracy                           0.82      2847
   macro avg       0.81      0.72      0.74      2847
weighted avg       0.81      0.82      0.80      2847

Randrom forest:
==================================================
🤖 TRAINING RANDOM FOREST...
==================================================

📊 ML ACCURACY ON TRAINING FRAMES:
              precision    recall  f1-score   support

No Handshake       0.87      0.86      0.87       195
   Handshake       0.86      0.87      0.87       194

    accuracy                           0.87       389
   macro avg       0.87      0.87      0.87       389
weighted avg       0.87      0.87      0.87       389


🔍 FEATURE IMPORTANCES:
 - Thumb_Distance: 23.10%
 - Reach_Z: 35.61%
 - Wrist_Altitude_Y: 13.86%
 - Palm_Tilt: 27.43%

✅ Success! Random Forest saved to: random_forest_model.pkl

python benchmark_random_forest.py 
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🌲 BENCHMARKING RANDOM FOREST (PURE GEOMETRY)
==================================================
I0000 00:00:1777809391.917274 13809698 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1777809391.927682 13809700 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777809391.941845 13809705 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

Evaluating Category: NONE
W0000 00:00:1777809392.899693 13809702 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[❌ FAIL - FP] Clip: clip_28
[❌ FAIL - FP] Clip: clip_21
[❌ FAIL - FP] Clip: clip_26
[❌ FAIL - FP] Clip: clip_27
[❌ FAIL - FP] Clip: clip_29
[❌ FAIL - FP] Clip: clip_33
[✅ CORRECT - TN] Clip: clip_34
[❌ FAIL - FP] Clip: clip_35
[✅ CORRECT - TN] Clip: clip_32
[❌ FAIL - FP] Clip: clip_25
[❌ FAIL - FP] Clip: clip_22
[❌ FAIL - FP] Clip: clip_40
[❌ FAIL - FP] Clip: clip_23
[❌ FAIL - FP] Clip: clip_24
[❌ FAIL - FP] Clip: clip_37
[❌ FAIL - FP] Clip: clip_30
[✅ CORRECT - TN] Clip: clip_39
[❌ FAIL - FP] Clip: clip_38
[❌ FAIL - FP] Clip: clip_31
[✅ CORRECT - TN] Clip: clip_36

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10
[✅ CORRECT - TP] Clip: clip_28
[❌ FAIL - FN] Clip: clip_17
[✅ CORRECT - TP] Clip: clip_21
[✅ CORRECT - TP] Clip: clip_19
[✅ CORRECT - TP] Clip: clip_26
[✅ CORRECT - TP] Clip: clip_18
[✅ CORRECT - TP] Clip: clip_27
[✅ CORRECT - TP] Clip: clip_20
[✅ CORRECT - TP] Clip: clip_29
[❌ FAIL - FN] Clip: clip_16
[✅ CORRECT - TP] Clip: clip_11
[✅ CORRECT - TP] Clip: clip_4
[✅ CORRECT - TP] Clip: clip_3
[✅ CORRECT - TP] Clip: clip_2
[✅ CORRECT - TP] Clip: clip_5
[✅ CORRECT - TP] Clip: clip_33
[✅ CORRECT - TP] Clip: clip_34
[✅ CORRECT - TP] Clip: clip_35
[✅ CORRECT - TP] Clip: clip_32
[✅ CORRECT - TP] Clip: clip_14
[❌ FAIL - FN] Clip: clip_13
[❌ FAIL - FN] Clip: clip_25
[✅ CORRECT - TP] Clip: clip_22
[❌ FAIL - FN] Clip: clip_40
[✅ CORRECT - TP] Clip: clip_23
[✅ CORRECT - TP] Clip: clip_24
[❌ FAIL - FN] Clip: clip_12
[❌ FAIL - FN] Clip: clip_15
[❌ FAIL - FN] Clip: clip_41
[✅ CORRECT - TP] Clip: clip_9
[✅ CORRECT - TP] Clip: clip_0
[✅ CORRECT - TP] Clip: clip_7
[✅ CORRECT - TP] Clip: clip_6
[✅ CORRECT - TP] Clip: clip_1
[❌ FAIL - FN] Clip: clip_8
[✅ CORRECT - TP] Clip: clip_37
[✅ CORRECT - TP] Clip: clip_30
[❌ FAIL - FN] Clip: clip_39
[✅ CORRECT - TP] Clip: clip_38
[✅ CORRECT - TP] Clip: clip_31
[✅ CORRECT - TP] Clip: clip_36

==================================================
🏁 RF BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  58.06%
Precision: 66.67%
Recall:    76.19%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3127
Avg Latency per Frame  : 22.04 ms
Estimated Real-Time FPS: 45.4 FPS

XGBoost:

python train_xgboost.py                   
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🚀 EXTRACTING FEATURES FOR XGBOOST...
==================================================
I0000 00:00:1777826531.829788 13986922 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1777826531.841518 13986926 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777826531.858275 13986924 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777826531.903781 13986923 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Extraction Complete! Extracted 389 valid hand frames.

==================================================
🤖 TRAINING XGBOOST...
==================================================
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [12:42:50] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:740: 
Parameters: { "use_label_encoder" } are not used.

  warnings.warn(smsg, UserWarning)

📊 ML ACCURACY ON TRAINING FRAMES:
              precision    recall  f1-score   support

No Handshake       0.98      0.99      0.99       195
   Handshake       0.99      0.98      0.99       194

    accuracy                           0.99       389
   macro avg       0.99      0.99      0.99       389
weighted avg       0.99      0.99      0.99       389


🔍 FEATURE IMPORTANCES:
 - Thumb_Distance: 21.91%
 - Reach_Z: 35.52%
 - Wrist_Altitude_Y: 18.67%
 - Palm_Tilt: 23.90%

✅ Success! XGBoost saved to: xgboost_model.pkl


python benchmark_xgboost.py 
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🚀 BENCHMARKING XGBOOST (PURE GEOMETRY)
==================================================
I0000 00:00:1777826772.844710 13991112 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1777826772.861160 13991113 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777826772.877240 13991117 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

Evaluating Category: NONE
W0000 00:00:1777826773.370676 13991120 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[❌ FAIL - FP] Clip: clip_28
[❌ FAIL - FP] Clip: clip_21
[✅ CORRECT - TN] Clip: clip_26
[❌ FAIL - FP] Clip: clip_27
[❌ FAIL - FP] Clip: clip_29
[❌ FAIL - FP] Clip: clip_33
[✅ CORRECT - TN] Clip: clip_34
[❌ FAIL - FP] Clip: clip_35
[❌ FAIL - FP] Clip: clip_32
[✅ CORRECT - TN] Clip: clip_25
[❌ FAIL - FP] Clip: clip_22
[❌ FAIL - FP] Clip: clip_40
[❌ FAIL - FP] Clip: clip_23
[❌ FAIL - FP] Clip: clip_24
[❌ FAIL - FP] Clip: clip_37
[❌ FAIL - FP] Clip: clip_30
[❌ FAIL - FP] Clip: clip_39
[❌ FAIL - FP] Clip: clip_38
[❌ FAIL - FP] Clip: clip_31
[❌ FAIL - FP] Clip: clip_36

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10
[✅ CORRECT - TP] Clip: clip_28
[❌ FAIL - FN] Clip: clip_17
[✅ CORRECT - TP] Clip: clip_21
[✅ CORRECT - TP] Clip: clip_19
[✅ CORRECT - TP] Clip: clip_26
[✅ CORRECT - TP] Clip: clip_18
[✅ CORRECT - TP] Clip: clip_27
[✅ CORRECT - TP] Clip: clip_20
[✅ CORRECT - TP] Clip: clip_29
[❌ FAIL - FN] Clip: clip_16
[✅ CORRECT - TP] Clip: clip_11
[✅ CORRECT - TP] Clip: clip_4
[✅ CORRECT - TP] Clip: clip_3
[✅ CORRECT - TP] Clip: clip_2
[✅ CORRECT - TP] Clip: clip_5
[✅ CORRECT - TP] Clip: clip_33
[✅ CORRECT - TP] Clip: clip_34
[✅ CORRECT - TP] Clip: clip_35
[✅ CORRECT - TP] Clip: clip_32
[✅ CORRECT - TP] Clip: clip_14
[❌ FAIL - FN] Clip: clip_13
[❌ FAIL - FN] Clip: clip_25
[✅ CORRECT - TP] Clip: clip_22
[✅ CORRECT - TP] Clip: clip_40
[✅ CORRECT - TP] Clip: clip_23
[✅ CORRECT - TP] Clip: clip_24
[✅ CORRECT - TP] Clip: clip_12
[❌ FAIL - FN] Clip: clip_15
[❌ FAIL - FN] Clip: clip_41
[✅ CORRECT - TP] Clip: clip_9
[✅ CORRECT - TP] Clip: clip_0
[✅ CORRECT - TP] Clip: clip_7
[✅ CORRECT - TP] Clip: clip_6
[✅ CORRECT - TP] Clip: clip_1
[✅ CORRECT - TP] Clip: clip_8
[✅ CORRECT - TP] Clip: clip_37
[✅ CORRECT - TP] Clip: clip_30
[✅ CORRECT - TP] Clip: clip_39
[✅ CORRECT - TP] Clip: clip_38
[✅ CORRECT - TP] Clip: clip_31
[✅ CORRECT - TP] Clip: clip_36

==================================================
🏁 XGBOOST BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  62.90%
Precision: 67.92%
Recall:    85.71%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 2682
Avg Latency per Frame  : 20.91 ms
Estimated Real-Time FPS: 47.8 FPS

python benchmark_custom_cnn.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🚀 BENCHMARKING HYBRID GATED PIPELINE (IEEE FINAL)
==================================================
I0000 00:00:1777828205.845857 14008550 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1777828205.857154 14008554 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777828205.870015 14008552 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Evaluating Category: NONE
W0000 00:00:1777828205.934842 14008551 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[❌ FAIL - FP] Clip: clip_28 | Score: 0.93
[❌ FAIL - FP] Clip: clip_21 | Score: 0.79
[❌ FAIL - FP] Clip: clip_26 | Score: 0.88
[❌ FAIL - FP] Clip: clip_27 | Score: 0.89
[❌ FAIL - FP] Clip: clip_29 | Score: 0.89
[✅ CORRECT - TN] Clip: clip_33 | Score: 0.21
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_32 | Score: 0.28
[❌ FAIL - FP] Clip: clip_25 | Score: 0.87
[❌ FAIL - FP] Clip: clip_22 | Score: 0.66
[✅ CORRECT - TN] Clip: clip_40 | Score: 0.59
[❌ FAIL - FP] Clip: clip_23 | Score: 0.72
[✅ CORRECT - TN] Clip: clip_24 | Score: 0.58
[❌ FAIL - FP] Clip: clip_37 | Score: 0.90
[❌ FAIL - FP] Clip: clip_30 | Score: 0.61
[✅ CORRECT - TN] Clip: clip_39 | Score: 0.59
[❌ FAIL - FP] Clip: clip_38 | Score: 0.70
[✅ CORRECT - TN] Clip: clip_31 | Score: 0.18
[❌ FAIL - FP] Clip: clip_36 | Score: 0.85

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.88
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.61
[❌ FAIL - FN] Clip: clip_17 | Score: 0.00
[❌ FAIL - FN] Clip: clip_21 | Score: 0.12
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.71
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.66
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.61
[❌ FAIL - FN] Clip: clip_20 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_29 | Score: 0.72
[❌ FAIL - FN] Clip: clip_16 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.86
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.95
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.87
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.91
[✅ CORRECT - TP] Clip: clip_5 | Score: 0.90
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.72
[✅ CORRECT - TP] Clip: clip_35 | Score: 0.73
[❌ FAIL - FN] Clip: clip_32 | Score: 0.25
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.90
[❌ FAIL - FN] Clip: clip_13 | Score: 0.00
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.41
[❌ FAIL - FN] Clip: clip_40 | Score: 0.00
[❌ FAIL - FN] Clip: clip_23 | Score: 0.09
[❌ FAIL - FN] Clip: clip_24 | Score: 0.55
[✅ CORRECT - TP] Clip: clip_12 | Score: 0.91
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.90
[❌ FAIL - FN] Clip: clip_41 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.87
[✅ CORRECT - TP] Clip: clip_0 | Score: 0.87
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.94
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.92
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.86
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.90
[❌ FAIL - FN] Clip: clip_37 | Score: 0.29
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.74
[❌ FAIL - FN] Clip: clip_39 | Score: 0.12
[❌ FAIL - FN] Clip: clip_38 | Score: 0.33
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.78
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.60

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  56.45%
Precision: 69.23%
Recall:    64.29%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3560
Avg Latency per Frame  : 20.75 ms
Estimated Real-Time FPS: 48.2 FPS


 python train_custom_cnn_dual_input.py 
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🧠 BUILDING DUAL-INPUT (FUSION) NEURAL NETWORK
==================================================
I0000 00:00:1777847723.950889 14053090 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1777847723.960672 14053091 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777847723.979379 14053091 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
Extracting Pixels + Geometry...
W0000 00:00:1777847724.064870 14053093 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Extraction Complete! 40 valid multimodal frames.

Compiling Dual-Branch CNN...

Training Dual-Input Model...
Epoch 1/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 1s 28ms/step - accuracy: 0.3578 - loss: 4.9602 
Epoch 2/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - accuracy: 0.6344 - loss: 2.4507
Epoch 3/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.6969 - loss: 2.0820
Epoch 4/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.6609 - loss: 1.2823
Epoch 5/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.6875 - loss: 0.6347
Epoch 6/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.5703 - loss: 0.6822
Epoch 7/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.7984 - loss: 0.4683
Epoch 8/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.7266 - loss: 0.4944
Epoch 9/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.7781 - loss: 0.4126
Epoch 10/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step - accuracy: 0.8344 - loss: 0.4707


python benchmark_dual_input.py 
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🚀 BENCHMARKING MULTIMODAL FUSION (PIXELS + MATH)
==================================================
I0000 00:00:1777847943.987442 14056277 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1777847944.001913 14056278 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777847944.015622 14056278 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Evaluating Category: NONE
W0000 00:00:1777847944.110033 14056281 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[❌ FAIL - FP] Clip: clip_28 | Score: 0.72
[❌ FAIL - FP] Clip: clip_21 | Score: 0.78
[❌ FAIL - FP] Clip: clip_26 | Score: 0.92
[❌ FAIL - FP] Clip: clip_27 | Score: 0.87
[❌ FAIL - FP] Clip: clip_29 | Score: 0.78
[❌ FAIL - FP] Clip: clip_33 | Score: 0.78
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[❌ FAIL - FP] Clip: clip_35 | Score: 0.97
[✅ CORRECT - TN] Clip: clip_32 | Score: 0.27
[❌ FAIL - FP] Clip: clip_25 | Score: 0.94
[❌ FAIL - FP] Clip: clip_22 | Score: 0.79
[❌ FAIL - FP] Clip: clip_40 | Score: 0.74
[❌ FAIL - FP] Clip: clip_23 | Score: 0.96
[❌ FAIL - FP] Clip: clip_24 | Score: 0.95
[❌ FAIL - FP] Clip: clip_37 | Score: 0.70
[❌ FAIL - FP] Clip: clip_30 | Score: 0.89
[❌ FAIL - FP] Clip: clip_39 | Score: 0.98
[❌ FAIL - FP] Clip: clip_38 | Score: 0.83
[❌ FAIL - FP] Clip: clip_31 | Score: 0.81
[❌ FAIL - FP] Clip: clip_36 | Score: 0.61

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.76
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.78
[✅ CORRECT - TP] Clip: clip_17 | Score: 0.86
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.82
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.84
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.70
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.81
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.81
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.89
[✅ CORRECT - TP] Clip: clip_29 | Score: 0.93
[✅ CORRECT - TP] Clip: clip_16 | Score: 0.87
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.71
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.94
[✅ CORRECT - TP] Clip: clip_5 | Score: 0.70
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.90
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.79
[✅ CORRECT - TP] Clip: clip_35 | Score: 0.82
[✅ CORRECT - TP] Clip: clip_32 | Score: 0.89
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.85
[✅ CORRECT - TP] Clip: clip_13 | Score: 0.85
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_22 | Score: 0.80
[✅ CORRECT - TP] Clip: clip_40 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_23 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.87
[✅ CORRECT - TP] Clip: clip_12 | Score: 0.95
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.88
[✅ CORRECT - TP] Clip: clip_41 | Score: 0.90
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.87
[✅ CORRECT - TP] Clip: clip_0 | Score: 0.85
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.88
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.85
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.67
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.96
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.83
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.87
[✅ CORRECT - TP] Clip: clip_39 | Score: 0.71
[✅ CORRECT - TP] Clip: clip_38 | Score: 0.78
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.76
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.70

==================================================
🏁 DUAL-INPUT BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  69.35%
Precision: 69.49%
Recall:    97.62%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 2146
Avg Latency per Frame  : 19.53 ms
Estimated Real-Time FPS: 51.2 FPS

removing thumb state as input on cnn+geometry improves accuracy

this is without thumb in CNN gated by geometry benchmark


==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  61.29%
Precision: 71.43%
Recall:    71.43%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3401
Avg Latency per Frame  : 20.70 ms
Estimated Real-Time FPS: 48.3 FPS

lets try same on dual input cnn

 python train_custom_cnn_dual_input.py 
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🧠 BUILDING DUAL-INPUT (FUSION) NEURAL NETWORK
==================================================
I0000 00:00:1777849453.348159 14073259 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1777849453.363187 14073260 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777849453.379453 14073260 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
Extracting Pixels + Geometry...
W0000 00:00:1777849453.485485 14073261 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Extraction Complete! 40 valid multimodal frames.

Compiling Dual-Branch CNN...

Training Dual-Input Model...
Epoch 1/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 1s 28ms/step - accuracy: 0.6469 - loss: 1.6721 
Epoch 2/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step - accuracy: 0.7812 - loss: 0.6356
Epoch 3/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.8141 - loss: 0.6703
Epoch 4/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.7875 - loss: 0.5697
Epoch 5/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.8188 - loss: 0.5415
Epoch 6/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.8062 - loss: 0.4780
Epoch 7/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.8781 - loss: 0.3689
Epoch 8/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - accuracy: 0.9281 - loss: 0.3217
Epoch 9/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - accuracy: 0.9062 - loss: 0.2832
Epoch 10/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - accuracy: 0.9344 - loss: 0.2110


python benchmark_dual_input.py 
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🚀 BENCHMARKING MULTIMODAL FUSION (PIXELS + MATH)
==================================================
I0000 00:00:1777849922.700066 14080648 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1777849922.714966 14080651 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777849922.729006 14080655 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Evaluating Category: NONE
W0000 00:00:1777849922.805745 14080654 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[❌ FAIL - FP] Clip: clip_28 | Score: 0.75
[❌ FAIL - FP] Clip: clip_21 | Score: 0.86
[❌ FAIL - FP] Clip: clip_26 | Score: 0.95
[❌ FAIL - FP] Clip: clip_27 | Score: 0.87
[❌ FAIL - FP] Clip: clip_29 | Score: 0.79
[❌ FAIL - FP] Clip: clip_33 | Score: 0.86
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[❌ FAIL - FP] Clip: clip_35 | Score: 0.97
[✅ CORRECT - TN] Clip: clip_32 | Score: 0.47
[❌ FAIL - FP] Clip: clip_25 | Score: 0.97
[❌ FAIL - FP] Clip: clip_22 | Score: 0.80
[❌ FAIL - FP] Clip: clip_40 | Score: 0.86
[❌ FAIL - FP] Clip: clip_23 | Score: 0.97
[❌ FAIL - FP] Clip: clip_24 | Score: 0.97
[❌ FAIL - FP] Clip: clip_37 | Score: 0.87
[❌ FAIL - FP] Clip: clip_30 | Score: 0.95
[❌ FAIL - FP] Clip: clip_39 | Score: 1.00
[❌ FAIL - FP] Clip: clip_38 | Score: 0.84
[❌ FAIL - FP] Clip: clip_31 | Score: 0.88
[❌ FAIL - FP] Clip: clip_36 | Score: 0.83

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.89
[✅ CORRECT - TP] Clip: clip_17 | Score: 0.90
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.84
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.89
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.92
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.87
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.92
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.95
[✅ CORRECT - TP] Clip: clip_29 | Score: 0.98
[✅ CORRECT - TP] Clip: clip_16 | Score: 0.91
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.72
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.66
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.70
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_5 | Score: 0.72
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.95
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.86
[✅ CORRECT - TP] Clip: clip_35 | Score: 0.94
[✅ CORRECT - TP] Clip: clip_32 | Score: 0.93
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.90
[✅ CORRECT - TP] Clip: clip_13 | Score: 0.89
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_22 | Score: 0.77
[✅ CORRECT - TP] Clip: clip_40 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_23 | Score: 0.66
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.91
[✅ CORRECT - TP] Clip: clip_12 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.93
[✅ CORRECT - TP] Clip: clip_41 | Score: 0.90
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.90
[✅ CORRECT - TP] Clip: clip_0 | Score: 0.86
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.92
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.89
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.70
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.99
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.90
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.91
[✅ CORRECT - TP] Clip: clip_39 | Score: 0.76
[✅ CORRECT - TP] Clip: clip_38 | Score: 0.92
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.81
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.91

==================================================
🏁 DUAL-INPUT BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  69.35%
Precision: 69.49%
Recall:    97.62%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 2104
Avg Latency per Frame  : 19.43 ms
Estimated Real-Time FPS: 51.5 FPS


Training on grey scale convereted mtcnn masked images
Benchmarking on RGB unedited frames

Isn't valid? because CNN shouldn't be biased but mediapipe needs rgb? 

Lets train cnn + geometry without masking and benchmark it

python train_custom_cnn.py 
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🧠 TRAINING CUSTOM CNN V2 (BATCH NORMALIZED)
==================================================
Loading dataset...
Found 545 files belonging to 2 classes.
Using 436 files for training.
Found 545 files belonging to 2 classes.
Using 109 files for validation.

🚀 Starting Training...
Epoch 1/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 6s 184ms/step - accuracy: 0.5481 - loss: 0.8790 - val_accuracy: 0.4862 - val_loss: 0.6914
Epoch 2/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 117ms/step - accuracy: 0.7451 - loss: 0.5603 - val_accuracy: 0.5963 - val_loss: 0.6520
Epoch 3/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 116ms/step - accuracy: 0.7404 - loss: 0.5655 - val_accuracy: 0.5413 - val_loss: 0.6834
Epoch 4/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 118ms/step - accuracy: 0.8289 - loss: 0.4229 - val_accuracy: 0.5596 - val_loss: 0.6930
Epoch 5/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 121ms/step - accuracy: 0.8163 - loss: 0.4168 - val_accuracy: 0.5596 - val_loss: 0.7271
Epoch 6/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 116ms/step - accuracy: 0.8091 - loss: 0.4147 - val_accuracy: 0.5505 - val_loss: 0.7319
Epoch 7/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 117ms/step - accuracy: 0.8030 - loss: 0.4539 - val_accuracy: 0.5505 - val_loss: 0.7482
Epoch 8/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 116ms/step - accuracy: 0.8640 - loss: 0.3459 - val_accuracy: 0.5596 - val_loss: 0.7175
Epoch 9/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 120ms/step - accuracy: 0.8268 - loss: 0.3928 - val_accuracy: 0.5596 - val_loss: 0.7674
Epoch 10/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 116ms/step - accuracy: 0.8462 - loss: 0.3890 - val_accuracy: 0.5780 - val_loss: 0.6755
Epoch 11/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 116ms/step - accuracy: 0.8265 - loss: 0.3803 - val_accuracy: 0.5688 - val_loss: 0.7946
Epoch 12/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 115ms/step - accuracy: 0.8075 - loss: 0.4036 - val_accuracy: 0.6055 - val_loss: 0.7406
Epoch 13/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 116ms/step - accuracy: 0.8905 - loss: 0.2916 - val_accuracy: 0.5963 - val_loss: 0.7658
Epoch 14/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 116ms/step - accuracy: 0.8477 - loss: 0.3519 - val_accuracy: 0.6055 - val_loss: 0.7143
Epoch 15/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 114ms/step - accuracy: 0.8825 - loss: 0.3278 - val_accuracy: 0.6055 - val_loss: 0.7904

python benchmark_custom_cnn.py 
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
==================================================
🚀 BENCHMARKING HYBRID GATED PIPELINE (IEEE FINAL)
==================================================
I0000 00:00:1777895615.701549 14152801 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1777895615.711139 14152802 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777895615.728290 14152802 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Evaluating Category: NONE
W0000 00:00:1777895615.817872 14152809 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT - TN] Clip: clip_28 | Score: 0.09
[✅ CORRECT - TN] Clip: clip_21 | Score: 0.18
[✅ CORRECT - TN] Clip: clip_26 | Score: 0.09
[✅ CORRECT - TN] Clip: clip_27 | Score: 0.09
[✅ CORRECT - TN] Clip: clip_29 | Score: 0.07
[✅ CORRECT - TN] Clip: clip_33 | Score: 0.18
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_32 | Score: 0.19
[✅ CORRECT - TN] Clip: clip_25 | Score: 0.08
[✅ CORRECT - TN] Clip: clip_22 | Score: 0.22
[✅ CORRECT - TN] Clip: clip_40 | Score: 0.27
[✅ CORRECT - TN] Clip: clip_23 | Score: 0.18
[✅ CORRECT - TN] Clip: clip_24 | Score: 0.18
[✅ CORRECT - TN] Clip: clip_37 | Score: 0.40
[✅ CORRECT - TN] Clip: clip_30 | Score: 0.37
[✅ CORRECT - TN] Clip: clip_39 | Score: 0.25
[✅ CORRECT - TN] Clip: clip_38 | Score: 0.49
[✅ CORRECT - TN] Clip: clip_31 | Score: 0.18
[✅ CORRECT - TN] Clip: clip_36 | Score: 0.41

Evaluating Category: HANDSHAKE
[❌ FAIL - FN] Clip: clip_10 | Score: 0.54
[❌ FAIL - FN] Clip: clip_28 | Score: 0.43
[❌ FAIL - FN] Clip: clip_17 | Score: 0.50
[❌ FAIL - FN] Clip: clip_21 | Score: 0.07
[❌ FAIL - FN] Clip: clip_19 | Score: 0.55
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.62
[❌ FAIL - FN] Clip: clip_18 | Score: 0.59
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.61
[❌ FAIL - FN] Clip: clip_20 | Score: 0.55
[❌ FAIL - FN] Clip: clip_29 | Score: 0.59
[❌ FAIL - FN] Clip: clip_16 | Score: 0.55
[❌ FAIL - FN] Clip: clip_11 | Score: 0.55
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.70
[❌ FAIL - FN] Clip: clip_3 | Score: 0.57
[❌ FAIL - FN] Clip: clip_2 | Score: 0.60
[❌ FAIL - FN] Clip: clip_5 | Score: 0.57
[❌ FAIL - FN] Clip: clip_33 | Score: 0.43
[❌ FAIL - FN] Clip: clip_34 | Score: 0.51
[❌ FAIL - FN] Clip: clip_35 | Score: 0.41
[❌ FAIL - FN] Clip: clip_32 | Score: 0.21
[❌ FAIL - FN] Clip: clip_14 | Score: 0.56
[❌ FAIL - FN] Clip: clip_13 | Score: 0.00
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.24
[❌ FAIL - FN] Clip: clip_40 | Score: 0.00
[❌ FAIL - FN] Clip: clip_23 | Score: 0.25
[❌ FAIL - FN] Clip: clip_24 | Score: 0.38
[❌ FAIL - FN] Clip: clip_12 | Score: 0.55
[❌ FAIL - FN] Clip: clip_15 | Score: 0.54
[❌ FAIL - FN] Clip: clip_41 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.64
[❌ FAIL - FN] Clip: clip_0 | Score: 0.58
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.66
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.67
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.63
[❌ FAIL - FN] Clip: clip_8 | Score: 0.53
[❌ FAIL - FN] Clip: clip_37 | Score: 0.24
[❌ FAIL - FN] Clip: clip_30 | Score: 0.39
[❌ FAIL - FN] Clip: clip_39 | Score: 0.06
[❌ FAIL - FN] Clip: clip_38 | Score: 0.19
[❌ FAIL - FN] Clip: clip_31 | Score: 0.44
[❌ FAIL - FN] Clip: clip_36 | Score: 0.47

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  43.55%
Precision: 100.00%
Recall:    16.67%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 5120
Avg Latency per Frame  : 22.44 ms
Estimated Real-Time FPS: 44.6 FPS

after reducing confidence to 50%

ython benchmark_custom_cnn.py
==================================================
🚀 BENCHMARKING HYBRID GATED PIPELINE (IEEE FINAL)
==================================================
I0000 00:00:1777898903.928089 14201235 init-domain.cc:128] Fiber init: default domain = pthread, concurrency = 8, prefix = pthread-default
I0000 00:00:1777898904.125255 14201235 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1777898904.134183 14201242 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777898904.150048 14201240 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Evaluating Category: NONE
W0000 00:00:1777898904.221225 14201240 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT - TN] Clip: clip_28 | Score: 0.09
[✅ CORRECT - TN] Clip: clip_21 | Score: 0.18
[✅ CORRECT - TN] Clip: clip_26 | Score: 0.09
[✅ CORRECT - TN] Clip: clip_27 | Score: 0.09
[✅ CORRECT - TN] Clip: clip_29 | Score: 0.07
[✅ CORRECT - TN] Clip: clip_33 | Score: 0.18
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_32 | Score: 0.19
[✅ CORRECT - TN] Clip: clip_25 | Score: 0.08
[✅ CORRECT - TN] Clip: clip_22 | Score: 0.22
[✅ CORRECT - TN] Clip: clip_40 | Score: 0.27
[✅ CORRECT - TN] Clip: clip_23 | Score: 0.18
[✅ CORRECT - TN] Clip: clip_24 | Score: 0.18
[✅ CORRECT - TN] Clip: clip_37 | Score: 0.40
[✅ CORRECT - TN] Clip: clip_30 | Score: 0.37
[✅ CORRECT - TN] Clip: clip_39 | Score: 0.25
[✅ CORRECT - TN] Clip: clip_38 | Score: 0.49
[✅ CORRECT - TN] Clip: clip_31 | Score: 0.18
[✅ CORRECT - TN] Clip: clip_36 | Score: 0.41

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.53
[❌ FAIL - FN] Clip: clip_28 | Score: 0.43
[❌ FAIL - FN] Clip: clip_17 | Score: 0.50
[❌ FAIL - FN] Clip: clip_21 | Score: 0.07
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.56
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.55
[✅ CORRECT - TP] Clip: clip_29 | Score: 0.59
[✅ CORRECT - TP] Clip: clip_16 | Score: 0.55
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.55
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.70
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.53
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.53
[✅ CORRECT - TP] Clip: clip_5 | Score: 0.55
[❌ FAIL - FN] Clip: clip_33 | Score: 0.43
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.50
[❌ FAIL - FN] Clip: clip_35 | Score: 0.41
[❌ FAIL - FN] Clip: clip_32 | Score: 0.21
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.53
[❌ FAIL - FN] Clip: clip_13 | Score: 0.00
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.24
[❌ FAIL - FN] Clip: clip_40 | Score: 0.00
[❌ FAIL - FN] Clip: clip_23 | Score: 0.25
[❌ FAIL - FN] Clip: clip_24 | Score: 0.38
[✅ CORRECT - TP] Clip: clip_12 | Score: 0.55
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.51
E0000 00:00:1777899024.081918 14201236 portable_clearcut_uploader.cc:90] Failed to send to clearcut: FAILED_PRECONDITION: Not valid for uploading until: 2026-05-04T09:04:24.073532-04:00
=== Source Location Trace: ===
wireless/android/play/playlog/cplusplus/portable_clearcut_uploader.cc:180
[❌ FAIL - FN] Clip: clip_41 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.64
[✅ CORRECT - TP] Clip: clip_0 | Score: 0.57
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.66
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.67
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.52
[❌ FAIL - FN] Clip: clip_37 | Score: 0.24
[❌ FAIL - FN] Clip: clip_30 | Score: 0.39
[❌ FAIL - FN] Clip: clip_39 | Score: 0.06
[❌ FAIL - FN] Clip: clip_38 | Score: 0.19
[❌ FAIL - FN] Clip: clip_31 | Score: 0.44
[❌ FAIL - FN] Clip: clip_36 | Score: 0.47

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  69.35%
Precision: 100.00%
Recall:    54.76%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 4255
Avg Latency per Frame  : 21.75 ms
Estimated Real-Time FPS: 46.0 FPS

Two approaches to solve the overfitting problem:
- Synthetic data generation using controlnet
- Crop the image for hand only and pass only that info to CNN.

I will try 2nd technique to see how it behaves and if it could solve this overfitting problem.

before that I notice, horizontal flip = true, that flips right and left hands so I removed that and applied heavy data augmentation and the result is underfitting

 python train_custom_cnn.py
🧠 TRAINING CNN: FULL FRAMES + HEAVY AUGMENTATION
==================================================
Loading dataset...
Found 545 files belonging to 2 classes.
Using 436 files for training.
Found 545 files belonging to 2 classes.
Using 109 files for validation.

🚀 Starting Training...
Epoch 1/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 7s 223ms/step - accuracy: 0.5711 - loss: 0.8036 - val_accuracy: 0.6239 - val_loss: 0.6512
Epoch 2/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 150ms/step - accuracy: 0.6261 - loss: 0.6689 - val_accuracy: 0.5505 - val_loss: 0.6737
Epoch 3/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 155ms/step - accuracy: 0.6124 - loss: 0.6494 - val_accuracy: 0.5413 - val_loss: 0.6760
Epoch 4/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 149ms/step - accuracy: 0.6491 - loss: 0.6304 - val_accuracy: 0.5413 - val_loss: 0.6816
Epoch 5/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 149ms/step - accuracy: 0.6376 - loss: 0.6294 - val_accuracy: 0.5413 - val_loss: 0.6853
Epoch 6/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 153ms/step - accuracy: 0.6628 - loss: 0.6298 - val_accuracy: 0.5413 - val_loss: 0.6835
Epoch 7/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 148ms/step - accuracy: 0.6674 - loss: 0.6141 - val_accuracy: 0.5505 - val_loss: 0.6787
Epoch 8/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 149ms/step - accuracy: 0.6904 - loss: 0.5967 - val_accuracy: 0.5505 - val_loss: 0.6849
Epoch 9/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 150ms/step - accuracy: 0.6651 - loss: 0.6069 - val_accuracy: 0.5505 - val_loss: 0.6782
Epoch 10/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 150ms/step - accuracy: 0.6651 - loss: 0.6171 - val_accuracy: 0.5688 - val_loss: 0.6777
Epoch 11/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 147ms/step - accuracy: 0.7018 - loss: 0.5815 - val_accuracy: 0.5688 - val_loss: 0.6754
Epoch 12/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 150ms/step - accuracy: 0.7156 - loss: 0.5790 - val_accuracy: 0.6147 - val_loss: 0.6550
Epoch 13/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 148ms/step - accuracy: 0.6720 - loss: 0.5932 - val_accuracy: 0.5963 - val_loss: 0.6704
Epoch 14/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 149ms/step - accuracy: 0.6674 - loss: 0.6142 - val_accuracy: 0.6514 - val_loss: 0.6379
Epoch 15/15
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 152ms/step - accuracy: 0.6812 - loss: 0.6046 - val_accuracy: 0.6514 - val_loss: 0.6259

==================================================
📦 EXPORTING PURE FLOAT32 TFLITE MODEL...

Accurancy tanked

python benchmark_custom_cnn.py 
==================================================
🚀 BENCHMARKING HYBRID GATED PIPELINE (IEEE FINAL)
==================================================
I0000 00:00:1777934604.515319 14284191 init-domain.cc:128] Fiber init: default domain = pthread, concurrency = 8, prefix = pthread-default
I0000 00:00:1777934604.671393 14284191 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1777934604.680114 14284197 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1777934604.692559 14284202 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Evaluating Category: NONE
W0000 00:00:1777934604.755032 14284203 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT - TN] Clip: clip_28 | Score: 0.27
[✅ CORRECT - TN] Clip: clip_21 | Score: 0.38
[✅ CORRECT - TN] Clip: clip_26 | Score: 0.29
[✅ CORRECT - TN] Clip: clip_27 | Score: 0.26
[✅ CORRECT - TN] Clip: clip_29 | Score: 0.26
[✅ CORRECT - TN] Clip: clip_33 | Score: 0.49
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[❌ FAIL - FP] Clip: clip_32 | Score: 0.50
[✅ CORRECT - TN] Clip: clip_25 | Score: 0.27
[✅ CORRECT - TN] Clip: clip_22 | Score: 0.35
[✅ CORRECT - TN] Clip: clip_40 | Score: 0.43
[✅ CORRECT - TN] Clip: clip_23 | Score: 0.27
[✅ CORRECT - TN] Clip: clip_24 | Score: 0.33
[✅ CORRECT - TN] Clip: clip_37 | Score: 0.48
[✅ CORRECT - TN] Clip: clip_30 | Score: 0.50
[❌ FAIL - FP] Clip: clip_39 | Score: 0.54
[✅ CORRECT - TN] Clip: clip_38 | Score: 0.48
[❌ FAIL - FP] Clip: clip_31 | Score: 0.54
[✅ CORRECT - TN] Clip: clip_36 | Score: 0.50

Evaluating Category: HANDSHAKE
[❌ FAIL - FN] Clip: clip_10 | Score: 0.36
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.50
[❌ FAIL - FN] Clip: clip_17 | Score: 0.36
[❌ FAIL - FN] Clip: clip_21 | Score: 0.30
[❌ FAIL - FN] Clip: clip_19 | Score: 0.48
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.51
[❌ FAIL - FN] Clip: clip_18 | Score: 0.48
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.54
[❌ FAIL - FN] Clip: clip_20 | Score: 0.44
[✅ CORRECT - TP] Clip: clip_29 | Score: 0.57
[❌ FAIL - FN] Clip: clip_16 | Score: 0.36
[❌ FAIL - FN] Clip: clip_11 | Score: 0.36
[❌ FAIL - FN] Clip: clip_4 | Score: 0.37
[❌ FAIL - FN] Clip: clip_3 | Score: 0.36
[❌ FAIL - FN] Clip: clip_2 | Score: 0.37
[❌ FAIL - FN] Clip: clip_5 | Score: 0.36
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.56
[✅ CORRECT - TP] Clip: clip_35 | Score: 0.58
[✅ CORRECT - TP] Clip: clip_32 | Score: 0.50
[❌ FAIL - FN] Clip: clip_14 | Score: 0.36
[❌ FAIL - FN] Clip: clip_13 | Score: 0.00
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.36
[❌ FAIL - FN] Clip: clip_40 | Score: 0.00
[❌ FAIL - FN] Clip: clip_23 | Score: 0.41
E0000 00:00:1777934724.654551 14284192 portable_clearcut_uploader.cc:90] Failed to send to clearcut: FAILED_PRECONDITION: Not valid for uploading until: 2026-05-04T18:59:24.647216-04:00
=== Source Location Trace: ===
wireless/android/play/playlog/cplusplus/portable_clearcut_uploader.cc:180
[❌ FAIL - FN] Clip: clip_24 | Score: 0.46
[❌ FAIL - FN] Clip: clip_12 | Score: 0.37
[❌ FAIL - FN] Clip: clip_15 | Score: 0.36
[❌ FAIL - FN] Clip: clip_41 | Score: 0.00
[❌ FAIL - FN] Clip: clip_9 | Score: 0.38
[❌ FAIL - FN] Clip: clip_0 | Score: 0.37
[❌ FAIL - FN] Clip: clip_7 | Score: 0.37
[❌ FAIL - FN] Clip: clip_6 | Score: 0.36
[❌ FAIL - FN] Clip: clip_1 | Score: 0.36
[❌ FAIL - FN] Clip: clip_8 | Score: 0.35
[❌ FAIL - FN] Clip: clip_37 | Score: 0.45
[❌ FAIL - FN] Clip: clip_30 | Score: 0.46
[❌ FAIL - FN] Clip: clip_39 | Score: 0.34
[❌ FAIL - FN] Clip: clip_38 | Score: 0.47
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.51
[❌ FAIL - FN] Clip: clip_36 | Score: 0.43

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  41.94%
Precision: 75.00%
Recall:    21.43%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 4950
Avg Latency per Frame  : 22.40 ms
Estimated Real-Time FPS: 44.6 FPS
==================================================

I noticed the way we are selecting hand for cropping is diff from how we select hand while benchmarking (using z-index)
so trying to crop hand by choosing hand using z-index similar to benchmarking.
why?
because to see if cropping just hand will help with accuracy since the tiny cnn model doesn't have many neurons to track
all context data when fed complete image also we shrink image to 160*160 so the hand might not be most highlighted feature
of the image.

train_dataset_v3_cropped contains cropped images, many contains hand which are not the active hand.
why not use same other features like thumb, tilt etc to crop the hand? 
y-axis might not work because if the receiving person is taller then hand will raise higher

Loading dataset...
Found 437 files belonging to 2 classes.
Using 350 files for training.
Found 437 files belonging to 2 classes.
Using 87 files for validation.

🚀 Starting Training...
Epoch 1/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 4s 181ms/step - accuracy: 0.5371 - loss: 0.8090 - val_accuracy: 0.5057 - val_loss: 0.6981
Epoch 2/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 163ms/step - accuracy: 0.5914 - loss: 0.6618 - val_accuracy: 0.4828 - val_loss: 0.7122
Epoch 3/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 162ms/step - accuracy: 0.5857 - loss: 0.6622 - val_accuracy: 0.4828 - val_loss: 0.7242
Epoch 4/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 162ms/step - accuracy: 0.5743 - loss: 0.6611 - val_accuracy: 0.4828 - val_loss: 0.7383
Epoch 5/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 165ms/step - accuracy: 0.6343 - loss: 0.6516 - val_accuracy: 0.4828 - val_loss: 0.7653
Epoch 6/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 172ms/step - accuracy: 0.6371 - loss: 0.6246 - val_accuracy: 0.4828 - val_loss: 0.8092
Epoch 7/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 218ms/step - accuracy: 0.6171 - loss: 0.6562 - val_accuracy: 0.4828 - val_loss: 0.8391
Epoch 8/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 170ms/step - accuracy: 0.6000 - loss: 0.6523 - val_accuracy: 0.4828 - val_loss: 0.8516
Epoch 9/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 163ms/step - accuracy: 0.6200 - loss: 0.6320 - val_accuracy: 0.4828 - val_loss: 0.8863
Epoch 10/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 164ms/step - accuracy: 0.6400 - loss: 0.6204 - val_accuracy: 0.4828 - val_loss: 0.9237
Epoch 11/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 163ms/step - accuracy: 0.6686 - loss: 0.6162 - val_accuracy: 0.4828 - val_loss: 0.9462
Epoch 12/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 163ms/step - accuracy: 0.6514 - loss: 0.6197 - val_accuracy: 0.4828 - val_loss: 0.9634
Epoch 13/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 164ms/step - accuracy: 0.6829 - loss: 0.6067 - val_accuracy: 0.4828 - val_loss: 0.9510
Epoch 14/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 162ms/step - accuracy: 0.6914 - loss: 0.5958 - val_accuracy: 0.4828 - val_loss: 0.9162
Epoch 15/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 164ms/step - accuracy: 0.6457 - loss: 0.6197 - val_accuracy: 0.4828 - val_loss: 0.8914

the model gave up, I think the data augmentation was aggressive for cropped hand than full image

==================================================
🧠 TRAINING CNN: FULL FRAMES + HEAVY AUGMENTATION
==================================================
Loading dataset...
Found 437 files belonging to 2 classes.
Using 350 files for training.
Found 437 files belonging to 2 classes.
Using 87 files for validation.

🚀 Starting Training...
Epoch 1/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 3s 181ms/step - accuracy: 0.6114 - loss: 0.8222 - val_accuracy: 0.4943 - val_loss: 0.7417
Epoch 2/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 168ms/step - accuracy: 0.6629 - loss: 0.6403 - val_accuracy: 0.4943 - val_loss: 0.7149
Epoch 3/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 173ms/step - accuracy: 0.7314 - loss: 0.5610 - val_accuracy: 0.4943 - val_loss: 0.7064
Epoch 4/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 172ms/step - accuracy: 0.7371 - loss: 0.5216 - val_accuracy: 0.4943 - val_loss: 0.7206
Epoch 5/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 172ms/step - accuracy: 0.7800 - loss: 0.4852 - val_accuracy: 0.4828 - val_loss: 0.7530
Epoch 6/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 182ms/step - accuracy: 0.7914 - loss: 0.4251 - val_accuracy: 0.4828 - val_loss: 0.7697
Epoch 7/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 173ms/step - accuracy: 0.8286 - loss: 0.4014 - val_accuracy: 0.4828 - val_loss: 0.8217
Epoch 8/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 178ms/step - accuracy: 0.8314 - loss: 0.3977 - val_accuracy: 0.4828 - val_loss: 0.7930
Epoch 9/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 173ms/step - accuracy: 0.8457 - loss: 0.3678 - val_accuracy: 0.4828 - val_loss: 0.8312
Epoch 10/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 172ms/step - accuracy: 0.8543 - loss: 0.3422 - val_accuracy: 0.4828 - val_loss: 0.8672
Epoch 11/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 170ms/step - accuracy: 0.8829 - loss: 0.3213 - val_accuracy: 0.4943 - val_loss: 0.7182
Epoch 12/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 174ms/step - accuracy: 0.8571 - loss: 0.3279 - val_accuracy: 0.4828 - val_loss: 1.0039
Epoch 13/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 174ms/step - accuracy: 0.8914 - loss: 0.3036 - val_accuracy: 0.4828 - val_loss: 1.1716
Epoch 14/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 173ms/step - accuracy: 0.8771 - loss: 0.2702 - val_accuracy: 0.4828 - val_loss: 0.9300
Epoch 15/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 181ms/step - accuracy: 0.8914 - loss: 0.2956 - val_accuracy: 0.4943 - val_loss: 0.8326

IDEA:
why not use same thumb open/close, palm tilt to select the hand in crop dataset? because I see some images like hand resting on a leg are selected since it satisfies the z-index.
also another idea gemini suggested is ranking the hands in the frame according to thumb state, z-index, palm tilt and picking best hand - I think this is good + we can store the non best hand in none category and use it for training?

Gemini: it is a trap! Remember the "Waste of Brain Space" rule we just talked about:
If you take a hand resting on a leg (the lowest score) and put it in the none folder, you are spending CNN neurons teaching it to recognize a resting hand. But in your live robot, Stage 1 (MediaPipe) will never pass a resting hand to the CNN anyway.

🧠 TRAINING CNN: FULL FRAMES + HEAVY AUGMENTATION
==================================================
Loading dataset...
Found 436 files belonging to 2 classes.
Using 349 files for training.
Found 436 files belonging to 2 classes.
Using 87 files for validation.

🚀 Starting Training...
Epoch 1/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 3s 160ms/step - accuracy: 0.5673 - loss: 0.8337 - val_accuracy: 0.5172 - val_loss: 0.7045
Epoch 2/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 152ms/step - accuracy: 0.7106 - loss: 0.5474 - val_accuracy: 0.5057 - val_loss: 0.6971
Epoch 3/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 146ms/step - accuracy: 0.7851 - loss: 0.4613 - val_accuracy: 0.5057 - val_loss: 0.6985
Epoch 4/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 148ms/step - accuracy: 0.7937 - loss: 0.4669 - val_accuracy: 0.5172 - val_loss: 0.7141
Epoch 5/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 155ms/step - accuracy: 0.8309 - loss: 0.4070 - val_accuracy: 0.5517 - val_loss: 0.6726
Epoch 6/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 148ms/step - accuracy: 0.8281 - loss: 0.3960 - val_accuracy: 0.5172 - val_loss: 0.7050
Epoch 7/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 148ms/step - accuracy: 0.8682 - loss: 0.3199 - val_accuracy: 0.5057 - val_loss: 0.7367
Epoch 8/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 147ms/step - accuracy: 0.8567 - loss: 0.3306 - val_accuracy: 0.5172 - val_loss: 0.7178
Epoch 9/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 147ms/step - accuracy: 0.8825 - loss: 0.3019 - val_accuracy: 0.5172 - val_loss: 0.7310
Epoch 10/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 146ms/step - accuracy: 0.9140 - loss: 0.2742 - val_accuracy: 0.5172 - val_loss: 0.7605
Epoch 11/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 145ms/step - accuracy: 0.9226 - loss: 0.2321 - val_accuracy: 0.5402 - val_loss: 0.7074
Epoch 12/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 147ms/step - accuracy: 0.9226 - loss: 0.2289 - val_accuracy: 0.5172 - val_loss: 0.7153
Epoch 13/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 148ms/step - accuracy: 0.9427 - loss: 0.2010 - val_accuracy: 0.5287 - val_loss: 0.7287
Epoch 14/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 146ms/step - accuracy: 0.9255 - loss: 0.1978 - val_accuracy: 0.5517 - val_loss: 0.7049
Epoch 15/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 149ms/step - accuracy: 0.9284 - loss: 0.2145 - val_accuracy: 0.5172 - val_loss: 0.7657

==================================================
📦 EXPORTING PURE FLOAT32 TFLITE MODEL...

The model val_accuracy was not improving more than 51% at all, so we decided to try mobilenetv2 again

python cnn_model_trainer.py
Found 350 images belonging to 2 classes.
Found 86 images belonging to 2 classes.

--- Starting Phase 1: Feature Extraction ---
Epoch 1/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 5s 322ms/step - accuracy: 0.5543 - loss: 0.8025 - val_accuracy: 0.6163 - val_loss: 0.6475
Epoch 2/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 188ms/step - accuracy: 0.6429 - loss: 0.6664 - val_accuracy: 0.6395 - val_loss: 0.6497
Epoch 3/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 185ms/step - accuracy: 0.6143 - loss: 0.6931 - val_accuracy: 0.6047 - val_loss: 0.6584
Epoch 4/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 186ms/step - accuracy: 0.6800 - loss: 0.6157 - val_accuracy: 0.5930 - val_loss: 0.6945
Epoch 5/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 4s 363ms/step - accuracy: 0.7143 - loss: 0.5573 - val_accuracy: 0.5930 - val_loss: 0.7736
Epoch 6/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 196ms/step - accuracy: 0.7600 - loss: 0.5128 - val_accuracy: 0.5581 - val_loss: 0.7872

--- Starting Phase 2: Fine-Tuning ---
Epoch 1/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 6s 262ms/step - accuracy: 0.5057 - loss: 0.7745 - val_accuracy: 0.6512 - val_loss: 0.6539
Epoch 2/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 193ms/step - accuracy: 0.6486 - loss: 0.6773 - val_accuracy: 0.5000 - val_loss: 0.6748
Epoch 3/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 194ms/step - accuracy: 0.5886 - loss: 0.6850 - val_accuracy: 0.5233 - val_loss: 0.7297
Epoch 4/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 199ms/step - accuracy: 0.6600 - loss: 0.6169 - val_accuracy: 0.5116 - val_loss: 0.7407
Epoch 5/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 192ms/step - accuracy: 0.6086 - loss: 0.6624 - val_accuracy: 0.4767 - val_loss: 0.7138
Epoch 6/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 191ms/step - accuracy: 0.6571 - loss: 0.6381 - val_accuracy: 0.5698 - val_loss: 0.7003
Epoch 7/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 198ms/step - accuracy: 0.6771 - loss: 0.6021 - val_accuracy: 0.5698 - val_loss: 0.7012
Epoch 8/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 196ms/step - accuracy: 0.6629 - loss: 0.6470 - val_accuracy: 0.6047 - val_loss: 0.7018
Epoch 9/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 193ms/step - accuracy: 0.7143 - loss: 0.5933 - val_accuracy: 0.5000 - val_loss: 0.7610
Epoch 10/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 192ms/step - accuracy: 0.6914 - loss: 0.5690 - val_accuracy: 0.5581 - val_loss: 0.6644
Epoch 11/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 206ms/step - accuracy: 0.7057 - loss: 0.5792 - val_accuracy: 0.5698 - val_loss: 0.6676
Epoch 12/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 210ms/step - accuracy: 0.7486 - loss: 0.5470 - val_accuracy: 0.5116 - val_loss: 0.7463
Epoch 13/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 194ms/step - accuracy: 0.7514 - loss: 0.5148 - val_accuracy: 0.6047 - val_loss: 0.7004
Epoch 14/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 205ms/step - accuracy: 0.7514 - loss: 0.5169 - val_accuracy: 0.5233 - val_loss: 0.7299
Epoch 15/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 199ms/step - accuracy: 0.7371 - loss: 0.5405 - val_accuracy: 0.5349 - val_loss: 0.7251
Model training complete and saved as handshake_model.keras!
Saved learning_curves.png

--- Running Final Evaluation ---
Found 436 images belonging to 2 classes.
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 142ms/step
Precision:   0.8182
Recall:      0.6711
F1-Score:    0.7373
Specificity: 0.8365
Saved confusion_matrix.png

The above training plan included less data augmentation and no gaussian blur, still validation accuracy didn't improve
so we decided to try mobilenetv2

this helped a bit

python cnn_model_trainer.py

==================================================
🧠 PHASE 3: MOBILENET-V2 (HARD-NEGATIVE CROPS)
==================================================
Found 350 images belonging to 2 classes.
Found 86 images belonging to 2 classes.

--- Phase 1: Feature Extraction ---
Epoch 1/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 5s 280ms/step - accuracy: 0.6343 - loss: 0.7327 - val_accuracy: 0.6395 - val_loss: 0.6496
Epoch 2/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 173ms/step - accuracy: 0.7543 - loss: 0.4899 - val_accuracy: 0.5930 - val_loss: 0.6973
Epoch 3/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 170ms/step - accuracy: 0.8229 - loss: 0.3925 - val_accuracy: 0.6279 - val_loss: 0.7119
Epoch 4/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 171ms/step - accuracy: 0.8514 - loss: 0.3927 - val_accuracy: 0.6047 - val_loss: 0.7581
Epoch 5/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 182ms/step - accuracy: 0.8600 - loss: 0.3339 - val_accuracy: 0.6163 - val_loss: 0.7451

--- Phase 2: Fine-Tuning ---
Epoch 1/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 6s 276ms/step - accuracy: 0.7257 - loss: 0.5553 - val_accuracy: 0.6512 - val_loss: 0.6440
Epoch 2/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 190ms/step - accuracy: 0.8400 - loss: 0.3893 - val_accuracy: 0.6860 - val_loss: 0.6539
Epoch 3/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 192ms/step - accuracy: 0.8600 - loss: 0.3254 - val_accuracy: 0.6977 - val_loss: 0.6787
Epoch 4/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 195ms/step - accuracy: 0.9200 - loss: 0.2499 - val_accuracy: 0.6628 - val_loss: 0.7344
Epoch 5/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 195ms/step - accuracy: 0.9229 - loss: 0.1931 - val_accuracy: 0.6512 - val_loss: 0.7424

✅ Saved Keras model: handshake_model.keras

--- Running Final Evaluation ---
Found 86 images belonging to 2 classes.
3/3 ━━━━━━━━━━━━━━━━━━━━ 1s 305ms/step
Precision:   0.6970
Recall:      0.5111
F1-Score:    0.5897
Specificity: 0.7561

📦 EXPORTING TO TFLITE...
WARNING:absl:Please consider providing the trackable_obj argument in the from_concrete_functions. Providing without the trackable_obj argument is deprecated and it will use the deprecated conversion path.
2026-05-07 08:34:17.468786: I tensorflow/core/grappler/devices.cc:75] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0 (Note: TensorFlow was not compiled with CUDA or ROCm support)
2026-05-07 08:34:17.468898: I tensorflow/core/grappler/clusters/single_machine.cc:361] Starting new session
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1778157258.046105 15064165 tf_tfl_flatbuffer_helpers.cc:390] Ignored output_format.
W0000 00:00:1778157258.046427 15064165 tf_tfl_flatbuffer_helpers.cc:393] Ignored drop_control_dependency.
2026-05-07 08:34:18.374570: I tensorflow/compiler/mlir/lite/flatbuffer_export.cc:3064] Estimated count of arithmetic ops: 309.452 M  ops, equivalently 154.726 M  MACs
✅ Success! MobileNet Edge model saved as mobilenet_handshake.tflite
(.venv_v2) nagashayanaramamurthy@Nagas-MacBook-Pro update_model % 
 
Lets benchmark test it


Evaluating Category: NONE
W0000 00:00:1778157725.904319 15072582 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT - TN] Clip: clip_28 | Score: 0.37
[❌ FAIL - FP] Clip: clip_21 | Score: 0.69
[❌ FAIL - FP] Clip: clip_26 | Score: 0.85
[❌ FAIL - FP] Clip: clip_27 | Score: 0.73
[✅ CORRECT - TN] Clip: clip_29 | Score: 0.46
[❌ FAIL - FP] Clip: clip_33 | Score: 0.77
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[❌ FAIL - FP] Clip: clip_32 | Score: 0.89
[❌ FAIL - FP] Clip: clip_25 | Score: 0.71
[❌ FAIL - FP] Clip: clip_22 | Score: 0.74
[✅ CORRECT - TN] Clip: clip_40 | Score: 0.47
[❌ FAIL - FP] Clip: clip_23 | Score: 0.73
[❌ FAIL - FP] Clip: clip_24 | Score: 0.71
[❌ FAIL - FP] Clip: clip_37 | Score: 0.82
[❌ FAIL - FP] Clip: clip_30 | Score: 0.60
[❌ FAIL - FP] Clip: clip_39 | Score: 0.54
[❌ FAIL - FP] Clip: clip_38 | Score: 0.53
[❌ FAIL - FP] Clip: clip_31 | Score: 0.58
[❌ FAIL - FP] Clip: clip_36 | Score: 0.62

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_17 | Score: 0.84
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.86
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.97
[❌ FAIL - FN] Clip: clip_29 | Score: 0.46
[✅ CORRECT - TP] Clip: clip_16 | Score: 0.55
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.58
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.82
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.58
[✅ CORRECT - TP] Clip: clip_5 | Score: 0.84
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.53
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.92
[✅ CORRECT - TP] Clip: clip_35 | Score: 0.82
[✅ CORRECT - TP] Clip: clip_32 | Score: 0.66
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.56
[❌ FAIL - FN] Clip: clip_13 | Score: 0.00
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.39
[❌ FAIL - FN] Clip: clip_40 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_23 | Score: 0.50
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.64
[✅ CORRECT - TP] Clip: clip_12 | Score: 0.85
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.78
[❌ FAIL - FN] Clip: clip_41 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_0 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.56
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.56
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_39 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_38 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.50

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  66.13%
Precision: 70.59%
Recall:    85.71%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 2874
Avg Latency per Frame  : 21.71 ms
Estimated Real-Time FPS: 46.1 FPS

change cnn threshold to 70%

Evaluating Category: NONE
W0000 00:00:1778158461.705279 15081656 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT - TN] Clip: clip_28 | Score: 0.37
[❌ FAIL - FP] Clip: clip_21 | Score: 0.71
[❌ FAIL - FP] Clip: clip_26 | Score: 0.85
[❌ FAIL - FP] Clip: clip_27 | Score: 0.73
[✅ CORRECT - TN] Clip: clip_29 | Score: 0.46
[❌ FAIL - FP] Clip: clip_33 | Score: 0.77
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[❌ FAIL - FP] Clip: clip_32 | Score: 0.89
[❌ FAIL - FP] Clip: clip_25 | Score: 0.71
[❌ FAIL - FP] Clip: clip_22 | Score: 0.74
[✅ CORRECT - TN] Clip: clip_40 | Score: 0.47
[❌ FAIL - FP] Clip: clip_23 | Score: 0.73
[❌ FAIL - FP] Clip: clip_24 | Score: 0.71
[❌ FAIL - FP] Clip: clip_37 | Score: 0.82
[✅ CORRECT - TN] Clip: clip_30 | Score: 0.60
[✅ CORRECT - TN] Clip: clip_39 | Score: 0.69
[✅ CORRECT - TN] Clip: clip_38 | Score: 0.65
[❌ FAIL - FP] Clip: clip_31 | Score: 0.72
[❌ FAIL - FP] Clip: clip_36 | Score: 0.71

Evaluating Category: HANDSHAKE
[❌ FAIL - FN] Clip: clip_10 | Score: 0.69
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_17 | Score: 0.84
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.70
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.75
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.86
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.76
[❌ FAIL - FN] Clip: clip_27 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.97
[❌ FAIL - FN] Clip: clip_29 | Score: 0.46
[✅ CORRECT - TP] Clip: clip_16 | Score: 0.71
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.71
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.71
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.82
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.72
[✅ CORRECT - TP] Clip: clip_5 | Score: 0.84
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.80
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.92
[✅ CORRECT - TP] Clip: clip_35 | Score: 0.82
[✅ CORRECT - TP] Clip: clip_32 | Score: 0.76
[❌ FAIL - FN] Clip: clip_14 | Score: 0.56
[❌ FAIL - FN] Clip: clip_13 | Score: 0.00
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.39
[❌ FAIL - FN] Clip: clip_40 | Score: 0.00
[❌ FAIL - FN] Clip: clip_23 | Score: 0.51
[❌ FAIL - FN] Clip: clip_24 | Score: 0.64
[✅ CORRECT - TP] Clip: clip_12 | Score: 0.85
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.78
[❌ FAIL - FN] Clip: clip_41 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.83
[✅ CORRECT - TP] Clip: clip_0 | Score: 0.78
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.71
[❌ FAIL - FN] Clip: clip_6 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.76
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.76
[❌ FAIL - FN] Clip: clip_37 | Score: 0.67
[❌ FAIL - FN] Clip: clip_30 | Score: 0.66
[❌ FAIL - FN] Clip: clip_39 | Score: 0.68
[❌ FAIL - FN] Clip: clip_38 | Score: 0.52
[❌ FAIL - FN] Clip: clip_31 | Score: 0.67
[❌ FAIL - FN] Clip: clip_36 | Score: 0.67

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  51.61%
Precision: 66.67%
Recall:    57.14%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3810
Avg Latency per Frame  : 22.62 ms
Estimated Real-Time FPS: 44.2 FPS

Try cnn threshold with 60%


Evaluating Category: NONE
W0000 00:00:1778167646.948906 15104556 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT - TN] Clip: clip_28 | Score: 0.37
[❌ FAIL - FP] Clip: clip_21 | Score: 0.69
[❌ FAIL - FP] Clip: clip_26 | Score: 0.85
[❌ FAIL - FP] Clip: clip_27 | Score: 0.73
[✅ CORRECT - TN] Clip: clip_29 | Score: 0.46
[❌ FAIL - FP] Clip: clip_33 | Score: 0.77
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[❌ FAIL - FP] Clip: clip_32 | Score: 0.89
[❌ FAIL - FP] Clip: clip_25 | Score: 0.71
[❌ FAIL - FP] Clip: clip_22 | Score: 0.74
[✅ CORRECT - TN] Clip: clip_40 | Score: 0.47
[❌ FAIL - FP] Clip: clip_23 | Score: 0.73
[❌ FAIL - FP] Clip: clip_24 | Score: 0.71
[❌ FAIL - FP] Clip: clip_37 | Score: 0.82
[✅ CORRECT - TN] Clip: clip_30 | Score: 0.60
[❌ FAIL - FP] Clip: clip_39 | Score: 0.69
[❌ FAIL - FP] Clip: clip_38 | Score: 0.65
[❌ FAIL - FP] Clip: clip_31 | Score: 0.64
[❌ FAIL - FP] Clip: clip_36 | Score: 0.62

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.69
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_17 | Score: 0.84
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.86
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.97
[❌ FAIL - FN] Clip: clip_29 | Score: 0.46
[✅ CORRECT - TP] Clip: clip_16 | Score: 0.71
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.64
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.82
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_5 | Score: 0.84
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.92
[✅ CORRECT - TP] Clip: clip_35 | Score: 0.82
[✅ CORRECT - TP] Clip: clip_32 | Score: 0.66
[❌ FAIL - FN] Clip: clip_14 | Score: 0.56
[❌ FAIL - FN] Clip: clip_13 | Score: 0.00
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.39
[❌ FAIL - FN] Clip: clip_40 | Score: 0.00
[❌ FAIL - FN] Clip: clip_23 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.64
[✅ CORRECT - TP] Clip: clip_12 | Score: 0.85
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.78
[❌ FAIL - FN] Clip: clip_41 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.64
[✅ CORRECT - TP] Clip: clip_0 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_39 | Score: 0.68
[❌ FAIL - FN] Clip: clip_38 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.67
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.67

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  62.90%
Precision: 70.21%
Recall:    78.57%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3188
Avg Latency per Frame  : 22.16 ms
Estimated Real-Time FPS: 45.1 FPS

Tried palm tilt 20-120 and cnn threshold 0.50

the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT - TN] Clip: clip_28 | Score: 0.37
[❌ FAIL - FP] Clip: clip_21 | Score: 0.69
[✅ CORRECT - TN] Clip: clip_26 | Score: 0.31
[❌ FAIL - FP] Clip: clip_27 | Score: 0.73
[✅ CORRECT - TN] Clip: clip_29 | Score: 0.46
[❌ FAIL - FP] Clip: clip_33 | Score: 0.77
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[❌ FAIL - FP] Clip: clip_32 | Score: 0.89
[❌ FAIL - FP] Clip: clip_25 | Score: 0.60
[❌ FAIL - FP] Clip: clip_22 | Score: 0.74
[✅ CORRECT - TN] Clip: clip_40 | Score: 0.47
[✅ CORRECT - TN] Clip: clip_23 | Score: 0.00
[❌ FAIL - FP] Clip: clip_24 | Score: 0.85
[❌ FAIL - FP] Clip: clip_37 | Score: 0.82
[❌ FAIL - FP] Clip: clip_30 | Score: 0.60
[❌ FAIL - FP] Clip: clip_39 | Score: 0.55
[❌ FAIL - FP] Clip: clip_38 | Score: 0.53
[❌ FAIL - FP] Clip: clip_31 | Score: 0.58
[❌ FAIL - FP] Clip: clip_36 | Score: 0.62

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.53
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_17 | Score: 0.84
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.86
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.97
[❌ FAIL - FN] Clip: clip_29 | Score: 0.46
[✅ CORRECT - TP] Clip: clip_16 | Score: 0.55
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.58
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.82
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.58
[✅ CORRECT - TP] Clip: clip_5 | Score: 0.84
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.53
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.92
[✅ CORRECT - TP] Clip: clip_35 | Score: 0.82
[✅ CORRECT - TP] Clip: clip_32 | Score: 0.55
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.56
[❌ FAIL - FN] Clip: clip_13 | Score: 0.00
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.39
[❌ FAIL - FN] Clip: clip_40 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_23 | Score: 0.50
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.64
[✅ CORRECT - TP] Clip: clip_12 | Score: 0.85
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.78
[❌ FAIL - FN] Clip: clip_41 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_0 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.56
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_39 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_38 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.50

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  69.35%
Precision: 73.47%
Recall:    85.71%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3016
Avg Latency per Frame  : 21.57 ms
Estimated Real-Time FPS: 46.4 FPS

increased palm tilt to 60-120 then accuracy jumped

python benchmark_script_v5.py         
==================================================
🚀 BENCHMARKING MACRO-TO-MICRO HYBRID PIPELINE
==================================================
I0000 00:00:1778168819.151213 15117336 init-domain.cc:128] Fiber init: default domain = pthread, concurrency = 8, prefix = pthread-default
I0000 00:00:1778168819.332537 15117336 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1778168819.353336 15117338 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1778168819.372699 15117338 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Evaluating Category: NONE
W0000 00:00:1778168819.437806 15117341 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT - TN] Clip: clip_28 | Score: 0.00
[❌ FAIL - FP] Clip: clip_21 | Score: 0.69
[✅ CORRECT - TN] Clip: clip_26 | Score: 0.00
[❌ FAIL - FP] Clip: clip_27 | Score: 0.73
[✅ CORRECT - TN] Clip: clip_29 | Score: 0.46
[❌ FAIL - FP] Clip: clip_33 | Score: 0.77
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_32 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FP] Clip: clip_22 | Score: 0.74
[✅ CORRECT - TN] Clip: clip_40 | Score: 0.47
[✅ CORRECT - TN] Clip: clip_23 | Score: 0.00
[❌ FAIL - FP] Clip: clip_24 | Score: 0.85
[❌ FAIL - FP] Clip: clip_37 | Score: 0.82
[✅ CORRECT - TN] Clip: clip_30 | Score: 0.49
[✅ CORRECT - TN] Clip: clip_39 | Score: 0.00
[❌ FAIL - FP] Clip: clip_38 | Score: 0.53
[❌ FAIL - FP] Clip: clip_31 | Score: 0.58
[❌ FAIL - FP] Clip: clip_36 | Score: 0.71

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.53
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_17 | Score: 0.84
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.86
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.97
[❌ FAIL - FN] Clip: clip_29 | Score: 0.46
[✅ CORRECT - TP] Clip: clip_16 | Score: 0.55
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.58
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.82
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_5 | Score: 0.84
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.53
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.92
[✅ CORRECT - TP] Clip: clip_35 | Score: 0.82
[✅ CORRECT - TP] Clip: clip_32 | Score: 0.55
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.56
[❌ FAIL - FN] Clip: clip_13 | Score: 0.00
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.39
[❌ FAIL - FN] Clip: clip_40 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_23 | Score: 0.50
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.64
[❌ FAIL - FN] Clip: clip_12 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.78
[❌ FAIL - FN] Clip: clip_41 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_0 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.56
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_39 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_38 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.50

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  74.19%
Precision: 79.55%
Recall:    83.33%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3257
Avg Latency per Frame  : 21.69 ms
Estimated Real-Time FPS: 46.1 FPS
==================================================


change thumbstate to 0.05? this was tanking accuracy but lets try

python benchmark_script_v5.py
==================================================
🚀 BENCHMARKING MACRO-TO-MICRO HYBRID PIPELINE
==================================================
I0000 00:00:1778169321.572569 15122886 init-domain.cc:128] Fiber init: default domain = pthread, concurrency = 8, prefix = pthread-default
I0000 00:00:1778169321.742547 15122886 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1778169321.755113 15122890 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1778169321.773590 15122890 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Evaluating Category: NONE
W0000 00:00:1778169321.835728 15122896 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT - TN] Clip: clip_28 | Score: 0.00
[❌ FAIL - FP] Clip: clip_21 | Score: 0.69
[✅ CORRECT - TN] Clip: clip_26 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_27 | Score: 0.46
[✅ CORRECT - TN] Clip: clip_29 | Score: 0.46
[❌ FAIL - FP] Clip: clip_33 | Score: 0.77
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_32 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FP] Clip: clip_22 | Score: 0.74
[✅ CORRECT - TN] Clip: clip_40 | Score: 0.47
[✅ CORRECT - TN] Clip: clip_23 | Score: 0.00
[❌ FAIL - FP] Clip: clip_24 | Score: 0.85
[❌ FAIL - FP] Clip: clip_37 | Score: 0.82
[✅ CORRECT - TN] Clip: clip_30 | Score: 0.49
[✅ CORRECT - TN] Clip: clip_39 | Score: 0.00
[❌ FAIL - FP] Clip: clip_38 | Score: 0.53
[❌ FAIL - FP] Clip: clip_31 | Score: 0.58
[❌ FAIL - FP] Clip: clip_36 | Score: 0.71

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.53
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.73
[❌ FAIL - FN] Clip: clip_17 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.86
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.54
[❌ FAIL - FN] Clip: clip_20 | Score: 0.00
[❌ FAIL - FN] Clip: clip_29 | Score: 0.38
[❌ FAIL - FN] Clip: clip_16 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.58
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.91
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_5 | Score: 0.84
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.53
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.92
[✅ CORRECT - TP] Clip: clip_35 | Score: 0.82
[✅ CORRECT - TP] Clip: clip_32 | Score: 0.55
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.56
[❌ FAIL - FN] Clip: clip_13 | Score: 0.00
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.39
[❌ FAIL - FN] Clip: clip_40 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_23 | Score: 0.50
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.51
[❌ FAIL - FN] Clip: clip_12 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.74
[❌ FAIL - FN] Clip: clip_41 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_0 | Score: 0.76
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.81
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.60
[❌ FAIL - FN] Clip: clip_8 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_39 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_38 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.50

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  69.35%
Precision: 79.49%
Recall:    73.81%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3607
Avg Latency per Frame  : 21.88 ms
Estimated Real-Time FPS: 45.7 FPS

is there anything better than mobilenetv2 we should consider? 
Lets consider MobileNetV3Small

python cnn_model_trainer.py 

==================================================
🧠 PHASE 3: MOBILENET-V3 SMALL (HARD-NEGATIVE CROPS)
==================================================
Found 350 images belonging to 2 classes.
Found 86 images belonging to 2 classes.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv_v2/lib/python3.10/site-packages/keras/src/applications/mobilenet_v3.py:454: UserWarning: `input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.
  return MobileNetV3(
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5
4334752/4334752 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step

--- Phase 1: Feature Extraction ---
Epoch 1/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 6s 386ms/step - accuracy: 0.5029 - loss: 0.7158 - val_accuracy: 0.4767 - val_loss: 0.6930
Epoch 2/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 1s 133ms/step - accuracy: 0.5057 - loss: 0.6944 - val_accuracy: 0.5233 - val_loss: 0.6887
Epoch 3/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 136ms/step - accuracy: 0.4914 - loss: 0.7029 - val_accuracy: 0.5581 - val_loss: 0.6902
Epoch 4/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 135ms/step - accuracy: 0.4886 - loss: 0.7021 - val_accuracy: 0.4767 - val_loss: 0.6932
Epoch 5/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 150ms/step - accuracy: 0.5029 - loss: 0.7041 - val_accuracy: 0.5233 - val_loss: 0.6880
Epoch 6/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 1s 132ms/step - accuracy: 0.5171 - loss: 0.6916 - val_accuracy: 0.4767 - val_loss: 0.6908
Epoch 7/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 1s 132ms/step - accuracy: 0.5086 - loss: 0.6937 - val_accuracy: 0.5930 - val_loss: 0.6862
Epoch 8/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 138ms/step - accuracy: 0.5486 - loss: 0.6849 - val_accuracy: 0.5698 - val_loss: 0.6881
Epoch 9/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 145ms/step - accuracy: 0.5543 - loss: 0.6835 - val_accuracy: 0.5698 - val_loss: 0.6873
Epoch 10/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 139ms/step - accuracy: 0.5143 - loss: 0.6881 - val_accuracy: 0.5465 - val_loss: 0.6891
Epoch 11/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 138ms/step - accuracy: 0.5543 - loss: 0.6817 - val_accuracy: 0.5581 - val_loss: 0.6850
Epoch 12/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 140ms/step - accuracy: 0.5571 - loss: 0.6856 - val_accuracy: 0.5814 - val_loss: 0.6823
Epoch 13/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 142ms/step - accuracy: 0.5629 - loss: 0.6750 - val_accuracy: 0.5000 - val_loss: 0.6953
Epoch 14/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 149ms/step - accuracy: 0.5343 - loss: 0.6817 - val_accuracy: 0.5814 - val_loss: 0.6809
Epoch 15/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 146ms/step - accuracy: 0.5657 - loss: 0.6811 - val_accuracy: 0.6047 - val_loss: 0.6787

--- Phase 2: Fine-Tuning ---
Epoch 1/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 5s 225ms/step - accuracy: 0.4971 - loss: 0.9128 - val_accuracy: 0.5930 - val_loss: 0.6788
Epoch 2/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 136ms/step - accuracy: 0.5457 - loss: 0.7897 - val_accuracy: 0.5581 - val_loss: 0.6810
Epoch 3/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 140ms/step - accuracy: 0.6286 - loss: 0.6882 - val_accuracy: 0.6395 - val_loss: 0.6758
Epoch 4/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 139ms/step - accuracy: 0.5829 - loss: 0.7014 - val_accuracy: 0.5581 - val_loss: 0.6850
Epoch 5/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 137ms/step - accuracy: 0.6086 - loss: 0.6498 - val_accuracy: 0.5930 - val_loss: 0.6829
Epoch 6/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 136ms/step - accuracy: 0.6800 - loss: 0.5925 - val_accuracy: 0.5349 - val_loss: 0.6861
Epoch 7/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 140ms/step - accuracy: 0.6686 - loss: 0.5710 - val_accuracy: 0.5349 - val_loss: 0.6835

✅ Saved Keras model: handshake_model.keras

--- Running Final Evaluation ---
Found 86 images belonging to 2 classes.
3/3 ━━━━━━━━━━━━━━━━━━━━ 1s 251ms/step
Precision:   0.6471
Recall:      0.7333
F1-Score:    0.6875
Specificity: 0.5610

📦 EXPORTING TO TFLITE...
WARNING:absl:Please consider providing the trackable_obj argument in the from_concrete_functions. Providing without the trackable_obj argument is deprecated and it will use the deprecated conversion path.
2026-05-07 12:11:07.381947: I tensorflow/core/grappler/devices.cc:75] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0 (Note: TensorFlow was not compiled with CUDA or ROCm support)
2026-05-07 12:11:07.382036: I tensorflow/core/grappler/clusters/single_machine.cc:361] Starting new session
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1778170267.912246 15131870 tf_tfl_flatbuffer_helpers.cc:390] Ignored output_format.
W0000 00:00:1778170267.912680 15131870 tf_tfl_flatbuffer_helpers.cc:393] Ignored drop_control_dependency.
2026-05-07 12:11:08.137220: I tensorflow/compiler/mlir/lite/flatbuffer_export.cc:3064] Estimated count of arithmetic ops: 57.667 M  ops, equivalently 28.833 M  MACs
✅ Success! MobileNet Edge model saved as mobilenet_handshake.tflite
This above thing was underfitting

so we tried mobilenetv3large

==================================================
🧠 PHASE 3: MOBILENET-V2 (HARD-NEGATIVE CROPS)
==================================================
Found 350 images belonging to 2 classes.
Found 86 images belonging to 2 classes.
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv_v2/lib/python3.10/site-packages/keras/src/applications/mobilenet_v3.py:519: UserWarning: `input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.
  return MobileNetV3(
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/weights_mobilenet_v3_large_224_1.0_float_no_top_v2.h5
12683000/12683000 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step

--- Phase 1: Feature Extraction ---
Epoch 1/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 5s 283ms/step - accuracy: 0.5914 - loss: 0.7689 - val_accuracy: 0.5581 - val_loss: 0.7157
Epoch 2/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 155ms/step - accuracy: 0.7229 - loss: 0.5165 - val_accuracy: 0.6047 - val_loss: 0.7579
Epoch 3/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 159ms/step - accuracy: 0.8229 - loss: 0.3814 - val_accuracy: 0.6163 - val_loss: 0.6977
Epoch 4/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 157ms/step - accuracy: 0.8886 - loss: 0.2864 - val_accuracy: 0.6163 - val_loss: 0.6843
Epoch 5/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 168ms/step - accuracy: 0.8857 - loss: 0.2822 - val_accuracy: 0.6512 - val_loss: 0.6291
Epoch 6/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 157ms/step - accuracy: 0.9114 - loss: 0.2391 - val_accuracy: 0.6163 - val_loss: 0.6845
Epoch 7/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 157ms/step - accuracy: 0.9000 - loss: 0.2311 - val_accuracy: 0.6628 - val_loss: 0.7513
Epoch 8/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 162ms/step - accuracy: 0.9400 - loss: 0.1942 - val_accuracy: 0.6512 - val_loss: 0.7958
Epoch 9/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 165ms/step - accuracy: 0.9229 - loss: 0.1990 - val_accuracy: 0.5930 - val_loss: 0.8631

--- Phase 2: Fine-Tuning ---
Epoch 1/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 6s 263ms/step - accuracy: 0.8457 - loss: 0.3193 - val_accuracy: 0.6279 - val_loss: 0.6889
Epoch 2/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 175ms/step - accuracy: 0.9057 - loss: 0.2417 - val_accuracy: 0.6628 - val_loss: 0.7044
Epoch 3/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 172ms/step - accuracy: 0.9114 - loss: 0.2439 - val_accuracy: 0.6628 - val_loss: 0.6910
Epoch 4/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 171ms/step - accuracy: 0.9343 - loss: 0.2258 - val_accuracy: 0.6512 - val_loss: 0.7331
Epoch 5/15
11/11 ━━━━━━━━━━━━━━━━━━━━ 2s 179ms/step - accuracy: 0.9257 - loss: 0.1934 - val_accuracy: 0.6395 - val_loss: 0.7297

✅ Saved Keras model: handshake_model.keras

--- Running Final Evaluation ---
Found 86 images belonging to 2 classes.
3/3 ━━━━━━━━━━━━━━━━━━━━ 1s 340ms/step
Precision:   0.6977
Recall:      0.6667
F1-Score:    0.6818
Specificity: 0.6829


This was overfitting and accuracy shot to 94% so better to stick with mobilenetv2

Todo while writing:
- Explain Fig 2, Fig 3, Fig 4 in detail under methadology, GRL section - explain reach, palm orientation calculation by drawing over the fig2, fig3/4 images. (partialliy done)
- Refine images, tables, graphs, confusion matrix etc
- spanwned a 6 page version for workshop


Refine table 3 in workshop paper

reach + loose tilt(20-160):
python benchmark_script_v5.py 
==================================================
🚀 BENCHMARKING MACRO-TO-MICRO HYBRID PIPELINE
==================================================
I0000 00:00:1778880782.882456 17951722 init-domain.cc:128] Fiber init: default domain = pthread, concurrency = 8, prefix = pthread-default
I0000 00:00:1778880783.084532 17951722 gl_context.cc:407] GL version: 2.1 (2.1 Metal - 90.5), renderer: Apple M1 Pro
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1778880783.095933 17951729 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1778880783.113271 17951726 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Evaluating Category: NONE
W0000 00:00:1778880783.185048 17951732 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT - TN] Clip: clip_28 | Score: 0.31
[❌ FAIL - FP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TN] Clip: clip_26 | Score: 0.36
[❌ FAIL - FP] Clip: clip_27 | Score: 0.68
[❌ FAIL - FP] Clip: clip_29 | Score: 0.62
[❌ FAIL - FP] Clip: clip_33 | Score: 0.70
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[❌ FAIL - FP] Clip: clip_32 | Score: 0.65
[✅ CORRECT - TN] Clip: clip_25 | Score: 0.32
[❌ FAIL - FP] Clip: clip_22 | Score: 0.65
[✅ CORRECT - TN] Clip: clip_40 | Score: 0.53
[✅ CORRECT - TN] Clip: clip_23 | Score: 0.35
[❌ FAIL - FP] Clip: clip_24 | Score: 0.67
[✅ CORRECT - TN] Clip: clip_37 | Score: 0.56
[✅ CORRECT - TN] Clip: clip_30 | Score: 0.58
[❌ FAIL - FP] Clip: clip_39 | Score: 0.73
[✅ CORRECT - TN] Clip: clip_38 | Score: 0.48
[❌ FAIL - FP] Clip: clip_31 | Score: 0.70
[❌ FAIL - FP] Clip: clip_36 | Score: 0.62

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.72
[❌ FAIL - FN] Clip: clip_17 | Score: 0.33
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.78
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.74
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.75
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.89
[❌ FAIL - FN] Clip: clip_29 | Score: 0.49
[❌ FAIL - FN] Clip: clip_16 | Score: 0.51
[❌ FAIL - FN] Clip: clip_11 | Score: 0.44
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.73
[❌ FAIL - FN] Clip: clip_3 | Score: 0.48
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.62
[❌ FAIL - FN] Clip: clip_5 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.76
[❌ FAIL - FN] Clip: clip_35 | Score: 0.44
[❌ FAIL - FN] Clip: clip_32 | Score: 0.48
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.67
[❌ FAIL - FN] Clip: clip_13 | Score: 0.00
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.42
[❌ FAIL - FN] Clip: clip_40 | Score: 0.00
[❌ FAIL - FN] Clip: clip_23 | Score: 0.39
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.68
[❌ FAIL - FN] Clip: clip_12 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.61
[❌ FAIL - FN] Clip: clip_41 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.61
E0000 00:00:1778880903.087834 17951723 portable_clearcut_uploader.cc:90] Failed to send to clearcut: FAILED_PRECONDITION: Not valid for uploading until: 2026-05-15T17:49:03.0798-04:00
=== Source Location Trace: === 
wireless/android/play/playlog/cplusplus/portable_clearcut_uploader.cc:180
[❌ FAIL - FN] Clip: clip_0 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.66
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.70
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.66
[❌ FAIL - FN] Clip: clip_39 | Score: 0.25
[❌ FAIL - FN] Clip: clip_38 | Score: 0.25
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.61

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  54.84%
Precision: 70.59%
Recall:    57.14%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3971
Avg Latency per Frame  : 22.75 ms
Estimated Real-Time FPS: 44.0 FPS

reach + no tilt(0-180):

[✅ CORRECT - TN] Clip: clip_28 | Score: 0.31
[❌ FAIL - FP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TN] Clip: clip_26 | Score: 0.36
[❌ FAIL - FP] Clip: clip_27 | Score: 0.68
[❌ FAIL - FP] Clip: clip_29 | Score: 0.62
[❌ FAIL - FP] Clip: clip_33 | Score: 0.70
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.28
[❌ FAIL - FP] Clip: clip_32 | Score: 0.65
[✅ CORRECT - TN] Clip: clip_25 | Score: 0.33
[❌ FAIL - FP] Clip: clip_22 | Score: 0.65
[✅ CORRECT - TN] Clip: clip_40 | Score: 0.53
[✅ CORRECT - TN] Clip: clip_23 | Score: 0.51
[❌ FAIL - FP] Clip: clip_24 | Score: 0.67
[✅ CORRECT - TN] Clip: clip_37 | Score: 0.56
[❌ FAIL - FP] Clip: clip_30 | Score: 0.78
[❌ FAIL - FP] Clip: clip_39 | Score: 0.73
[✅ CORRECT - TN] Clip: clip_38 | Score: 0.48
[❌ FAIL - FP] Clip: clip_31 | Score: 0.70
[❌ FAIL - FP] Clip: clip_36 | Score: 0.62

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.72
[❌ FAIL - FN] Clip: clip_17 | Score: 0.33
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.78
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.74
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.75
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.89
[❌ FAIL - FN] Clip: clip_29 | Score: 0.49
[❌ FAIL - FN] Clip: clip_16 | Score: 0.51
[❌ FAIL - FN] Clip: clip_11 | Score: 0.44
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.73
[❌ FAIL - FN] Clip: clip_3 | Score: 0.49
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.62
[❌ FAIL - FN] Clip: clip_5 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.76
[❌ FAIL - FN] Clip: clip_35 | Score: 0.44
[❌ FAIL - FN] Clip: clip_32 | Score: 0.48
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.67
[❌ FAIL - FN] Clip: clip_13 | Score: 0.00
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.42
[❌ FAIL - FN] Clip: clip_40 | Score: 0.00
[❌ FAIL - FN] Clip: clip_23 | Score: 0.39
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.68
[❌ FAIL - FN] Clip: clip_12 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.61
[❌ FAIL - FN] Clip: clip_41 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.61
[❌ FAIL - FN] Clip: clip_0 | Score: 0.51
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.66
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.70
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.66
[❌ FAIL - FN] Clip: clip_39 | Score: 0.25
[❌ FAIL - FN] Clip: clip_38 | Score: 0.25
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.61

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  53.23%
Precision: 68.57%
Recall:    57.14%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3968
Avg Latency per Frame  : 22.00 ms
Estimated Real-Time FPS: 45.4 FPS

no reach + no tilt(0-180):

MAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT - TN] Clip: clip_28 | Score: 0.58
[❌ FAIL - FP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TN] Clip: clip_26 | Score: 0.36
[❌ FAIL - FP] Clip: clip_27 | Score: 0.68
[❌ FAIL - FP] Clip: clip_29 | Score: 0.62
[❌ FAIL - FP] Clip: clip_33 | Score: 0.70
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.28
[❌ FAIL - FP] Clip: clip_32 | Score: 0.65
[✅ CORRECT - TN] Clip: clip_25 | Score: 0.33
[❌ FAIL - FP] Clip: clip_22 | Score: 0.65
[❌ FAIL - FP] Clip: clip_40 | Score: 0.63
[✅ CORRECT - TN] Clip: clip_23 | Score: 0.51
[❌ FAIL - FP] Clip: clip_24 | Score: 0.67
[✅ CORRECT - TN] Clip: clip_37 | Score: 0.56
[❌ FAIL - FP] Clip: clip_30 | Score: 0.72
[❌ FAIL - FP] Clip: clip_39 | Score: 0.63
[✅ CORRECT - TN] Clip: clip_38 | Score: 0.48
[❌ FAIL - FP] Clip: clip_31 | Score: 0.70
[❌ FAIL - FP] Clip: clip_36 | Score: 0.62

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.72
[❌ FAIL - FN] Clip: clip_17 | Score: 0.33
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.78
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.74
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.75
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.89
[❌ FAIL - FN] Clip: clip_29 | Score: 0.49
[✅ CORRECT - TP] Clip: clip_16 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_5 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.76
[✅ CORRECT - TP] Clip: clip_35 | Score: 0.66
[❌ FAIL - FN] Clip: clip_32 | Score: 0.48
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.67
[✅ CORRECT - TP] Clip: clip_13 | Score: 0.68
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.47
[✅ CORRECT - TP] Clip: clip_40 | Score: 0.61
[❌ FAIL - FN] Clip: clip_23 | Score: 0.46
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_12 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_41 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.61
[❌ FAIL - FN] Clip: clip_0 | Score: 0.59
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.66
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.70
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.66
[❌ FAIL - FN] Clip: clip_39 | Score: 0.52
[❌ FAIL - FN] Clip: clip_38 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.61

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  66.13%
Precision: 73.33%
Recall:    78.57%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3422
Avg Latency per Frame  : 21.88 ms
Estimated Real-Time FPS: 45.7 FPS

reach + strict tilt(60-120):
MAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT - TN] Clip: clip_28 | Score: 0.00
[❌ FAIL - FP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TN] Clip: clip_26 | Score: 0.00
[❌ FAIL - FP] Clip: clip_27 | Score: 0.68
[❌ FAIL - FP] Clip: clip_29 | Score: 0.62
[❌ FAIL - FP] Clip: clip_33 | Score: 0.70
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_32 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FP] Clip: clip_22 | Score: 0.65
[✅ CORRECT - TN] Clip: clip_40 | Score: 0.53
[✅ CORRECT - TN] Clip: clip_23 | Score: 0.00
[❌ FAIL - FP] Clip: clip_24 | Score: 0.61
[✅ CORRECT - TN] Clip: clip_37 | Score: 0.56
[✅ CORRECT - TN] Clip: clip_30 | Score: 0.58
[✅ CORRECT - TN] Clip: clip_39 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_38 | Score: 0.48
[❌ FAIL - FP] Clip: clip_31 | Score: 0.70
[✅ CORRECT - TN] Clip: clip_36 | Score: 0.35

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.72
[❌ FAIL - FN] Clip: clip_17 | Score: 0.33
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.78
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.74
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.75
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.89
[❌ FAIL - FN] Clip: clip_29 | Score: 0.49
[❌ FAIL - FN] Clip: clip_16 | Score: 0.51
[❌ FAIL - FN] Clip: clip_11 | Score: 0.35
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.73
[❌ FAIL - FN] Clip: clip_3 | Score: 0.48
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.62
[❌ FAIL - FN] Clip: clip_5 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.76
[❌ FAIL - FN] Clip: clip_35 | Score: 0.44
[❌ FAIL - FN] Clip: clip_32 | Score: 0.43
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.67
[❌ FAIL - FN] Clip: clip_13 | Score: 0.00
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.42
[❌ FAIL - FN] Clip: clip_40 | Score: 0.00
[❌ FAIL - FN] Clip: clip_23 | Score: 0.39
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.61
[❌ FAIL - FN] Clip: clip_12 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.61
[❌ FAIL - FN] Clip: clip_41 | Score: 0.00
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.61
[❌ FAIL - FN] Clip: clip_0 | Score: 0.42
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.66
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.70
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.66
[❌ FAIL - FN] Clip: clip_39 | Score: 0.25
[❌ FAIL - FN] Clip: clip_38 | Score: 0.25
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.61

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  59.68%
Precision: 77.42%
Recall:    57.14%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 4154
Avg Latency per Frame  : 21.77 ms
Estimated Real-Time FPS: 45.9 FPS

Reach 0.0 and strict tilt (20-160)

Evaluating Category: NONE
W0000 00:00:1778882307.783869 17970862 landmark_projection_calculator.cc:81] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
[✅ CORRECT - TN] Clip: clip_28 | Score: 0.54
[❌ FAIL - FP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TN] Clip: clip_26 | Score: 0.36
[❌ FAIL - FP] Clip: clip_27 | Score: 0.68
[❌ FAIL - FP] Clip: clip_29 | Score: 0.62
[❌ FAIL - FP] Clip: clip_33 | Score: 0.70
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[❌ FAIL - FP] Clip: clip_32 | Score: 0.65
[✅ CORRECT - TN] Clip: clip_25 | Score: 0.32
[❌ FAIL - FP] Clip: clip_22 | Score: 0.65
[❌ FAIL - FP] Clip: clip_40 | Score: 0.63
[✅ CORRECT - TN] Clip: clip_23 | Score: 0.35
[❌ FAIL - FP] Clip: clip_24 | Score: 0.67
[✅ CORRECT - TN] Clip: clip_37 | Score: 0.56
[❌ FAIL - FP] Clip: clip_30 | Score: 0.72
[❌ FAIL - FP] Clip: clip_39 | Score: 0.73
[✅ CORRECT - TN] Clip: clip_38 | Score: 0.48
[❌ FAIL - FP] Clip: clip_31 | Score: 0.70
[❌ FAIL - FP] Clip: clip_36 | Score: 0.62

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.72
[❌ FAIL - FN] Clip: clip_17 | Score: 0.33
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.78
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.74
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.75
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.89
[❌ FAIL - FN] Clip: clip_29 | Score: 0.49
[❌ FAIL - FN] Clip: clip_16 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_5 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.76
[❌ FAIL - FN] Clip: clip_35 | Score: 0.44
[❌ FAIL - FN] Clip: clip_32 | Score: 0.48
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.67
[✅ CORRECT - TP] Clip: clip_13 | Score: 0.72
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.45
[✅ CORRECT - TP] Clip: clip_40 | Score: 0.61
[❌ FAIL - FN] Clip: clip_23 | Score: 0.46
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_12 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.61
[❌ FAIL - FN] Clip: clip_41 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.61
[❌ FAIL - FN] Clip: clip_0 | Score: 0.58
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.66
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.70
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.66
[❌ FAIL - FN] Clip: clip_39 | Score: 0.43
[❌ FAIL - FN] Clip: clip_38 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.61

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  61.29%
Precision: 71.43%
Recall:    71.43%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3489
Avg Latency per Frame  : 21.58 ms
Estimated Real-Time FPS: 46.3 FPS

No reach & loose tilt(20-120)

[✅ CORRECT - TN] Clip: clip_28 | Score: 0.58
[❌ FAIL - FP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TN] Clip: clip_26 | Score: 0.36
[❌ FAIL - FP] Clip: clip_27 | Score: 0.68
[❌ FAIL - FP] Clip: clip_29 | Score: 0.62
[❌ FAIL - FP] Clip: clip_33 | Score: 0.70
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[❌ FAIL - FP] Clip: clip_32 | Score: 0.65
[✅ CORRECT - TN] Clip: clip_25 | Score: 0.32
[❌ FAIL - FP] Clip: clip_22 | Score: 0.65
[❌ FAIL - FP] Clip: clip_40 | Score: 0.63
[✅ CORRECT - TN] Clip: clip_23 | Score: 0.35
[❌ FAIL - FP] Clip: clip_24 | Score: 0.67
[✅ CORRECT - TN] Clip: clip_37 | Score: 0.56
[❌ FAIL - FP] Clip: clip_30 | Score: 0.72
[❌ FAIL - FP] Clip: clip_39 | Score: 0.73
[✅ CORRECT - TN] Clip: clip_38 | Score: 0.48
[❌ FAIL - FP] Clip: clip_31 | Score: 0.70
[❌ FAIL - FP] Clip: clip_36 | Score: 0.62

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.72
[❌ FAIL - FN] Clip: clip_17 | Score: 0.33
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.78
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.74
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.75
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.89
[❌ FAIL - FN] Clip: clip_29 | Score: 0.49
[❌ FAIL - FN] Clip: clip_16 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_5 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.76
[✅ CORRECT - TP] Clip: clip_35 | Score: 0.66
[❌ FAIL - FN] Clip: clip_32 | Score: 0.48
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.67
[✅ CORRECT - TP] Clip: clip_13 | Score: 0.72
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.47
[✅ CORRECT - TP] Clip: clip_40 | Score: 0.61
[❌ FAIL - FN] Clip: clip_23 | Score: 0.46
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_12 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.61
[❌ FAIL - FN] Clip: clip_41 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.61
[❌ FAIL - FN] Clip: clip_0 | Score: 0.59
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.66
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.70
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.66
[❌ FAIL - FN] Clip: clip_39 | Score: 0.43
[❌ FAIL - FN] Clip: clip_38 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.61

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  62.90%
Precision: 72.09%
Recall:    73.81%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3489
Avg Latency per Frame  : 21.72 ms
Estimated Real-Time FPS: 46.1 FPS

no reach + strict tilt

STABILITY_FRAMES = 5
MAX_WRIST_DRIFT = 0.08  
REACH_MIN = -999.0       
TILT_MIN = 60           
TILT_MAX = 120          
THUMB_MIN = 0       
CNN_THRESHOLD = 0.60

[✅ CORRECT - TN] Clip: clip_28 | Score: 0.58
[❌ FAIL - FP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TN] Clip: clip_26 | Score: 0.00
[❌ FAIL - FP] Clip: clip_27 | Score: 0.68
[❌ FAIL - FP] Clip: clip_29 | Score: 0.62
[❌ FAIL - FP] Clip: clip_33 | Score: 0.70
[✅ CORRECT - TN] Clip: clip_34 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_35 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_32 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FP] Clip: clip_22 | Score: 0.65
[❌ FAIL - FP] Clip: clip_40 | Score: 0.63
[✅ CORRECT - TN] Clip: clip_23 | Score: 0.00
[❌ FAIL - FP] Clip: clip_24 | Score: 0.61
[✅ CORRECT - TN] Clip: clip_37 | Score: 0.56
[❌ FAIL - FP] Clip: clip_30 | Score: 0.72
[✅ CORRECT - TN] Clip: clip_39 | Score: 0.00
[✅ CORRECT - TN] Clip: clip_38 | Score: 0.48
[❌ FAIL - FP] Clip: clip_31 | Score: 0.70
[✅ CORRECT - TN] Clip: clip_36 | Score: 0.35

Evaluating Category: HANDSHAKE
[✅ CORRECT - TP] Clip: clip_10 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_28 | Score: 0.72
[❌ FAIL - FN] Clip: clip_17 | Score: 0.33
[✅ CORRECT - TP] Clip: clip_21 | Score: 0.72
[✅ CORRECT - TP] Clip: clip_19 | Score: 0.78
[✅ CORRECT - TP] Clip: clip_26 | Score: 0.74
[✅ CORRECT - TP] Clip: clip_18 | Score: 0.75
[✅ CORRECT - TP] Clip: clip_27 | Score: 0.63
[✅ CORRECT - TP] Clip: clip_20 | Score: 0.89
[❌ FAIL - FN] Clip: clip_29 | Score: 0.49
[❌ FAIL - FN] Clip: clip_16 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_11 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_4 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_3 | Score: 0.60
[✅ CORRECT - TP] Clip: clip_2 | Score: 0.62
[❌ FAIL - FN] Clip: clip_5 | Score: 0.54
[✅ CORRECT - TP] Clip: clip_33 | Score: 0.62
[✅ CORRECT - TP] Clip: clip_34 | Score: 0.76
[❌ FAIL - FN] Clip: clip_35 | Score: 0.44
[❌ FAIL - FN] Clip: clip_32 | Score: 0.43
[✅ CORRECT - TP] Clip: clip_14 | Score: 0.67
[✅ CORRECT - TP] Clip: clip_13 | Score: 0.80
[❌ FAIL - FN] Clip: clip_25 | Score: 0.00
[❌ FAIL - FN] Clip: clip_22 | Score: 0.42
[✅ CORRECT - TP] Clip: clip_40 | Score: 0.61
[❌ FAIL - FN] Clip: clip_23 | Score: 0.44
[✅ CORRECT - TP] Clip: clip_24 | Score: 0.61
[❌ FAIL - FN] Clip: clip_12 | Score: 0.43
[✅ CORRECT - TP] Clip: clip_15 | Score: 0.61
[❌ FAIL - FN] Clip: clip_41 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_9 | Score: 0.61
[❌ FAIL - FN] Clip: clip_0 | Score: 0.59
[✅ CORRECT - TP] Clip: clip_7 | Score: 0.65
[✅ CORRECT - TP] Clip: clip_6 | Score: 0.66
[✅ CORRECT - TP] Clip: clip_1 | Score: 0.70
[✅ CORRECT - TP] Clip: clip_8 | Score: 0.73
[✅ CORRECT - TP] Clip: clip_37 | Score: 0.68
[✅ CORRECT - TP] Clip: clip_30 | Score: 0.66
[❌ FAIL - FN] Clip: clip_39 | Score: 0.26
[❌ FAIL - FN] Clip: clip_38 | Score: 0.52
[✅ CORRECT - TP] Clip: clip_31 | Score: 0.61
[✅ CORRECT - TP] Clip: clip_36 | Score: 0.61

==================================================
🏁 HYBRID BENCHMARK COMPLETE
==================================================
--- 📈 STATISTICAL METRICS ---
Total Clips Evaluated: 62
Accuracy:  62.90%
Precision: 75.68%
Recall:    66.67%

--- ⚡ HARDWARE METRICS ---
Total Frames Processed : 3844
Avg Latency per Frame  : 21.88 ms
Estimated Real-Time FPS: 45.7 FPS