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
  5849468592: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849467536: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849496032: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849497088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849494976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849495856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849531488: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849531312: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849555184: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849555360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849533072: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849533600: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849556944: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849556768: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849575840: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849576896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849558704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849575664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849578832: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849578480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849593632: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849594688: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849592752: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849593456: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849595744: TensorSpec(shape=(), dtype=tf.resource, name=None)
  5849604336: TensorSpec(shape=(), dtype=tf.resource, name=None)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1775562451.669633 4169949 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1775562451.669664 4169949 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
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
