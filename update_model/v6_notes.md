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
