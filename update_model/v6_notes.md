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



4. what if I use latest mediapipe and try to tune geometry to see both hands and pick one near by? can that solve adversal dataset problem?

