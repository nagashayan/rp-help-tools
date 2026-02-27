

uname -m # it must be aarch64

sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# 1. Create a folder for your project
mkdir handshake-project
cd handshake-project

# 2. Create the virtual environment (we will call it 'edge_env')
python3 -m venv .venv

# 3. Activate the environment
source .venv/bin/activate

pip install --upgrade pip
pip install tflite-runtime mediapipe opencv-python psutil

# Convert keras model to lite model

python convert_to_lite.py

To move files from mac to pi
hostname -I will tell local ip addres

```# Transfer the Python script
scp predict_cnn_pi.py handshake_model_optimized.tflite hand_landmarker.task nagashayanaramamurthy@10.0.0.49:~/handshake-project/

# Transfer the TFLite model
scp handshake_model_optimized.tflite pi@192.168.1.50:~/Desktop/handshake_project/

# Transfer the MediaPipe task file
scp hand_landmarker.task pi@192.168.1.50:~/Desktop/handshake_project/```

Pi and Mac should be on same Wifi, my keyboard Macally B3 is connected to Pi. Use hP mouse it's plug is plugged to Pi already.

How to Enable SSH on your Raspberry Pi
Method 1: The Terminal Way (Since you are already there)

Type this exact command into your Raspberry Pi terminal and hit Enter:

Bash
sudo raspi-config
A blue menu will pop up. Use your keyboard's arrow keys to scroll down and select 3 Interface Options (hit Enter).

Scroll down to I2 SSH and hit Enter.

It will ask: "Would you like the SSH server to be enabled?" Use the arrow keys to select <Yes> and hit Enter.

It will confirm SSH is enabled. Hit Enter, then use the Right Arrow key to select <Finish> at the bottom of the main menu to exit.

# Added 200 more images of my friend (Shyla)

python cnn_model_trainer.py
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
Found 398 images belonging to 2 classes.
Found 99 images belonging to 2 classes.

--- Starting Phase 1: Feature Extraction ---
/Users/nagashayanaramamurthy/GitHub/rp-help-tools/update_model/.venv/lib/python3.9/site-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
Epoch 1/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 9s 540ms/step - accuracy: 0.5448 - loss: 0.8834 - val_accuracy: 0.6162 - val_loss: 0.7306
Epoch 2/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 5s 415ms/step - accuracy: 0.7263 - loss: 0.5559 - val_accuracy: 0.5051 - val_loss: 0.7433
Epoch 3/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 5s 406ms/step - accuracy: 0.7608 - loss: 0.4420 - val_accuracy: 0.6061 - val_loss: 0.8471
Epoch 4/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 6s 448ms/step - accuracy: 0.7997 - loss: 0.4462 - val_accuracy: 0.5859 - val_loss: 0.7856
Epoch 5/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 6s 448ms/step - accuracy: 0.8434 - loss: 0.3533 - val_accuracy: 0.6465 - val_loss: 0.8228
Epoch 6/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 5s 379ms/step - accuracy: 0.8527 - loss: 0.3582 - val_accuracy: 0.5253 - val_loss: 0.8704

--- Starting Phase 2: Fine-Tuning ---
Epoch 1/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 10s 509ms/step - accuracy: 0.6751 - loss: 0.6257 - val_accuracy: 0.5859 - val_loss: 0.7248
Epoch 2/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 6s 430ms/step - accuracy: 0.6973 - loss: 0.5748 - val_accuracy: 0.5354 - val_loss: 0.7126
Epoch 3/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 6s 428ms/step - accuracy: 0.7559 - loss: 0.5417 - val_accuracy: 0.5758 - val_loss: 0.7219
Epoch 4/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 5s 388ms/step - accuracy: 0.7953 - loss: 0.4814 - val_accuracy: 0.5152 - val_loss: 0.7490
Epoch 5/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 5s 394ms/step - accuracy: 0.7999 - loss: 0.4653 - val_accuracy: 0.5960 - val_loss: 0.6919
Epoch 6/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 5s 378ms/step - accuracy: 0.8436 - loss: 0.4247 - val_accuracy: 0.5152 - val_loss: 0.7519
Epoch 7/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 5s 383ms/step - accuracy: 0.8093 - loss: 0.4221 - val_accuracy: 0.5960 - val_loss: 0.7416
Epoch 8/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 5s 388ms/step - accuracy: 0.8481 - loss: 0.3675 - val_accuracy: 0.5859 - val_loss: 0.7363
Epoch 9/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 6s 461ms/step - accuracy: 0.8195 - loss: 0.4169 - val_accuracy: 0.5556 - val_loss: 0.7616
Epoch 10/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 6s 432ms/step - accuracy: 0.8473 - loss: 0.3869 - val_accuracy: 0.5859 - val_loss: 0.7227
Epoch 11/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 6s 440ms/step - accuracy: 0.8562 - loss: 0.3503 - val_accuracy: 0.5556 - val_loss: 0.7742
Epoch 12/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 5s 408ms/step - accuracy: 0.8278 - loss: 0.3625 - val_accuracy: 0.5455 - val_loss: 0.7876
Epoch 13/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 5s 373ms/step - accuracy: 0.8776 - loss: 0.3436 - val_accuracy: 0.6061 - val_loss: 0.7478
Epoch 14/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 5s 379ms/step - accuracy: 0.8359 - loss: 0.3632 - val_accuracy: 0.6061 - val_loss: 0.7443
Epoch 15/15
13/13 ━━━━━━━━━━━━━━━━━━━━ 5s 377ms/step - accuracy: 0.8542 - loss: 0.3217 - val_accuracy: 0.6263 - val_loss: 0.7462
Model training complete and saved as handshake_model.keras!
Saved learning_curves.png

--- Running Final Evaluation ---
Found 497 images belonging to 2 classes.
16/16 ━━━━━━━━━━━━━━━━━━━━ 3s 171ms/step 
Precision:   0.7566
Recall:      0.7487
F1-Score:    0.7526
Specificity: 0.8497
Saved confusion_matrix.png