import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# --- CONFIGURATION ---
VALIDATION_DIR = '../images/train_dataset_v2/' # Path to your dataset
MODEL_PATH = 'handshake_model.keras'
ALTITUDE_THRESH = 0.45
REACH_THRESH = 0.08

# --- SETUP ---
print("Loading model and MediaPipe...")
model = tf.keras.models.load_model(MODEL_PATH)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Containers for results
y_true = []
y_pred_cnn = []   # CNN Only
y_pred_sbf = []   # CNN + SBF Logic

# --- PROCESSING LOOP ---
class_names = ['None', 'Handshake'] # Assuming folder structure: 0=None, 1=Handshake

for label_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(VALIDATION_DIR, class_name)
    if not os.path.exists(class_dir):
        print(f"Warning: Directory not found {class_dir}")
        continue
        
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        image = cv2.imread(img_path)
        if image is None: continue

        # 1. GROUND TRUTH
        y_true.append(label_idx)

        # 2. CNN PREDICTION (Standard Preprocessing)
        img_resized = cv2.resize(image, (224, 224))
        img_input = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized.astype(np.float32))
        img_input = np.expand_dims(img_input, axis=0)
        
        cnn_prob = model.predict(img_input, verbose=0)[0][0]
        cnn_pred = 1 if cnn_prob > 0.5 else 0
        y_pred_cnn.append(cnn_pred)

        # 3. SBF LOGIC CHECK (Run MediaPipe on same image)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        sbf_verified = False
        
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            
            # --- THE SBF MATH ---
            # Altitude Check (y_wrist > 0.45 means below shoulder)
            wrist_y = lm[0].y
            altitude_pass = wrist_y > ALTITUDE_THRESH
            
            # Reach Check (Z-Vector)
            # Note: Z is relative in MediaPipe, but works for "extension" logic
            wrist_z = lm[0].z
            middle_tip_z = lm[12].z
            reach_z = abs(wrist_z - middle_tip_z)
            reach_pass = reach_z > REACH_THRESH
            
            if altitude_pass and reach_pass:
                sbf_verified = True
        
        # FINAL LOGIC: It is ONLY a handshake if CNN says Yes AND SBF says Yes
        final_pred = 1 if (cnn_pred == 1 and sbf_verified) else 0
        y_pred_sbf.append(final_pred)

print("Processing Complete. Generating Comparison Plot...")

# --- PLOTTING ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: CNN Only
cm_cnn = confusion_matrix(y_true, y_pred_cnn)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Reds', ax=axes[0], cbar=False,
            xticklabels=class_names, yticklabels=class_names)
axes[0].set_title('Baseline: CNN Only')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Plot 2: CNN + SBF
cm_sbf = confusion_matrix(y_true, y_pred_sbf)
sns.heatmap(cm_sbf, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
            xticklabels=class_names, yticklabels=class_names)
axes[1].set_title('Ours: Neuro-Symbolic (CNN + SBF)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('') # Hide Y label for cleaner look

plt.tight_layout()
plt.savefig('comparison_matrix.png')
print("Saved comparison_matrix.png")