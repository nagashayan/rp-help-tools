import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

print("\n==================================================")
print("🧠 PHASE 3: MOBILENET-V2 (HARD-NEGATIVE CROPS)")
print("==================================================")

# ==========================================
# 1. Data Loading & Augmentation
# ==========================================
# ONLY MobileNet scaling. No Blur! We need the high-freq textures.
def combined_preprocessing(img):
    return preprocess_input(img)

# Subtle augmentation since the images are already tightly cropped
datagen = ImageDataGenerator(
    preprocessing_function=combined_preprocessing,
    brightness_range=[0.8, 1.2],
    rotation_range=10,         # Subtle rotation
    width_shift_range=0.05,    # Subtle shift (keep fingers on screen!)
    height_shift_range=0.05,
    validation_split=0.2
)

data_dir = '../images/train_dataset_v5_cropped'

train_data = datagen.flow_from_directory(
    data_dir, target_size=(160, 160), batch_size=32,
    class_mode='binary', subset='training', shuffle=True
)

validation_data = datagen.flow_from_directory(
    data_dir, target_size=(160, 160), batch_size=32,
    class_mode='binary', subset='validation', shuffle=True
)

class_weights = compute_class_weight('balanced', classes=np.unique(train_data.classes), y=train_data.classes)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# ==========================================
# 2. Model Architecture
# ==========================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False  # Freeze for Phase 1

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'), # Shrunk to prevent memorization
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ==========================================
# 3. Phase 1: Feature Extraction
# ==========================================
print("\n--- Phase 1: Feature Extraction ---")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

history_phase1 = model.fit(
    train_data, validation_data=validation_data, epochs=15,
    class_weight=class_weight_dict, callbacks=[early_stop]
)

# ==========================================
# 4. Phase 2: Fine-Tuning 
# ==========================================
print("\n--- Phase 2: Fine-Tuning ---")
base_model.trainable = True

# Freeze bottom layers, train top 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Slower LR
              loss='binary_crossentropy', metrics=['accuracy'])

early_stop_ft = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

history_phase2 = model.fit(
    train_data, validation_data=validation_data, epochs=15,
    class_weight=class_weight_dict, callbacks=[early_stop_ft]
)

model.save("handshake_model.keras")
print("\n✅ Saved Keras model: handshake_model.keras")

# ==========================================
# 5. Graphing & Evaluation Metrics
# ==========================================
def plot_combined_learning_curves(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    epochs_range = range(len(acc))
    phase1_end = len(h1.history['accuracy']) - 1

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training', color='blue')
    plt.plot(epochs_range, val_acc, label='Validation', color='orange')
    plt.axvline(x=phase1_end, color='red', linestyle='--', label='Fine-tuning')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training', color='blue')
    plt.plot(epochs_range, val_loss, label='Validation', color='orange')
    plt.axvline(x=phase1_end, color='red', linestyle='--', label='Fine-tuning')
    plt.title('Loss')
    plt.legend()
    plt.savefig('learning_curves.png')

plot_combined_learning_curves(history_phase1, history_phase2)

print("\n--- Running Final Evaluation ---")
# CRITICAL FIX: Evaluate ONLY on the validation set, strictly NO SHUFFLE
eval_datagen = ImageDataGenerator(preprocessing_function=combined_preprocessing, validation_split=0.2)
evaluation_data = eval_datagen.flow_from_directory(
    data_dir, target_size=(160, 160), batch_size=32,
    class_mode='binary', subset='validation', shuffle=False 
)

Y_pred = model.predict(evaluation_data)
y_pred = (Y_pred > 0.5).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(evaluation_data.classes, y_pred, average='binary')
tn, fp, fn, tp = confusion_matrix(evaluation_data.classes, y_pred).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1-Score:    {f1:.4f}")
print(f"Specificity: {specificity:.4f}")

cm = confusion_matrix(evaluation_data.classes, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# ==========================================
# 6. TFLite Export (The LLVM Crash Workaround)
# ==========================================
print("\n📦 EXPORTING TO TFLITE...")
try:
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 160, 160, 3], tf.float32))

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open("mobilenet_handshake.tflite", 'wb') as f:
        f.write(tflite_model)
    print("✅ Success! MobileNet Edge model saved as mobilenet_handshake.tflite")
except Exception as e:
    print(f"❌ TFLite Export Failed (LLVM Error likely): {e}")
    print("Don't worry, the .keras model is saved. We can use a separate script to convert it.")
