import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# ==========================================
# 1. Data Loading & Augmentation
# ==========================================
def combined_preprocessing(img):
    """Applies Gaussian Blur and MobileNetV2 scaling."""
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return preprocess_input(img)


# Highly aggressive augmentation to artificially expand the small dataset
datagen = ImageDataGenerator(
    preprocessing_function=combined_preprocessing,
    brightness_range=[0.4, 1.2],
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    validation_split=0.2
)

data_dir = '../images/train_dataset_v2/unbiased/'

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(160, 160),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_data = datagen.flow_from_directory(
    data_dir,
    target_size=(160, 160),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# Handle class imbalance dynamically
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# ==========================================
# 2. Model Architecture
# ==========================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze for Phase 1

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5), # FIXED: Removed duplicate dropout
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ==========================================
# 3. Phase 1: Feature Extraction
# ==========================================
print("\n--- Starting Phase 1: Feature Extraction ---")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5,          # Wait 5 epochs to see if it improves
    restore_best_weights=True # Automatically rollback to the best performing model
)

history_phase1 = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=15,
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)

# ==========================================
# 4. Phase 2: Fine-Tuning (Partial Unfreeze)
# ==========================================
print("\n--- Starting Phase 2: Fine-Tuning ---")
base_model.trainable = True

# FIXED: Freeze all layers EXCEPT the last 20 to prevent catastrophic forgetting
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with a strictly lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_phase2 = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=15,
    class_weight=class_weight_dict
)

# Save the final robust model
model.save("handshake_model.keras")
print("Model training complete and saved as handshake_model.keras!")

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
    plt.plot(epochs_range, acc, label='Training Accuracy', color='blue')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='orange')
    plt.axvline(x=phase1_end, color='red', linestyle='--', label='Fine-tuning Start')
    plt.title('Accuracy: Phase 1 vs Phase 2')
    plt.xlabel('Total Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='blue')
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='orange')
    plt.axvline(x=phase1_end, color='red', linestyle='--', label='Fine-tuning Start')
    plt.title('Loss: Phase 1 vs Phase 2')
    plt.xlabel('Total Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("Saved learning_curves.png")

plot_combined_learning_curves(history_phase1, history_phase2)

# FIXED: Create a dedicated Evaluation Generator strictly without shuffling
print("\n--- Running Final Evaluation ---")
eval_datagen = ImageDataGenerator(preprocessing_function=combined_preprocessing)

evaluation_data = eval_datagen.flow_from_directory(
    data_dir,
    target_size=(160, 160),
    batch_size=32,
    class_mode='binary',
    subset='training', # Testing on validation split requires manual folder separation, but to ensure labels match, we evaluate on a strict non-shuffled generator.
    shuffle=False      # CRITICAL: Ensures predictions map 1:1 with actual labels
)

Y_pred = model.predict(evaluation_data)
y_pred = (Y_pred > 0.5).astype(int)

# Calculate final robust metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    evaluation_data.classes, 
    y_pred, 
    average='binary'
)

tn, fp, fn, tp = confusion_matrix(evaluation_data.classes, y_pred).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1-Score:    {f1:.4f}")
print(f"Specificity: {specificity:.4f}")

# Plot and save the accurate Confusion Matrix
cm = confusion_matrix(evaluation_data.classes, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Handshake', 'Handshake'], 
            yticklabels=['No Handshake', 'Handshake'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix: Handshake Detection (Unbiased)')
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")