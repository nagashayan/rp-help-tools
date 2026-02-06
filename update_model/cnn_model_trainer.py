import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import cv2

# New preprocessing wrapper
def combined_preprocessing(img):
    # Apply Gaussian Blur (simulating motion/webcam noise)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply MobileNetV2 scaling [-1, 1]
    return preprocess_input(img)

# Update the datagen
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

train_data = datagen.flow_from_directory(
    '../images/train_dataset_v2/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_data = datagen.flow_from_directory(
    '../images/train_dataset_v2/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Handle class imbalance
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# 2. Define and Fine-Tune MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # Freeze for initial training

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 3. Compile and Train
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Initial training also capture learning curve
history_phase1 = model.fit(train_data,
          validation_data=validation_data,
          epochs=15,
          class_weight=class_weight_dict)

# Fine-tune by unfreezing the base model
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_phase2 = model.fit(train_data,
          validation_data=validation_data,
          epochs=15,
          class_weight=class_weight_dict)

# Save model
model.save("handshake_model.keras")

print("Model training complete!")

import matplotlib.pyplot as plt
def plot_combined_learning_curves(h1, h2):
    # Combine metrics from both phases
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    
    epochs_range = range(len(acc))
    phase1_end = len(h1.history['accuracy']) - 1

    plt.figure(figsize=(15, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='blue')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='orange')
    plt.axvline(x=phase1_end, color='red', linestyle='--', label='Fine-tuning Start')
    plt.title('Accuracy: Initial Training vs. Fine-tuning')
    plt.xlabel('Total Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='blue')
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='orange')
    plt.axvline(x=phase1_end, color='red', linestyle='--', label='Fine-tuning Start')
    plt.title('Loss: Initial Training vs. Fine-tuning')
    plt.xlabel('Total Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_combined_learning_curves(history_phase1, history_phase2)
'''
This plot will show a clear "elbow" or jump at the red dashed line. This jump represents the model adapting its deep convolutional filters to the specific geometry of handshakes rather than general ImageNet features.Proof of Generalization: If the validation loss (orange line) stays close to the training loss during Phase 2, it proves your 0.15 spatial shifts and 0.5 Dropout are successfully preventing the model from just memorizing the background.Hyperparameter Justification: You can explain in your paper that the lower learning rate ($10^{-5}$) in Phase 2 was necessary to preserve the pre-trained knowledge while allowing for fine-grained adjustments.
'''
# More metrics

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf

# Load your latest model
model = tf.keras.models.load_model('handshake_model.keras')

# 1. Generate Predictions on Validation Data
# Ensure shuffle=False so labels match predictions
validation_data.shuffle = False
validation_data.reset()
Y_pred = model.predict(validation_data)
y_pred = (Y_pred > 0.5).astype(int)

# 2. Calculate Precision, Recall, F1, and Specificity
precision, recall, f1, _ = precision_recall_fscore_support(validation_data.classes, y_pred, average='binary')
tn, fp, fn, tp = confusion_matrix(validation_data.classes, y_pred).ravel()
specificity = tn / (tn + fp)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")

# 3. Plot Confusion Matrix for the Paper
cm = confusion_matrix(validation_data.classes, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['None', 'Handshake'], 
            yticklabels=['None', 'Handshake'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix: Handshake Detection')
plt.savefig('confusion_matrix.png')