import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load the model
# Make sure the file name matches exactly what you saved (e.g., 'handshake_model.keras')
model = tf.keras.models.load_model('handshake_model.keras')

# 2. Setup the data generator
# FIX: Added 'validation_split=0.2' so it knows to grab the 20% validation set
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2  # <--- THIS WAS MISSING
)

# 3. Load the validation data
validation_data = datagen.flow_from_directory(
    '../images/train_dataset_v2/', # Ensure this path is correct relative to where you run this script
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False # Crucial for confusion matrix!
)

# 4. Predict
print("Generating predictions...")
if validation_data.samples > 0:
    Y_pred = model.predict(validation_data)
    y_pred = (Y_pred > 0.5).astype(int)
    y_true = validation_data.classes

    # 5. Draw the Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Background', 'Handshake'],
                yticklabels=['Background', 'Handshake'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Success! Saved confusion_matrix.png")
else:
    print("Error: Still found 0 images. Check your path '../images/train_dataset_v2/'")