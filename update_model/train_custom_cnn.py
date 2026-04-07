import tensorflow as tf
from tensorflow.keras import layers, models
import os

print("==================================================")
print("🧠 TRAINING CUSTOM CNN V2 (BATCH NORMALIZED)")
print("==================================================")

DATASET_DIR = "../images/train_dataset_v2" 
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 15
CLASSES = ['none', 'handshake']

print("Loading dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="training", seed=123,
    class_names=CLASSES, color_mode="rgb", image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation", seed=123,
    class_names=CLASSES, color_mode="rgb", image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

def to_grayscale(image, label):
    return tf.image.rgb_to_grayscale(image), label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(to_grayscale, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(to_grayscale, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomZoom(0.1),
  layers.RandomTranslation(0.1, 0.1),
])

# --- UPDATED ARCHITECTURE: Added Batch Normalization ---
model = models.Sequential([
    tf.keras.Input(shape=(160, 160, 1)),
    data_augmentation,
    layers.Rescaling(1./127.5, offset=-1),
    
    layers.Conv2D(16, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(32, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    
    layers.Flatten(),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

print("\n🚀 Starting Training...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

print("\n==================================================")
print("📦 EXPORTING PURE FLOAT32 TFLITE MODEL...")
print("==================================================")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# FIX: We completely removed converter.optimizations to prevent weight crushing!
tflite_model = converter.convert()

output_path = "custom_handshake_cnn.tflite"
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Success! V2 Edge model saved as: {output_path}")
print(f"File Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")