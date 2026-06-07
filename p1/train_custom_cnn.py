import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
from tensorflow.keras import layers, models

print("==================================================")
print("🧠 TRAINING CNN: FULL FRAMES + HEAVY AUGMENTATION")
print("==================================================")

# POINT THIS TO YOUR FULL, UNCROPPED, UNMASKED DATASET
DATASET_DIR = "../images/train_dataset_v3_cropped"  # <-- This should be the cropped dataset you just generated with build_v4_cropped_dataset.py
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

# --- SUBTLE AUGMENTATION FOR CROPPED HANDS ---
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(factor=0.02),                  # Tiny 7-degree rotation
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05), # Tiny 5% shift
    layers.RandomBrightness(factor=0.2),                 # Light lighting changes
    # Removed RandomZoom completely so we don't cut off fingers
])

# --- ARCHITECTURE ---
model = models.Sequential([
    layers.Input(shape=(160, 160, 1)),
    layers.Rescaling(1./127.5, offset=-1), # Replaces MobileNet preprocess_input
    
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

full_model = tf.keras.Sequential([
    layers.Input(shape=(160, 160, 1)),
    data_augmentation,
    model
])

full_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

print("\n🚀 Starting Training...")
history = full_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

print("\n==================================================")
print("📦 EXPORTING PURE FLOAT32 TFLITE MODEL...")
print("==================================================")
# --- 2. THE CONCRETE FUNCTION FIX ---
# This converts the Keras Sequential model into a low-level TF graph trace.
# It bypasses the Keras 3 '_get_save_spec' bug entirely.
run_model = tf.function(lambda x: model(x))

# Define the input signature: Batch=1, Height=160, Width=160, Channels=1
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 160, 160, 1], tf.float32)
)

# --- 3. CONVERSION ---
# Use 'from_concrete_functions' instead of 'from_keras_model'
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# Optimization for M1 Pro Performance
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]

try:
    tflite_model = converter.convert()
    
    # --- 4. SAVE ---
    TFLITE_OUTPUT = "custom_handshake_cnn.tflite"
    with open(TFLITE_OUTPUT, "wb") as f:
        f.write(tflite_model)
    
    print(f"✅ SUCCESS: Model saved to {TFLITE_OUTPUT}")
    print("You can now run benchmark_custom_cnn.py with this new model.")
except Exception as e:
    print(f"❌ EXPORT FAILED: {e}")

print("="*50)