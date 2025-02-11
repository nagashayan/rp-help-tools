import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load Data and Augment
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of data as validation
)

train_data = datagen.flow_from_directory(
    '../images/train_dataset_v2/',
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary',
    subset='training'
)

validation_data = datagen.flow_from_directory(
    '../images/train_dataset_v2/',
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary',
    subset='validation'
)

# 2. Model Definition
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # Freeze base model layers

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Train Model
model.fit(train_data, validation_data=validation_data, epochs=10)

model.save("handshake_model.keras")
