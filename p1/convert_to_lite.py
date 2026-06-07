"""
Docstring for update_model.convert_to_lite

This converts your newly trained Keras model into a TFLite format optimized for Raspberry Pi deployment.

"""
import tensorflow as tf

# 1. Load your newly trained Keras model
model = tf.keras.models.load_model("handshake_model.keras")

# 2. Initialize the TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Apply Edge Optimizations (Crucial for the Raspberry Pi)
# This compresses the model weights, making it run much faster and cooler
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Convert and Save
tflite_model = converter.convert()

with open("handshake_model_optimized.tflite", "wb") as f:
    f.write(tflite_model)

print("Successfully created highly optimized TFLite model for Raspberry Pi!")