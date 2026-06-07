import tensorflow as tf
import numpy as np
import cv2
import glob

print("🚀 Starting INT8 Quantization Process...")

# 1. Load your fully trained Keras model
model = tf.keras.models.load_model('handshake_model.keras')

# 2. Initialize the TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Turn on the optimization flag
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Create a Representative Dataset Generator
# This feeds 50 sample images through the converter to calibrate the 8-bit math
def representative_dataset():
    # Grab the first 50 images from any of your sequence clips
    image_paths = glob.glob("sequence_dataset/*/*/*.jpg")[:50] 
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is None: continue
        
        # Exact same preprocessing used on the Raspberry Pi
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_1c = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_3c = cv2.cvtColor(gray_1c, cv2.COLOR_GRAY2RGB)
        
        img = cv2.resize(gray_3c, (224, 224))
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Float32 Normalization (-1.0 to 1.0)
        img = img.astype(np.float32)
        img /= 127.5
        img -= 1.0
        
        # Yield the image back to the converter
        yield [np.expand_dims(img, axis=0)]

# Attach the dataset to the converter
converter.representative_dataset = representative_dataset

# 5. Restrict operations strictly to INT8 (Full Integer Quantization)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# 6. Float Fallback: Keep Inputs/Outputs as Float32 so the Pi script doesn't break
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

# 7. Convert and Save!
print("⚙️ Calibrating weights... (This may take a minute)")
tflite_quant_model = converter.convert()

with open('handshake_model_int8.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("✅ Success! Model saved as 'handshake_model_int8.tflite'")