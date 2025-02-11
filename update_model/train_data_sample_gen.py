from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Define the augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Load a sample image for demonstration
sample_image_path = '../images/train_dataset_v2/handshake/PXL_20241018_130346355.jpg'  # Replace with the path to one of your images
img = cv2.imread(sample_image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
img = cv2.resize(img, (224, 224))  # Resize to the target size used in training
img = np.expand_dims(img, axis=0)  # Add batch dimension
# Generate augmented images from the single input image
augmented_images = datagen.flow(img, batch_size=1)

# Plot a batch of augmented images
plt.figure(figsize=(10, 10))
for i in range(9):  # Display 9 augmented images
    augmented_image = next(augmented_images)[0].astype('uint8')
    plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image)
    plt.axis('off')
plt.tight_layout()
plt.show()
