import cv2
import os
import glob

# 1. Load OpenCV's built-in Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Define your folders
input_folder = "../images/train_dataset_v2/handshake/"
output_folder = "../images/train_dataset_v2/unbiased/handshake/"

os.makedirs(output_folder, exist_ok=True)

# Helper function to resize while keeping the image from stretching
def resize_image(image, target_width=800):
    h, w = image.shape[:2]
    ratio = target_width / float(w)
    target_height = int(h * ratio)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

# 3. Loop through all images
for img_path in glob.glob(input_folder + "*.*"): # Catches .jpg, .png, etc.
    img = cv2.imread(img_path)
    if img is None: 
        continue
    
    # --- NEW: Standardize the image size first ---
    # This makes mobile photos and webcam photos exactly the same width
    img = resize_image(img, target_width=800)
    
    gray_1_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- STEP A: Detect and Blackout the Face ---
    faces = face_cascade.detectMultiScale(gray_1_channel, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        padding = 20
        start_point = (max(0, x - padding), max(0, y - padding))
        end_point = (min(img.shape[1], x + w + padding), min(img.shape[0], y + h + padding))
        
        cv2.rectangle(gray_1_channel, start_point, end_point, (0, 0, 0), -1)
            
    # --- STEP B: 3-Channel Duplication ---
    gray_3_channel = cv2.cvtColor(gray_1_channel, cv2.COLOR_GRAY2BGR)
            
    # Save the final unbiased image
    filename = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_folder, filename), gray_3_channel)
    print(f"Standardized, prepped, and saved: {filename}")

print("\nDataset perfectly standardized and prepped!")