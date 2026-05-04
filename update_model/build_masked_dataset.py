import os
import cv2
import glob
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("==================================================")
print("⬛ BUILDING MASKED DATASET (TASKS API FACE BLACKOUT)")
print("==================================================")

INPUT_DIR = "../images/train_dataset_v2_gray"
OUTPUT_DIR = "../images/train_dataset_v2_masked"
categories = ['none', 'handshake']

# 1. Create output directories
for cat in categories:
    os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)

# 2. Auto-Download the Face Detection Model if missing
model_path = 'blaze_face_short_range.tflite'
if not os.path.exists(model_path):
    print("Downloading Google's Face Detection model...")
    url = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 3. Initialize the Modern Face Detector
base_options = python.BaseOptions(model_asset_path=model_path)
# min_detection_confidence=0.7 prevents random black boxes on walls/shirts
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.4)
detector = vision.FaceDetector.create_from_options(options)

total_processed = 0
faces_masked = 0

for category in categories:
    print(f"\nProcessing category: {category.upper()}...")
    image_paths = glob.glob(os.path.join(INPUT_DIR, category, "*.jpg"))
    
    for img_path in image_paths:
        total_processed += 1
        
        # Read the image
        frame = cv2.imread(img_path)
        if frame is None: continue
        h, w, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect Faces
        detection_result = detector.detect(mp_image)
        
        # 4. If a face is found, mask it!
        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                
                # The modern API gives us absolute pixel coordinates directly
                xmin = bbox.origin_x
                ymin = bbox.origin_y
                box_w = bbox.width
                box_h = bbox.height
                
                # Add 10% padding to cover hair and chin
                pad_x = int(box_w * 0.1)
                pad_y = int(box_h * 0.1)
                
                start_x = max(0, xmin - pad_x)
                start_y = max(0, ymin - pad_y)
                end_x = min(w, xmin + box_w + pad_x)
                end_y = min(h, ymin + box_h + pad_y)
                
                # Draw a solid black box
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
                faces_masked += 1
        
        # 5. Save the image
        filename = os.path.basename(img_path)
        out_path = os.path.join(OUTPUT_DIR, category, f"masked_{filename}")
        cv2.imwrite(out_path, frame)

print("\n==================================================")
print(f"✅ Smart Masking Complete!")
print(f"Analyzed {total_processed} total images.")
print(f"Successfully applied precise black-out masks to {faces_masked} faces.")
print("==================================================")