import os
import cv2
import glob
from mtcnn import MTCNN

print("==================================================")
print("🤖 SWEEP: MTCNN DEEP LEARNING MASKING")
print("==================================================")

# Point this to your unmasked grayscale or original RGB folder
INPUT_DIR = "../images/train_dataset_v2_gray"
OUTPUT_DIR = "../images/train_dataset_v2_mtcnn"
categories = ['none', 'handshake']

for cat in categories:
    os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)

# Initialize the Deep Learning Face Detector
detector = MTCNN()

total_processed = 0
faces_masked = 0

for category in categories:
    print(f"\nProcessing category: {category.upper()}...")
    image_paths = glob.glob(os.path.join(INPUT_DIR, category, "*.jpg"))
    
    for img_path in image_paths:
        total_processed += 1
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # MTCNN explicitly requires RGB format to work properly
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(frame_rgb)
        
        if faces:
            for face in faces:
                # MTCNN gives us a confidence score!
                if face['confidence'] > 0.70:
                    x, y, w, h = face['box']
                    
                    # Add 10% padding
                    pad_x = int(w * 0.1)
                    pad_y = int(h * 0.1)
                    
                    xmin = max(0, x - pad_x)
                    ymin = max(0, y - pad_y)
                    xmax = min(frame.shape[1], x + w + pad_x)
                    ymax = min(frame.shape[0], y + h + pad_y)
                    
                    # Draw solid black box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), -1)
                    faces_masked += 1
        
        # Save the image
        filename = os.path.basename(img_path)
        out_path = os.path.join(OUTPUT_DIR, category, f"mtcnn_{filename}")
        cv2.imwrite(out_path, frame)

print("\n==================================================")
print("✅ MTCNN Masking Complete!")
print(f"Analyzed {total_processed} total images.")
print(f"Successfully applied deep learning masks to {faces_masked} faces.")
print("==================================================")