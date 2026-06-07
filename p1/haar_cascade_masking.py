import os
import cv2
import glob

print("==================================================")
print("⬛ SWEEP 2: HAAR CASCADE AGGRESSIVE MASKING")
print("==================================================")

# 1. Configuration: Read from the folder where MediaPipe just finished!
INPUT_DIR = "../images/train_dataset_v2_masked"
OUTPUT_DIR = "../images/train_dataset_v2_masked"
categories = ['none', 'handshake']

for cat in categories:
    os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)

# 2. Load Haar Cascades (Frontal AND Profile to catch weird chest-cam angles!)
frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

total_processed = 0
new_faces_masked = 0

for category in categories:
    print(f"\nProcessing category: {category.upper()}...")
    image_paths = glob.glob(os.path.join(INPUT_DIR, category, "*.jpg"))
    
    for img_path in image_paths:
        total_processed += 1
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # Haar Cascades require pure 1-channel grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 3. Detect Faces (minNeighbors=3 makes it aggressive)
        frontal_faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        profile_faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        # Combine all detections from both cascades
        all_faces = list(frontal_faces) + list(profile_faces)
        
        if len(all_faces) > 0:
            for (x, y, w, h) in all_faces:
                # Add 10% padding to the bounding box to cover the neck and hair
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                
                xmin = max(0, x - pad_x)
                ymin = max(0, y - pad_y)
                xmax = min(frame.shape[1], x + w + pad_x)
                ymax = min(frame.shape[0], y + h + pad_y)
                
                # Draw a solid black box over the new face
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), -1)
                new_faces_masked += 1
        
        # 4. Save the image (whether Haar touched it or not)
        filename = os.path.basename(img_path)
        out_path = os.path.join(OUTPUT_DIR, category, filename)
        cv2.imwrite(out_path, frame)

print("\n==================================================")
print("✅ Sweep 2 Complete!")
print(f"Analyzed {total_processed} images from MediaPipe's output.")
print(f"Haar Cascade found and masked {new_faces_masked} additional regions.")
print("==================================================")