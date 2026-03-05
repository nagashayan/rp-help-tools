import cv2
import os

# Set up your native edge resolution!
CAM_WIDTH, CAM_HEIGHT = 640, 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# Create folders
BASE_DIR = "../images/sequence_datasetv3"
os.makedirs(f"{BASE_DIR}/handshake", exist_ok=True)
os.makedirs(f"{BASE_DIR}/none", exist_ok=True)

print("🎥 DATASET RECORDER READY 🎥")
print("Press 'h' to record a 3-second HANDSHAKE clip.")
print("Press 'n' to record a 3-second NONE clip.")
print("Press 'q' to QUIT.")

clip_counter = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    cv2.imshow("Recorder (640x480)", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key in [ord('h'), ord('n')]:
        category = "handshake" if key == ord('h') else "none"
        clip_dir = f"{BASE_DIR}/{category}/clip_{clip_counter}"
        os.makedirs(clip_dir, exist_ok=True)
        
        print(f"🔴 RECORDING {category.upper()} (Clip {clip_counter})...")
        
        # Record roughly 3 seconds of video (~90 frames at 30fps)
        for i in range(90):
            ret, rec_frame = cap.read()
            if ret:
                cv2.imwrite(f"{clip_dir}/frame_{i:04d}.jpg", rec_frame)
                
                # Visual indicator
                cv2.circle(rec_frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.imshow("Recorder (640x480)", rec_frame)
                cv2.waitKey(33) # roughly 30fps wait
                
        print(f"✅ Saved to {clip_dir}")
        clip_counter += 1

cap.release()
cv2.destroyAllWindows()