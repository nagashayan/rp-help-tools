import cv2
import os

cap = cv2.VideoCapture(0)

base_dir = "../images/sequence_dataset"
label = "handshake"  # default
recording = False
clip_id = 1
frame_count = 0

# Control frame saving rate
save_interval = 0.1  # seconds between saved frames (~20 FPS)
last_save_time = 0

print("Press:")
print("h → handshake mode")
print("n → none mode")
print("r → start recording")
print("s → stop recording")
print("q → quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Recorder", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('h'):
        label = "handshake"
        print("Label set to handshake")

    elif key == ord('n'):
        label = "none"
        print("Label set to none")

    elif key == ord('r'):
        recording = True
        frame_count = 0

        clip_path = os.path.join(base_dir, label, f"clip_{clip_id}")
        os.makedirs(clip_path, exist_ok=True)

        print(f"Recording started: {clip_path}")

    elif key == ord('s'):
        recording = False
        clip_id += 1
        print("Recording stopped")

    elif key == ord('q'):
        break

    if recording:
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if current_time - last_save_time >= save_interval:
            frame_path = os.path.join(
                base_dir,
                label,
                f"clip_{clip_id}",
                f"frame_{frame_count:03d}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            last_save_time = current_time

cap.release()
cv2.destroyAllWindows()