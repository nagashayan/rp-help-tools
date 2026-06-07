import cv2
import mediapipe as mp
import csv
import os
import glob

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       
    max_num_hands=2,               
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Build CSV Header dynamically
CSV_HEADER = ["video_name", "frame_number", "handedness"]
for i in range(21):
    CSV_HEADER.extend([f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"])

def mirror_landmarks(landmarks_list, handedness_label):
    """Flips the X axis to create a mirrored version of the hand"""
    mirrored_list = []
    for lm in landmarks_list:
        mirrored_list.append(1.0 - lm.x) # The mathematical flip
        mirrored_list.append(lm.y)
        mirrored_list.append(lm.z)
        
    mirrored_label = "Left" if handedness_label == "Right" else "Right"
    return mirrored_list, mirrored_label

def process_video(video_path, csv_writer):
    video_name = os.path.basename(video_path)
    print(f"Processing {video_name}...")
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break 
            
        frame_count += 1
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                # --- THE PHYSICAL FILTER ---
                # Calculate bounding box area to ensure hand is close to the camera
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                
                box_width = max(x_coords) - min(x_coords)
                box_height = max(y_coords) - min(y_coords)
                hand_area = box_width * box_height
                
                # If hand takes up less than 5% of the screen, it's a background person. Skip it!
                if hand_area < 0.05:
                    continue
                # ---------------------------

                label = handedness.classification[0].label # "Left" or "Right"
                
                # Extract original coordinates
                original_coords = []
                for lm in hand_landmarks.landmark:
                    original_coords.extend([lm.x, lm.y, lm.z])
                    
                # Write original frame data
                csv_writer.writerow([video_name, frame_count, label] + original_coords)
                
                # Create and write mirrored augmented data
                mirrored_coords, mirrored_label = mirror_landmarks(hand_landmarks.landmark, label)
                csv_writer.writerow([video_name, frame_count, mirrored_label + "_mirrored"] + mirrored_coords)
                    
    cap.release()

if __name__ == "__main__":
    # Point this to your new categorized pilot folder
    CATEGORIZED_DIR = os.path.expanduser("~/ego4d_data/categorized_pilot")
    OUTPUT_CSV = "pilot_dataset_features.csv"
    
    # Open CSV once and append all videos to it
    with open(OUTPUT_CSV, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_HEADER)
        
        # Look for all .mp4 files inside all subfolders
        search_pattern = os.path.join(CATEGORIZED_DIR, "**", "*.mp4")
        video_files = glob.glob(search_pattern, recursive=True)
        
        if not video_files:
            print(f"No MP4 files found in {CATEGORIZED_DIR}. Please check the path.")
        else:
            print(f"Found {len(video_files)} clips. Starting MediaPipe extraction...")
            for video in video_files:
                process_video(video, csv_writer)
                
            print(f"\nSUCCESS! All features cleanly extracted to {OUTPUT_CSV}")