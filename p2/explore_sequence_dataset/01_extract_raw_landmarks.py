"""
=============================================================================
Script 1: Raw MediaPipe Landmark Extractor
=============================================================================
Purpose:
    This script serves as the foundational data extraction step (The "Data Vault").
    It scans a directory of egocentric video frames (organized by class folders), 
    processes each frame through MediaPipe Hands to extract the 21 3D spatial 
    landmarks (X, Y, Z coordinates), and outputs a consolidated CSV file.
    
Inputs:
    - A base directory containing class subfolders (e.g., 'handshake/', 'none/'): "/Users/nagashayanaramamurthy/GitHub/rp-help-tools/images/p1_dataset_combined"
      where each subfolder contains video sequence folders of raw .jpg/.png frames.
      
Outputs:
    - 'p1_dataset_combined_raw.csv': A 66-column dataset containing the raw X,Y,Z 
      coordinates for every frame, alongside tracking metadata (video_name, 
      frame_number) and a binary ground-truth label (1 = handshake, 0 = none).
      
Note:
    If MediaPipe fails to detect a hand in a given frame, the script safely 
    pads the row with zeros (0.0) to preserve the temporal sequence length 
    required for downstream Time-Series architectures (Temporal CNN / GRU).
=============================================================================
"""

import cv2
import mediapipe as mp
import csv
import os
import glob
import re

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

CSV_HEADER = ["video_name", "frame_number", "label"]
for i in range(21):
    CSV_HEADER.extend([f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"])

def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

def extract_landmarks_from_frames(base_dir, output_csv):
    print(f"Scanning directory: {base_dir}")
    
    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_HEADER)
        
        for class_name in os.listdir(base_dir):
            class_path = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_path): continue
                
            label = 1 if class_name.lower() == "handshake" else 0
            
            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)
                if not os.path.isdir(video_path): continue
                
                # FIX: Create a globally unique video name to prevent overwriting!
                unique_video_name = f"{class_name}_{video_folder}"
                print(f"Processing Sequence: {unique_video_name}")
                
                frame_files = glob.glob(os.path.join(video_path, "*.jpg")) + glob.glob(os.path.join(video_path, "*.png"))
                frame_files.sort(key=numerical_sort)
                
                for frame_idx, frame_path in enumerate(frame_files):
                    image = cv2.imread(frame_path)
                    if image is None: continue
                        
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)
                    
                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        row_data = [unique_video_name, frame_idx, label]
                        for lm in hand_landmarks.landmark:
                            row_data.extend([lm.x, lm.y, lm.z])
                        csv_writer.writerow(row_data)
                    else:
                        row_data = [unique_video_name, frame_idx, label] + [0.0] * (21 * 3)
                        csv_writer.writerow(row_data)

    print(f"\nSUCCESS! Raw extraction complete. Saved to {output_csv}")

if __name__ == "__main__":
    BASE_FRAMES_DIR = "/Users/nagashayanaramamurthy/GitHub/rp-help-tools/images/p1_dataset_combined" 
    OUTPUT_CSV_NAME = "p1_dataset_combined_raw.csv"
    if os.path.exists(BASE_FRAMES_DIR):
        extract_landmarks_from_frames(BASE_FRAMES_DIR, OUTPUT_CSV_NAME)
    else:
        print(f"ERROR: Could not find '{BASE_FRAMES_DIR}'")
