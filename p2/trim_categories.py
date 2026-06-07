import json
import os
import subprocess

# Grouping variations into distinct categories
CATEGORIES = {
    "Handshake": ["handshake", "shake hand", "shaking"],
    "Wave": ["wave", "waving"],
    "Fist_Bump": ["fist bump"],
    "Pointing": ["point", "pointing"],
    "Object_Handoff": ["hand off", "hands over", "handing"]
}

def trim_video_clip(input_path, output_path, start_time, duration=4):
    """Uses FFmpeg to re-encode a clean, frame-accurate slice of video"""
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(max(0, start_time)), # Start time 
        '-i', input_path,               # Input file
        '-t', str(duration),            # Duration of clip
        '-c:v', 'libx264',              # Force re-encoding with H.264
        '-preset', 'fast',              # Speed up the encoding
        '-crf', '23',                   # High quality setting
        '-c:a', 'aac',                  # Keep audio safe
        output_path
    ]
    
    # Run the command silently
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def categorize_and_trim(raw_dir, output_dir, annotations_file):
    print("Loading annotations...")
    with open(annotations_file, 'r') as f:
        data = json.load(f)
        
    # Create the category directories if they don't exist
    for cat_name in CATEGORIES.keys():
        os.makedirs(os.path.join(output_dir, cat_name), exist_ok=True)
        
    print("Scanning raw pilot folder and cutting clips...")
    
    # Loop through the files actually present in your raw pilot folder
    video_files = [f for f in os.listdir(raw_dir) if f.endswith('.mp4')]
    
    clip_counts = {cat: 0 for cat in CATEGORIES.keys()}
    
    for filename in video_files:
        video_uid = filename.replace('.mp4', '')
        input_video_path = os.path.join(raw_dir, filename)
        
        if video_uid not in data:
            continue
            
        video_data = data[video_uid]
        
        # Parse through the narrations to find the timestamps
        for pass_key, pass_val in video_data.items():
            if isinstance(pass_val, dict) and 'narrations' in pass_val:
                for narration in pass_val['narrations']:
                    text = narration.get('narration_text', '').lower()
                    timestamp = narration.get('timestamp_sec')
                    
                    if timestamp is None:
                        continue
                        
                    # Check which category this narration belongs to
                    for cat_name, keywords in CATEGORIES.items():
                        if any(keyword in text for keyword in keywords):
                            
                            # Center the 4-second clip around the timestamp (2 seconds before, 2 seconds after)
                            start_time = timestamp - 2
                            
                            clip_name = f"{video_uid}_frame_{narration.get('annotation_uid', clip_counts[cat_name])}.mp4"
                            output_video_path = os.path.join(output_dir, cat_name, clip_name)
                            
                            # Trim!
                            trim_video_clip(input_video_path, output_video_path, start_time, duration=4)
                            clip_counts[cat_name] += 1
                            
    print("\n" + "="*40)
    print("TRIMMING COMPLETE! Here is your clean dataset:")
    print("="*40)
    for cat, count in clip_counts.items():
        print(f"Folder: {cat.ljust(15)} -> Extracted {count} sub-clips")
    print("="*40)

if __name__ == "__main__":
    # Update these paths to match where your files are located!
    RAW_DIR = os.path.expanduser("~/ego4d_data/raw_pilot_videos/v2/video_540ss")
    OUTPUT_DIR = os.path.expanduser("~/ego4d_data/categorized_pilot")
    ANNOTATIONS_FILE = os.path.expanduser("~/ego4d_data/v2/annotations/narration.json")
    
    categorize_and_trim(RAW_DIR, OUTPUT_DIR, ANNOTATIONS_FILE)