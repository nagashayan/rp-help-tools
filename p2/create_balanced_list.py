import json
import os

CATEGORIES = {
    "Handshake": ["handshake", "shake hand", "shaking"],
    "Wave": ["wave", "waving"],
    "Fist_Bump": ["fist bump"],
    "Pointing": ["point", "pointing"],
    "Object_Handoff": ["hand off", "hands over", "handing"]
}

ANNOTATIONS_FILE = os.path.expanduser("~/ego4d_data/v2/annotations/narration.json") 
PILOT_SIZE = 5

def is_valid_egocentric_action(text, keywords):
    text = text.lower()
    
    # 1. Does the text contain the target gesture?
    if not any(kw in text for kw in keywords):
        return False
        
    # 2. Is the camera wearer (#C C) actively involved?
    if "#c c" not in text:
        return False
        
    # 3. Ban "voyeur" words (watching third parties)
    banned_words = ["watches", "sees", "observes", "looks at", "notices", "watching"]
    if any(banned in text for banned in banned_words):
        return False
        
    return True

def create_pilot_lists():
    print("Finding the strictly first-person pilot videos...")
    
    try:
        with open(ANNOTATIONS_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Could not find {ANNOTATIONS_FILE}")
        return
            
    used_uids = set()
    
    for cat_name, keywords in CATEGORIES.items():
        cat_uids = []
        
        for video_uid, video_data in data.items():
            if len(cat_uids) >= PILOT_SIZE:
                break 
                
            if video_uid in used_uids:
                continue 
                
            found = False
            for pass_key, pass_val in video_data.items():
                if isinstance(pass_val, dict) and 'narrations' in pass_val:
                    for narration in pass_val['narrations']:
                        text = narration.get('narration_text', '')
                        
                        if is_valid_egocentric_action(text, keywords):
                            cat_uids.append(video_uid)
                            used_uids.add(video_uid)
                            found = True
                            break
                if found: break
                
        # Save a text file just for this category
        filename = f"pilot_{cat_name.lower()}.txt"
        with open(filename, "w") as out_file:
            for uid in cat_uids:
                out_file.write(f"{uid}\n")
                
        print(f"Saved {len(cat_uids)} UIDs to {filename}")

if __name__ == "__main__":
    create_pilot_lists()