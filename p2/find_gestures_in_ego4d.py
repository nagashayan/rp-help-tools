import json
import os
from collections import defaultdict

# Grouping variations into distinct categories for clean stats
CATEGORIES = {
    "Handshake": ["handshake", "shake hand", "shaking"],
    "Wave": ["wave", "waving"],
    "Fist Bump": ["fist bump"],
    "Pointing": ["point", "pointing"],
    "Object Handoff": ["hand off", "hands over", "handing"]
}

ANNOTATIONS_FILE = os.path.expanduser("~/ego4d_data/v2/annotations/narration.json") 

def get_category_stats():
    # This will map video_uid -> set of categories found in that video
    video_categories = defaultdict(set)
    
    print("Loading the narration file into memory... (approx 1-2 mins)")
    
    try:
        with open(ANNOTATIONS_FILE, 'r') as f:
            data = json.load(f)
            
        print(f"Loaded {len(data)} total videos. Scanning for category stats...")
        
        for video_uid, video_data in data.items():
            for pass_key, pass_val in video_data.items():
                if isinstance(pass_val, dict) and 'narrations' in pass_val:
                    for narration in pass_val['narrations']:
                        text = narration.get('narration_text', '')
                        if text:
                            text = text.lower()
                            
                            # Check text against all our categories
                            for cat_name, keywords in CATEGORIES.items():
                                if any(keyword in text for keyword in keywords):
                                    video_categories[video_uid].add(cat_name)
                                    
    except FileNotFoundError:
        print(f"ERROR: Could not find {ANNOTATIONS_FILE}")
        return

    # Calculate how many unique videos contain each category
    category_counts = {cat: 0 for cat in CATEGORIES}
    for uid, cats in video_categories.items():
        for cat in cats:
            category_counts[cat] += 1
            
    # Print the beautiful stats
    print("\n" + "="*40)
    print("CATEGORY-WISE VIDEO STATS:")
    print("="*40)
    for cat, count in category_counts.items():
        print(f"{cat.ljust(15)}: {count} unique videos")
    
    print("="*40)
    print(f"Total Unique Videos matching at least one category: {len(video_categories)}")
    
    # Still save the file just in case you want to use it
    with open("my_social_clips.txt", "w") as out_file:
        for uid in video_categories.keys():
            out_file.write(f"{uid}\n")

if __name__ == "__main__":
    get_category_stats()
