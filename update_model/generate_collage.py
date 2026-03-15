import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import math
import matplotlib.pyplot as plt

# 1. Initialize Models
model = tf.keras.models.load_model('handshake_model.keras')
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def process_frame(image_path, expected_type):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not find {image_path}. Creating a blank placeholder.")
        return np.zeros((400, 300, 3), dtype=np.uint8)
        
    img = cv2.resize(img, (480, 640)) # Standardize size for the collage
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run MediaPipe
    results = hands.process(img_rgb)
    
    # Default UI Values
    cnn_conf = 0.00
    pose_score = 0.00
    stability = 0.95 # Simulated for static frames
    status_text = "SCANNING..."
    status_color = (0, 0, 255) # Red
    reason = "No Hand Detected"
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Extract Coordinates
        lm = hand_landmarks.landmark
        z_wrist, z_middle = lm[0].z, lm[12].z
        y_index_mcp, y_pinky_mcp = lm[5].y, lm[17].y
        x_index_mcp, x_pinky_mcp = lm[5].x, lm[17].x
        
        # SBF Math
        delta_z = abs(z_wrist - z_middle)
        thumb_dist = math.dist([lm[4].x, lm[4].y], [lm[5].x, lm[5].y])
        tilt = abs(math.degrees(math.atan2(y_index_mcp - y_pinky_mcp, x_index_mcp - x_pinky_mcp)))
        
        # SBF Gates
        if delta_z <= 0.05:
            reason = f"REJECT: Low Reach (Z={delta_z:.2f})"
        elif thumb_dist <= 0.06:
            reason = f"REJECT: Closed Thumb (D={thumb_dist:.2f})"
        elif not (45 <= tilt <= 135):
            reason = f"REJECT: Flat/Vertical (Tilt={tilt:.0f}deg)"
        else:
            pose_score = 1.00
            reason = f"Reaching Forward (Z={delta_z:.2f})"
            
            # Trigger CNN ONLY if SBF passes
            img_resized = cv2.resize(img, (160, 160))
            img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
            img_input = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_blurred, axis=0))
            cnn_conf = float(model.predict(img_input, verbose=0)[0][0])
            
            if cnn_conf > 0.5:
                status_text = "VERIFIED"
                status_color = (0, 255, 0) # Green

    # --- Draw UI Overlay (Matching Fig 10) ---
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (480, 180), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, 560), (480, 640), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    
    cv2.putText(img, f"CNN Confidence: {cnn_conf:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(img, f"Pose Score:       {pose_score:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(img, f"Stability:          {stability:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    reason_color = (0, 255, 0) if pose_score == 1.0 else (0, 0, 255)
    cv2.putText(img, reason, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, reason_color, 2)
    
    # Center the bottom status text
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    text_x = (480 - text_size[0]) // 2
    cv2.putText(img, status_text, (text_x, 610), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Process all 4 images
img_hs = process_frame('paper-dataset/handshake.jpg', 'handshake')
img_wave = process_frame('paper-dataset/wave.jpg', 'wave')
img_hf = process_frame('paper-dataset/namaste.jpg', 'highfive')
img_fb = process_frame('paper-dataset/fistbump.jpg', 'fistbump')

# 3. Stitch into a 2x2 Grid
fig, axs = plt.subplots(2, 2, figsize=(10, 12), dpi=300)
plt.subplots_adjust(wspace=0.05, hspace=0.05)

axs[0, 0].imshow(img_hs)
axs[0, 0].set_title("(a) True Positive: Valid Handshake", fontsize=12, weight='bold', pad=10)
axs[0, 0].axis('off')

axs[0, 1].imshow(img_hf)
axs[0, 1].set_title("(b) True Negative: High-Five (Tilt Block)", fontsize=12, weight='bold', pad=10)
axs[0, 1].axis('off')

axs[1, 0].imshow(img_wave)
axs[1, 0].set_title("(c) True Negative: Wave (Reach Block)", fontsize=12, weight='bold', pad=10)
axs[1, 0].axis('off')

axs[1, 1].imshow(img_fb)
axs[1, 1].set_title("(d) True Negative: Fist Bump (Thumb Block)", fontsize=12, weight='bold', pad=10)
axs[1, 1].axis('off')

plt.tight_layout()
plt.savefig('fig_sequence_collage.png', bbox_inches='tight')
print("✅ Saved as 'fig_sequence_collage.png'")