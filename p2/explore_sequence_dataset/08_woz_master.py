"""
=============================================================================
Script 8: Wizard of Oz Master Server (Reality vs. Expected Mode)
=============================================================================
Purpose:
    Executes the true Wizard of Oz architecture. 
    
    *Architectural Update:* Added a manual override system. Press 'm' to 
    toggle between AI inference (Reality) and manual Spacebar triggers 
    (Expected) for pristine HCI testing. The AI continues to calculate 
    probabilities in the background for research logging.
=============================================================================
"""

import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from flask import Flask, jsonify, send_file
import threading
import socket
import numpy as np
import os
import logging
import time

# ==========================================
# 1. Setup PyTorch Architecture
# ==========================================
class TemporalCNN(nn.Module):
    def __init__(self, num_features=63, num_classes=2, window_size=30):
        super(TemporalCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        flattened_length = window_size // 4
        self.fc_input_size = 64 * flattened_length
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.conv_block(x)
        return self.classifier(features)

model = TemporalCNN()
model.load_state_dict(torch.load("temporal_cnn_raw.pth", map_location='cpu'))
model.eval()

# ==========================================
# 2. Global State Variables
# ==========================================
current_state = "IDLE"
current_prob = 0.0
operating_mode = "AI_MODE" # Can be "AI_MODE" or "WIZARD_MODE"
manual_trigger_time = 0.0
ai_trigger_time = 0.0 # <--- ADD THIS LINE

# ==========================================
# 3. Web Server for the Pixel 9 (Background)
# ==========================================
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/')
def index():
    if not os.path.exists('woz_client.html'):
        return "ERROR: woz_client.html is missing!", 404
    return send_file('woz_client.html')

@app.route('/status')
def status():
    return jsonify({"state": current_state})

def run_flask():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        mac_ip = s.getsockname()[0]
    except Exception:
        mac_ip = '127.0.0.1'
    finally:
        s.close()
    
    print("\n=====================================================")
    print(f" PAGER SERVER LIVE! Go to http://{mac_ip}:5000 on Pixel")
    print("=====================================================\n")
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)

# ==========================================
# 4. Main OpenCV Loop (Must run on Main Thread)
# ==========================================
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    sequence_buffer = []
    
    print("Camera Online!")
    print("Controls:")
    print("  'm' - Toggle between AI Mode and Wizard Mode")
    print("  'SPACE' - Trigger manual handshake (Wizard Mode only)")
    print("  'q' - Quit")
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Always extract AI math in the background so you can monitor its accuracy
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            current_landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                current_landmarks.extend([lm.x, lm.y, lm.z])
                
            sequence_buffer.append(current_landmarks)
            if len(sequence_buffer) > 30:
                sequence_buffer.pop(0)
                
            if len(sequence_buffer) == 30:
                np_data = np.array(sequence_buffer).T 
                tensor_data = torch.tensor([np_data], dtype=torch.float32)
                
                with torch.no_grad():
                    logits = model(tensor_data)
                    probabilities = torch.softmax(logits, dim=1)
                    current_prob = probabilities[0][1].item()
        else:
            sequence_buffer.clear()
            current_prob = 0.0


        # ==========================================
        # STATE DETERMINATION LOGIC
        # ==========================================
        if operating_mode == "AI_MODE":
            # 1. Check if the AI sees a handshake
            if current_prob > 0.80 and len(sequence_buffer) == 30:
                ai_trigger_time = time.time() # Record the exact time
                sequence_buffer.clear() # Empty buffer so it doesn't double-trigger
            
            # 2. Hold the "HANDSHAKE" state for 0.5 seconds so the phone has time to read it!
            if time.time() - ai_trigger_time < 0.5:
                current_state = "HANDSHAKE"
            else:
                current_state = "IDLE"
                
        elif operating_mode == "WIZARD_MODE":
            # Human is in the driver's seat
            # Keep state as HANDSHAKE for 0.5 seconds after spacebar is pressed
            # so the 10-FPS phone poller is guaranteed to catch it.
            if time.time() - manual_trigger_time < 0.5:
                current_state = "HANDSHAKE"
            else:
                current_state = "IDLE"

        # ==========================================
        # DRAW UI OVERLAY
        # ==========================================
        # Mode Indicator (Yellow for AI, Magenta for Wizard)
        mode_color = (0, 255, 255) if operating_mode == "AI_MODE" else (255, 0, 255)
        cv2.putText(frame, f"MODE: {operating_mode}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, mode_color, 4)
        
        # State & Confidence
        state_color = (0, 255, 0) if current_state == "HANDSHAKE" else (255, 255, 255)
        cv2.putText(frame, f"State: {current_state}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, state_color, 4)
        cv2.putText(frame, f"AI Confidence: {current_prob*100:.1f}%", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 2, state_color, 4)

        cv2.imshow('Wizard of Oz - Mac Vision', frame)
        
        # ==========================================
        # KEYBOARD CONTROLS
        # ==========================================
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            operating_mode = "WIZARD_MODE" if operating_mode == "AI_MODE" else "AI_MODE"
            print(f"\nSwitched to {operating_mode}")
        elif key == ord(' '): # Spacebar
            if operating_mode == "WIZARD_MODE":
                manual_trigger_time = time.time()
                print("\n[WIZARD] Handshake Triggered!")
            
    cap.release()
    cv2.destroyAllWindows()