"""
=============================================================================
Script 7: Mac Camera Server (Wizard of Oz Snapshot API)
=============================================================================
Purpose:
    Turns the Mac into a lag-free snapshot server. A background thread 
    constantly clears the OpenCV camera buffer to prevent latency. The 
    mobile client pulls individual JPEGs via the /snapshot endpoint, 
    completely bypassing the WebAssembly MJPEG crash.
=============================================================================
"""

import cv2
from flask import Flask, Response, jsonify, send_file, send_from_directory
import socket
import threading

app = Flask(__name__, static_folder='.')

# 1. Open the camera
camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
latest_frame = None

def capture_loop():
    """Background thread to keep the camera buffer empty and lag-free."""
    global latest_frame
    while True:
        success, frame = camera.read()
        if success:
            latest_frame = frame

# Start the camera thread
thread = threading.Thread(target=capture_loop, daemon=True)
thread.start()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/snapshot')
def snapshot():
    """Serves a heavily downscaled JPEG to prevent Pixel 9 memory crashes."""
    global latest_frame
    if latest_frame is None:
        return "Waiting for camera", 503
        
    # THE FIX: Crush the 1080p image down to a tiny 320x240 footprint
    small_frame = cv2.resize(latest_frame, (320, 240))
    
    # Compress the tiny image into a JPEG
    ret, buffer = cv2.imencode('.jpg', small_frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/health')
def health():
    ready = latest_frame is not None and camera.isOpened()
    status_code = 200 if ready else 503
    return jsonify({"ready": ready}), status_code

if __name__ == "__main__":
    mac_ip = get_local_ip()
    print("=====================================================")
    print(" SNAPSHOT SERVER IS LIVE!")
    print(f" Pixel 9 URL: http://{mac_ip}:5000")
    print("=====================================================")
    app.run(host='0.0.0.0', port=5000, threaded=True)
    
