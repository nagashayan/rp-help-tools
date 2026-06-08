"""
=============================================================================
Script 7: Mac Camera Server (Wizard of Oz Edge Setup)
=============================================================================
Purpose:
    Turns the Mac into a "dumb" IP camera and web server. It broadcasts a 
    live MJPEG video stream from the Mac's webcam over the local Wi-Fi, and 
    serves the client-side HTML/JS application to the mobile device.
    
Inputs:
    - Mac's built-in webcam.
    - 'temporal_cnn_raw.onnx' (must be in the same folder).
      
Outputs:
    - Hosts a local web server on port 5000.
=============================================================================
"""

import cv2
from flask import Flask, Response, send_file, send_from_directory
import socket

app = Flask(__name__, static_folder='.')

def get_local_ip():
    """Finds the Mac's local Wi-Fi IP address so you can type it into your phone."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def generate_video_stream():
    """Captures the Mac webcam and converts it to a browser-friendly video stream."""
    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to capture video frame from webcam.")
            continue
        # Compress to JPEG to send over Wi-Fi quickly
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame in byte format for the HTML <img> tag
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    # This serves your .onnx file to the Pixel 9 when the browser requests it
    return send_from_directory('.', filename)

if __name__ == "__main__":
    mac_ip = get_local_ip()
    print("=====================================================")
    print(" MAC SERVER IS LIVE!")
    print(f" 1. Connect your Pixel 9 to the same Wi-Fi network.")
    print(f" 2. Open Chrome on the Pixel 9 and go to:")
    print(f"    http://{mac_ip}:5000")
    print("=====================================================")
    app.run(host='0.0.0.0', port=5000)
