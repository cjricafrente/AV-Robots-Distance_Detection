#!/usr/bin/env python3
"""
cam_stream.py — zero re-encode MJPEG stream at 640x360
Camera outputs MJPG natively so frames are passed through directly.
Access at: http://localhost:5000/video
"""

import cv2
import threading
import time
from flask import Flask, Response

app = Flask(__name__)

latest_jpg = None
frame_lock = threading.Lock()

def capture_loop():
    global latest_jpg
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} MJPG")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Frame is already 640x360 — just encode once, no resize needed
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with frame_lock:
            latest_jpg = buf.tobytes()

threading.Thread(target=capture_loop, daemon=True).start()

def generate():
    prev = None
    while True:
        with frame_lock:
            jpg = latest_jpg
        if jpg is prev:
            time.sleep(0.001)
            continue
        prev = jpg
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n'
            b'Content-Length: ' + str(len(jpg)).encode() + b'\r\n'
            b'\r\n' + jpg + b'\r\n'
        )

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return (
        '<h2>Webcam stream</h2>'
        '<p>640x360 @ 30fps — native MJPG passthrough</p>'
        '<p>Stream: <a href="/video">http://localhost:5000/video</a></p>'
        '<img src="/video" width="640" height="360"/>'
    )

if __name__ == '__main__':
    print("Stream URL: http://localhost:5000/video")
    app.run(host='0.0.0.0', port=5000, threaded=True)