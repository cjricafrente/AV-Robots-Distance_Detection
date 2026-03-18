from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)

# Shared latest frame — one thread reads, all clients use this
latest_frame = None
lock = threading.Lock()

def camera_reader():
    global latest_frame
    cap = cv2.VideoCapture('/dev/video0')
    if not cap.isOpened():
        cap = cv2.VideoCapture('/dev/video2')
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        with lock:
            latest_frame = buffer.tobytes()

# Start ONE camera thread at startup — all clients share it
t = threading.Thread(target=camera_reader, daemon=True)
t.start()

def generate():
    while True:
        with lock:
            frame = latest_frame
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return '''
    <html>
    <body style="margin:0;background:#000;display:flex;align-items:center;justify-content:center;height:100vh;">
        <img src="/stream" style="max-width:100%;max-height:100vh;">
    </body>
    </html>
    '''

@app.route('/stream')
def stream():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Stream ready at http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, threaded=True)