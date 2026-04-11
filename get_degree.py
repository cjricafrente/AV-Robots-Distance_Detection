import numpy as np
import cv2
import math

# --- CONFIGURATION ---
MARKER_SIZE_CM = 9.8
CALIBRATION_FILE = 'calibration_data.npz'

def get_orientation(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)

# --- 1. Load Calibration Data ---
# calibration_data.npz is a separate file from calibration_params.npz
# Keys here are 'mtx' and 'dist' — intentionally different from the GPM calibration file
try:
    calib = np.load(CALIBRATION_FILE)
    mtx  = calib['mtx']
    dist = calib['dist']
    print(f"Loaded calibration from '{CALIBRATION_FILE}'")
    print(f"  FX={mtx[0,0]:.2f}  FY={mtx[1,1]:.2f}  CX={mtx[0,2]:.2f}  CY={mtx[1,2]:.2f}")
except FileNotFoundError:
    print(f"Error: '{CALIBRATION_FILE}' not found.")
    exit()
except KeyError as e:
    print(f"Error: Key {e} not found in '{CALIBRATION_FILE}'.")
    exit()

# --- 2. Define Marker Object Points ---
ms = MARKER_SIZE_CM / 2.0
marker_points = np.array([
    [-ms,  ms, 0],
    [ ms,  ms, 0],
    [ ms, -ms, 0],
    [-ms, -ms, 0]
], dtype=np.float32)

# --- 3. Start Camera ---
# CAP_DSHOW + 1280x720 must match the backend used during recalibration
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector   = cv2.aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # CAP_DSHOW matches recalibration backend
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)  # must match calibration resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Verify true delivered resolution
for _ in range(3):
    cap.read()
ret, probe = cap.read()
if ret:
    true_w, true_h = probe.shape[1], probe.shape[0]
    print(f"Camera delivering: {true_w}x{true_h}")
    if true_w != 1280 or true_h != 720:
        print(f"  ⚠  Resolution mismatch — matrix was calibrated at 1280x720.")
        print(f"     Tilt readings may be inaccurate.")
    else:
        print(f"  ✔  Resolution matches calibration.")

print(f"\n--- TILT MEASUREMENT STARTED ---")
print(f"Marker size: {MARKER_SIZE_CM} cm")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # No resize — matrix is valid for 1280x720 frames as-is
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        for i in range(len(ids)):
            success, rvec, tvec = cv2.solvePnP(marker_points, corners[i], mtx, dist)
            if success:
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 5)

                pitch, yaw, roll = get_orientation(rvec)
                true_camera_tilt = abs(90.0 - abs(pitch))

                cv2.putText(frame, f"True Tilt (from horizon): {true_camera_tilt:.1f} deg",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw (Pan): {yaw:.1f} deg",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.1f} deg",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)

    cv2.imshow('Camera Tilt Measurement', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()