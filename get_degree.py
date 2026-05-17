<<<<<<< HEAD
import numpy as np
import cv2
import math

# --- CONFIGURATION ---
MARKER_SIZE_CM = 9.8  # <--- ENTER YOUR MEASURED SIZE HERE
CALIBRATION_FILE = 'calibration_data.npz'

def get_orientation(rvec):
    # Converts rotation vector to angles (Pitch, Yaw, Roll)
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
try:
    with np.load(CALIBRATION_FILE) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
except FileNotFoundError:
    print("Error: 'calibration_data.npz' not found.")
    exit()

# --- 2. Define the Marker Object (The Fix for the Error) ---
# We define the 3D coordinates of the marker corners (TopLeft, TopRight, BottomRight, BottomLeft)
# The center of the marker is (0,0,0)
ms = MARKER_SIZE_CM / 2.0
marker_points = np.array([
    [-ms,  ms, 0],
    [ ms,  ms, 0],
    [ ms, -ms, 0],
    [-ms, -ms, 0]
], dtype=np.float32)

# --- 3. Start Camera ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(0)

print("--- MEASUREMENT STARTED ---")
print(f"Marker Size set to: {MARKER_SIZE_CM} cm")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        # Loop through all markers found
        for i in range(len(ids)):
            # --- THE FIX: USE solvePnP INSTEAD OF estimatePoseSingleMarkers ---
            # matches the specific corners of the detected marker to our 3D model
            success, rvec, tvec = cv2.solvePnP(marker_points, corners[i], mtx, dist)
            
            if success:
                # Draw the marker border
                cv2.aruco.drawDetectedMarkers(frame, corners)
                
                # Draw the 3D axis (Red=X, Green=Y, Blue=Z)
                # Length of axis is 5 cm
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 5)

                # Calculate angles
                pitch, yaw, roll = get_orientation(rvec)

                # Adjust for the paper lying flat on the floor
                true_camera_tilt = abs(90.0 - abs(pitch))

                # Display text
                text_pitch = f"True Tilt (from horizon): {true_camera_tilt:.1f} deg"
                text_yaw =   f"Yaw (Pan):   {yaw:.1f}"  
                
                cv2.putText(frame, text_pitch, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, text_yaw, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Camera Tilt Measurement', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
=======
import numpy as np
import cv2
import math

# --- CONFIGURATION ---
MARKER_SIZE_CM = 9.8  # <--- ENTER YOUR MEASURED SIZE HERE
CALIBRATION_FILE = 'calibration_data.npz'

def get_orientation(rvec):
    # Converts rotation vector to angles (Pitch, Yaw, Roll)
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
try:
    with np.load(CALIBRATION_FILE) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
except FileNotFoundError:
    print("Error: 'calibration_data.npz' not found.")
    exit()

# --- 2. Define the Marker Object (The Fix for the Error) ---
# We define the 3D coordinates of the marker corners (TopLeft, TopRight, BottomRight, BottomLeft)
# The center of the marker is (0,0,0)
ms = MARKER_SIZE_CM / 2.0
marker_points = np.array([
    [-ms,  ms, 0],
    [ ms,  ms, 0],
    [ ms, -ms, 0],
    [-ms, -ms, 0]
], dtype=np.float32)

# --- 3. Start Camera ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(0)

print("--- MEASUREMENT STARTED ---")
print(f"Marker Size set to: {MARKER_SIZE_CM} cm")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        # Loop through all markers found
        for i in range(len(ids)):
            # --- THE FIX: USE solvePnP INSTEAD OF estimatePoseSingleMarkers ---
            # matches the specific corners of the detected marker to our 3D model
            success, rvec, tvec = cv2.solvePnP(marker_points, corners[i], mtx, dist)
            
            if success:
                # Draw the marker border
                cv2.aruco.drawDetectedMarkers(frame, corners)
                
                # Draw the 3D axis (Red=X, Green=Y, Blue=Z)
                # Length of axis is 5 cm
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 5)

                # Calculate angles
                pitch, yaw, roll = get_orientation(rvec)

                # Adjust for the paper lying flat on the floor
                true_camera_tilt = abs(90.0 - abs(pitch))

                # Display text
                text_pitch = f"True Tilt (from horizon): {true_camera_tilt:.1f} deg"
                text_yaw =   f"Yaw (Pan):   {yaw:.1f}"  
                
                cv2.putText(frame, text_pitch, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, text_yaw, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Camera Tilt Measurement', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
>>>>>>> 94f2ccf65ca0e7d11679167c3c3008a168eb7924
cv2.destroyAllWindows()