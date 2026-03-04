import cv2
import math
import time
import numpy as np
import csv
import os
import json
from collections import deque
from ultralytics import YOLO

# --- CALIBRATION & EMPIRICAL TUNING ---
camera_height_m = 0.23     
camera_tilt_deg = 12.0      # <--- ENTER YOUR TRUE TILT HERE

try:
    calib_data = np.load('calibration_params.npz')
    CAMERA_MATRIX = calib_data['camera_matrix']
    DIST_COEFFS = calib_data['dist_coeffs']
    CX = CAMERA_MATRIX[0, 2]
    CY = CAMERA_MATRIX[1, 2]
    FX = CAMERA_MATRIX[0, 0]
    FY = CAMERA_MATRIX[1, 1]
    print("Successfully loaded calibration_params.npz")
except Exception as e:
    print(f"Error loading calibration: {e}")
    exit()

# DIORAMA THRESHOLDS (Meters)
D_STOP = 0.52             
D_SLOW = 0.80    
TTC_THRESHOLD = 2.0

TIME_OBSERVE = 1.5 
TIME_TURN = 1.0
TIME_PASS = 1.5
TIME_RETURN = 1.0

MEDIAN_WINDOW = 5
DIST_ALPHA = 0.6
SPEED_ALPHA = 0.3

class State:
    FOLLOW = "FOLLOW_GLOBAL_PATH"
    SLOW = "LOCAL_SLOW"
    OBSERVE = "LOCAL_AVOID_STATIC (OBSERVE)" 
    DECIDE = "LOCAL_AVOID_STATIC (DECIDE)"
    OVERTAKE = "LOCAL_AVOID_STATIC (OVERTAKE)"
    REJOIN = "REJOIN_PATH"
    STOP_DYNAMIC = "LOCAL_AVOID_DYNAMIC (STOP)"

# --- CLASSES ---
class FPSMeter:
    def __init__(self, window_size=30):
        self.times = deque(maxlen=window_size)
    def tick(self):
        self.times.append(time.time())
    def get_fps(self):
        if len(self.times) < 2: return 30.0
        return len(self.times) / (self.times[-1] - self.times[0])

class KalmanFilter1D:
    def __init__(self, initial_dist):
        self.x = np.array([[initial_dist], [0.0]])
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[0.1]])
        self.Q = np.array([[0.01, 0.0], [0.0, 0.01]])

    def update_and_predict(self, measurement, dt):
        A = np.array([[1.0, dt], [0.0, 1.0]])
        x_pred = np.dot(A, self.x)
        P_pred = np.dot(np.dot(A, self.P), A.T) + self.Q

        innovation = measurement - np.dot(self.H, x_pred)
        S = np.dot(np.dot(self.H, P_pred), self.H.T) + self.R
        K = np.dot(np.dot(P_pred, self.H.T), np.linalg.inv(S))

        self.x = x_pred + np.dot(K, innovation)
        self.P = P_pred - np.dot(np.dot(K, self.H), P_pred)

        return self.x[0][0], self.x[1][0] 

class ObjectTracker:
    def __init__(self):
        self.tracks = {}

    def update(self, object_id, raw_distance, current_fps, current_fsm_state, fp_x):
        current_time = time.time()
        dt = 1.0 / current_fps if current_fps > 0 else 0.033

        if object_id not in self.tracks:
            self.tracks[object_id] = {
                'history': deque(maxlen=MEDIAN_WINDOW),
                'x_history': deque(maxlen=MEDIAN_WINDOW),
                'exp_dist': raw_distance,
                'exp_x': fp_x,
                'speed': 0.0,
                'last_time': current_time,
                'kalman': KalmanFilter1D(raw_distance)
            }
            return raw_distance, 0.0, float('inf'), 0.0

        track = self.tracks[object_id]
        
        # Depth filtering
        track['history'].append(raw_distance)
        median_dist = np.median(track['history'])
        exp_dist = (DIST_ALPHA * median_dist) + ((1 - DIST_ALPHA) * track['exp_dist'])
        track['exp_dist'] = exp_dist
        final_dist, kalman_speed = track['kalman'].update_and_predict(exp_dist, dt)
        
        # Calculate raw mathematical velocity
        new_speed = (SPEED_ALPHA * kalman_speed) + ((1 - SPEED_ALPHA) * track['speed'])
        
        # Calculate Closing Speed (Positive value means the object is getting closer)
        closing_speed = -new_speed
        
        # Lateral filtering (Left/Right movement in pixels)
        track['x_history'].append(fp_x)
        median_x = np.median(track['x_history'])
        exp_x = (DIST_ALPHA * median_x) + ((1 - DIST_ALPHA) * track['exp_x'])
        lat_speed_px = abs(exp_x - track['exp_x']) * current_fps
        track['exp_x'] = exp_x

        # Universal TTC Calculation (Calculates for ANYTHING getting closer)
        ttc = float('inf')
        if closing_speed > 0.01:
            ttc = final_dist / closing_speed 

        track['speed'] = new_speed 
        track['last_time'] = current_time
        
        return final_dist, closing_speed, ttc, lat_speed_px

class SystemLogger:
    def __init__(self, run_name="live_run"):
        self.run_name = run_name
        self.base_log_dir = "logs"
        self.run_dir = os.path.join(self.base_log_dir, self.run_name)
        
        if not os.path.exists(self.base_log_dir):
            os.makedirs(self.base_log_dir)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
            
        self.frame_file = open(os.path.join(self.run_dir, f"{run_name}_frame_log.csv"), 'w', newline='')
        self.obj_file = open(os.path.join(self.run_dir, f"{run_name}_object_log.csv"), 'w', newline='')
        
        self.frame_writer = csv.writer(self.frame_file)
        self.obj_writer = csv.writer(self.obj_file)
        
        self.frame_writer.writerow(["frame_id", "timestamp", "fps", "total_detections", "global_action"])
        self.obj_writer.writerow(["frame_id", "track_id", "class", "dist_m", "speed_ms", "ttc_s", "state"])
        
        self.decision_events = []
        self.action_counts = {"GO": 0, "SLOW": 0, "STOP": 0, "OVERTAKE": 0}

    def log_frame(self, frame_id, fps, num_detections, global_action):
        self.frame_writer.writerow([frame_id, time.time(), round(fps, 2), num_detections, global_action])
        if global_action in self.action_counts:
            self.action_counts[global_action] += 1

    def log_object(self, frame_id, obj_data):
        state_str = "DYNAMIC" if obj_data['is_dynamic'] else "STATIC"
        self.obj_writer.writerow([
            frame_id, obj_data['id'], obj_data['label'], 
            round(obj_data['dist'], 3), round(obj_data['speed'], 3), 
            round(obj_data['ttc'], 2), state_str
        ])

    def log_decision_event(self, frame_id, trigger_reason, action, obj_id):
        event = {
            "time": time.time(),
            "frame": frame_id,
            "trigger": trigger_reason,
            "action": action,
            "target_track_id": obj_id
        }
        self.decision_events.append(event)

    def close(self, total_frames, avg_fps):
        self.frame_file.close()
        self.obj_file.close()
        
        decision_path = os.path.join(self.run_dir, f"{self.run_name}_decision_log.json")
        with open(decision_path, 'w') as f:
            json.dump(self.decision_events, f, indent=4)
            
        summary = {
            "total_frames": total_frames,
            "average_fps": round(avg_fps, 2),
            "action_distribution": self.action_counts
        }
        summary_path = os.path.join(self.run_dir, f"{self.run_name}_clip_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

def calculate_calibrated_distance(footpoint_x, footpoint_y, current_h, current_tilt):
    pixel_point = np.array([[[float(footpoint_x), float(footpoint_y)]]], dtype=np.float32)
    undistorted_point = cv2.undistortPoints(pixel_point, CAMERA_MATRIX, DIST_COEFFS, P=CAMERA_MATRIX)
    
    v_prime = undistorted_point[0][0][1]
    y_norm = (v_prime - CY) / FY
    theta_pixel = math.atan(y_norm)
    
    total_angle = math.radians(current_tilt) + theta_pixel
    if total_angle <= 0: return float('inf')
    
    return current_h / math.tan(total_angle)

def draw_outlined_text(img, text, pos, scale, color):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

# --- LIVE EXECUTION SETUP ---
print("Loading custom weights: best.pt")
model = YOLO('best.pt') 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open the live camera feed.")
    exit()

print("\n--- Starting Live Diorama Feed ---")
print("Press 'q' in the video window to stop the run and save logs.")

run_timestamp = time.strftime("%Y%m%d-%H%M%S")
sys_logger = SystemLogger(f"live_run_{run_timestamp}")

video_path = os.path.join(sys_logger.run_dir, f"{sys_logger.run_name}_recording.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))

tracker = ObjectTracker()
fps_meter = FPSMeter()

current_state = State.FOLLOW
state_start_time = 0
overtake_direction = "NONE"
overtake_angle = 0.0          # Tracks the exact steering degree required
LATERAL_OFFSET_M = 0.30       # The assumed sideways clearance in meters
frame_count = 0

while True:
    success, frame = cap.read()
    if not success: 
        print("Camera feed interrupted.")
        break
        
    h, w, _ = frame.shape
    fps_meter.tick()
    current_fps = fps_meter.get_fps()
    frame_count += 1
    
    results = model.track(frame, persist=True, stream=True, verbose=False)
    
    critical_obstacle = None
    min_dist = float('inf')
    all_detections = []
    rc_car_speed_cm_s = 0.0 
    
    for r in results:
        if r.boxes.id is not None:
            track_ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                oid = track_ids[i]
                label_name = model.names[int(box.cls[0])]
                fp_x = int((x1 + x2) / 2)
                fp_y = int(y2)
                
                raw_dist = calculate_calibrated_distance(fp_x, fp_y, camera_height_m, camera_tilt_deg)
                
                if raw_dist != float('inf'):
                    dist, closing_speed, ttc, lat_speed = tracker.update(oid, raw_dist, current_fps, current_state, fp_x)
                    
                    # ADAPTABLE DYNAMIC CHECK 
                    # It is dynamic if it darts sideways OR if it is still moving toward us while we are stopped
                    is_dynamic = lat_speed > 40.0 or (current_state in [State.OBSERVE, State.STOP_DYNAMIC] and closing_speed > 0.03)

                    detection_data = {
                        'id': oid, 'label': label_name, 'box': (x1, y1, x2, y2), 
                        'fp': (fp_x, fp_y), 'dist': dist, 'speed': closing_speed, 
                        'ttc': ttc, 'lat_speed': lat_speed, 'center_x': fp_x,
                        'is_dynamic': is_dynamic
                    }
                    all_detections.append(detection_data)
                    sys_logger.log_object(frame_count, detection_data)

                    if dist < min_dist:
                        min_dist = dist
                        critical_obstacle = detection_data

    # --- SEPARATED FSM LOGIC ---
    action_text = "GO"
    hud_color = (0, 255, 0)
    global_action = "GO"

    if critical_obstacle:
        dist = critical_obstacle['dist']
        ttc = critical_obstacle['ttc']
        closing_speed = critical_obstacle['speed']
        is_dynamic = critical_obstacle['is_dynamic']
        
        # Display the car speed roughly based on closing speed while driving
        if current_state in [State.FOLLOW, State.SLOW] and not is_dynamic:
            rc_car_speed_cm_s = closing_speed * 100
            if rc_car_speed_cm_s < 0: rc_car_speed_cm_s = 0.0 

        if current_state in [State.FOLLOW, State.SLOW]:
            # Universal Safety Net: React to TTC or Distance regardless of static/dynamic label
            if ttc <= TTC_THRESHOLD or dist <= D_STOP:
                current_state = State.OBSERVE
                state_start_time = time.time()
                action_text = "BRAKING TO OBSERVE"
                hud_color = (0, 0, 255)
                global_action = "STOP"
                sys_logger.log_decision_event(frame_count, "Distance_or_TTC_Safety", "OBSERVE", critical_obstacle['id'])
            elif dist <= D_SLOW:
                current_state = State.SLOW
                action_text = "SLOW DOWN"
                hud_color = (0, 255, 255)
                global_action = "SLOW"
                
        elif current_state == State.OBSERVE:
            action_text = "WAITING TO CONFIRM STATIC"
            hud_color = (0, 0, 255)
            global_action = "STOP"
            
            # The Brake Test: The car is physically stopped. Does the object keep moving?
            if is_dynamic:
                current_state = State.STOP_DYNAMIC
                action_text = "STOP (DYNAMIC THREAT)"
                sys_logger.log_decision_event(frame_count, "Dynamic_Confirmed", "STOP", critical_obstacle['id'])
            elif time.time() - state_start_time > TIME_OBSERVE:
                # 1.5 seconds passed and the object is NOT moving. It is confirmed STATIC.
                current_state = State.DECIDE
                state_start_time = time.time()

        elif current_state == State.STOP_DYNAMIC:
            action_text = "STOP (WAITING FOR CLEAR PATH)"
            hud_color = (0, 0, 255)
            global_action = "STOP"
            # If the object stops moving or moves away, revert to driving
            if not is_dynamic and closing_speed < 0.02:
                current_state = State.FOLLOW

        elif current_state == State.DECIDE:
            overtake_direction = "LEFT" if critical_obstacle['center_x'] > CX else "RIGHT"
            longitudinal_dist = critical_obstacle['dist']
            
            # Mathematical calculation for the turning angle using inverse tangent
            angle_radians = math.atan2(LATERAL_OFFSET_M, longitudinal_dist)
            overtake_angle = math.degrees(angle_radians)
            
            current_state = State.OVERTAKE
            state_start_time = time.time()
            action_text = "DECIDING DIRECTION"
            global_action = "STOP"
            sys_logger.log_decision_event(frame_count, "Overtake_Planned", f"OVERTAKE {overtake_direction} AT {overtake_angle:.1f} DEG", critical_obstacle['id'])

        elif current_state == State.OVERTAKE:
            elapsed = time.time() - state_start_time
            hud_color = (255, 165, 0)
            global_action = "OVERTAKE"
            if elapsed < TIME_TURN:
                action_text = f"TURN {overtake_direction} {overtake_angle:.1f} DEG"
            elif elapsed < (TIME_TURN + TIME_PASS):
                action_text = "DRIVE STRAIGHT"
            elif elapsed < (TIME_TURN + TIME_PASS + TIME_RETURN):
                return_dir = "RIGHT" if overtake_direction == "LEFT" else "LEFT"
                action_text = f"RETURN {return_dir} {overtake_angle:.1f} DEG"
            else:
                current_state = State.REJOIN
                state_start_time = time.time()

        elif current_state == State.REJOIN:
            action_text = "REALIGNING"
            hud_color = (255, 255, 0)
            global_action = "OVERTAKE"
            if time.time() - state_start_time > 1.0:
                current_state = State.FOLLOW
    else:
        if current_state == State.STOP_DYNAMIC:
            current_state = State.FOLLOW
            
    sys_logger.log_frame(frame_count, current_fps, len(all_detections), global_action)

    # --- DRAWING ---
    for d in all_detections:
        x1, y1, x2, y2 = d['box']
        is_crit = (critical_obstacle and d['id'] == critical_obstacle['id'])
        color = hud_color if is_crit else (100, 100, 100)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, d['fp'], 5, (0, 0, 255), -1)
        
        dist_cm = d['dist'] * 100 
        if is_crit:
            data_str = f"Dist:{dist_cm:.1f}cm | TTC:{d['ttc']:.1f}s"
            draw_outlined_text(frame, data_str, (x1, y1 - 5), 0.6, color)

    draw_outlined_text(frame, "LIVE DIORAMA TEST (RECORDING)", (20, 40), 1.0, (255, 255, 255))
    draw_outlined_text(frame, f"STATE: {current_state}", (20, 80), 1.0, (255, 255, 255))
    draw_outlined_text(frame, action_text, (20, 120), 1.0, hud_color)
    draw_outlined_text(frame, f"RC Car Speed: {rc_car_speed_cm_s:.1f} cm/s", (20, 160), 0.8, (0, 255, 255))

    video_writer.write(frame)
    cv2.imshow("Live Diorama Feed", frame)
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'): 
        break

cap.release()
video_writer.release()
sys_logger.close(frame_count, fps_meter.get_fps())
cv2.destroyAllWindows()
print(f"Live evaluation complete. Video and logs saved in {sys_logger.run_dir}")