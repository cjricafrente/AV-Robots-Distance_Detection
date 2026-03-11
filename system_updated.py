import cv2
import math
import time
import numpy as np
import csv
import os
import json
from collections import deque
from ultralytics import YOLO

# --- OPTIMIZATION CONFIGURATION ---
SKIP_FRAMES = 2  # 1 = YOLO every frame, 2 = YOLO every other frame
TARGET_WIDTH = 640
TARGET_HEIGHT = 360 # Preserves 16:9 aspect ratio

# --- CALIBRATION & EMPIRICAL TUNING ---
camera_height_m = 0.23
camera_tilt_deg = 12.0      

try:
    calib_data = np.load('calibration_params.npz')
    CAMERA_MATRIX = calib_data['camera_matrix'].copy()
    DIST_COEFFS = calib_data['dist_coeffs']
    
    # Scale parameters for the new 640x360 resolution
    scale_x = TARGET_WIDTH / 1280.0
    scale_y = TARGET_HEIGHT / 720.0
    
    CX = CAMERA_MATRIX[0, 2] * scale_x
    CY = CAMERA_MATRIX[1, 2] * scale_y
    FX = CAMERA_MATRIX[0, 0] * scale_x
    FY = CAMERA_MATRIX[1, 1] * scale_y
    
    CAMERA_MATRIX[0, 2] = CX
    CAMERA_MATRIX[1, 2] = CY
    CAMERA_MATRIX[0, 0] = FX
    CAMERA_MATRIX[1, 1] = FY
    print("Successfully loaded and scaled calibration_params.npz")
except Exception as e:
    print(f"Error loading calibration: {e}")
    exit()

# DIORAMA THRESHOLDS
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
BLIND_SPOT_TIMEOUT = 2.0
CAMERA_BUMPER_OFFSET_M = 0.16   
MAX_STEERING_ANGLE     = 40.0   
EVAL_GROUND_TRUTH_M = 0.50

class State:
    FOLLOW = "FOLLOW_GLOBAL_PATH"
    SLOW = "LOCAL_SLOW"
    OBSERVE = "LOCAL_AVOID_STATIC (OBSERVE)"
    DECIDE = "LOCAL_AVOID_STATIC (DECIDE)"
    OVERTAKE = "LOCAL_AVOID_STATIC (OVERTAKE)"
    REJOIN = "REJOIN_PATH"
    STOP_DYNAMIC = "LOCAL_AVOID_DYNAMIC (STOP)"

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

    def update(self, object_id, raw_distance, dt, current_fsm_state, fp_x):
        current_time = time.time()
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
        track['history'].append(raw_distance)
        median_dist = np.median(track['history'])
        exp_dist = (DIST_ALPHA * median_dist) + ((1 - DIST_ALPHA) * track['exp_dist'])
        track['exp_dist'] = exp_dist
        final_dist, kalman_speed = track['kalman'].update_and_predict(exp_dist, dt)
        new_speed = (SPEED_ALPHA * kalman_speed) + ((1 - SPEED_ALPHA) * track['speed'])
        closing_speed = -new_speed

        track['x_history'].append(fp_x)
        median_x = np.median(track['x_history'])
        exp_x = (DIST_ALPHA * median_x) + ((1 - DIST_ALPHA) * track['exp_x'])
        lat_speed_px = abs(exp_x - track['exp_x']) / dt if dt > 0 else 0.0
        track['exp_x'] = exp_x

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
        os.makedirs(self.base_log_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        self.frame_file = open(os.path.join(self.run_dir, f"{run_name}_frame_log.csv"), 'w', newline='')
        self.obj_file = open(os.path.join(self.run_dir, f"{run_name}_object_log.csv"), 'w', newline='')
        self.frame_writer = csv.writer(self.frame_file)
        self.obj_writer = csv.writer(self.obj_file)
        self.frame_writer.writerow(["frame_id", "timestamp", "fps", "total_detections", "global_action"])
        self.obj_writer.writerow(["frame_id", "track_id", "class", "dist_m", "speed_ms", "ttc_s", "state"])
        self.decision_events = []
        self.action_counts = {"GO": 0, "SLOW": 0, "STOP": 0, "OVERTAKE": 0}
        self.eval_dist_histories = {}

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
        tid = obj_data['id']
        if tid not in self.eval_dist_histories:
            self.eval_dist_histories[tid] = []
        self.eval_dist_histories[tid].append(obj_data['dist'])

    def log_decision_event(self, frame_id, trigger_reason, action, obj_id):
        event = {"time": time.time(), "frame": frame_id, "trigger": trigger_reason, "action": action, "target_track_id": obj_id}
        self.decision_events.append(event)

    def close(self, total_frames, avg_fps):
        self.frame_file.close()
        self.obj_file.close()

        with open(os.path.join(self.run_dir, f"{self.run_name}_decision_log.json"), 'w') as f:
            json.dump(self.decision_events, f, indent=4)

        summary = {
            "total_frames": total_frames,
            "average_fps": round(avg_fps, 2),
            "action_distribution": self.action_counts
        }
        with open(os.path.join(self.run_dir, f"{self.run_name}_clip_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)

        # --- FULL EVALUATION REPORT METRICS ---
        report = {
            "run_name": self.run_name,
            "total_frames": total_frames,
            "average_fps": round(avg_fps, 2),
            "ground_truth_m": EVAL_GROUND_TRUTH_M,
        }

        EVAL_MAX_RELIABLE_DIST_M = 0.85
        EVAL_PLAUSIBILITY_BAND_M = 0.40

        if EVAL_GROUND_TRUTH_M is not None and self.eval_dist_histories:
            candidate_tracks = {
                tid: dists
                for tid, dists in self.eval_dist_histories.items()
                if dists and
                   float(np.mean(dists)) <= EVAL_MAX_RELIABLE_DIST_M and
                   abs(float(np.mean(dists)) - EVAL_GROUND_TRUTH_M) <= EVAL_PLAUSIBILITY_BAND_M
            }

            if candidate_tracks:
                primary_tid = max(candidate_tracks, key=lambda tid: len(candidate_tracks[tid]))
                primary_dists = candidate_tracks[primary_tid]
                errors = [abs(d - EVAL_GROUND_TRUTH_M) for d in primary_dists]
                sq_errors = [e ** 2 for e in errors]

                mae = float(np.mean(errors))
                rmse = float(np.sqrt(np.mean(sq_errors)))
                p95_error = float(np.percentile(errors, 95))

                report["distance_accuracy"] = {
                    "primary_track_id": primary_tid,
                    "candidates_after_filter": len(candidate_tracks),
                    "tracks_rejected": len(self.eval_dist_histories) - len(candidate_tracks),
                    "sample_count": len(primary_dists),
                    "mean_dist_m": round(float(np.mean(primary_dists)), 4),
                    "std_dist_m": round(float(np.std(primary_dists)), 4),
                    "MAE_m": round(mae, 4),
                    "RMSE_m": round(rmse, 4),
                    "p95_error_m": round(p95_error, 4),
                }
            else:
                all_means = {str(tid): round(float(np.mean(dists)), 4) for tid, dists in self.eval_dist_histories.items() if dists}
                report["distance_accuracy"] = {
                    "error": "no_plausible_track_found",
                    "max_reliable_dist_m": EVAL_MAX_RELIABLE_DIST_M,
                    "plausibility_band_m": EVAL_PLAUSIBILITY_BAND_M,
                    "ground_truth_m": EVAL_GROUND_TRUTH_M,
                    "track_mean_dists": all_means,
                }
        else:
            report["distance_accuracy"] = "skipped_no_ground_truth" if EVAL_GROUND_TRUTH_M is None else "no_detections_logged"

        track_stability = {}
        for tid, dists in self.eval_dist_histories.items():
            lifespan = len(dists)
            if lifespan >= 2:
                frame_diffs = [dists[i] - dists[i - 1] for i in range(1, lifespan)]
                ftf_variance = float(np.var(frame_diffs))
                ftf_std = float(np.std(frame_diffs))
            else:
                ftf_variance = None
                ftf_std = None

            track_stability[str(tid)] = {
                "lifespan_frames": lifespan,
                "ftf_variance_m2": round(ftf_variance, 6) if ftf_variance is not None else None,
                "ftf_std_m": round(ftf_std, 6) if ftf_std is not None else None,
            }

        valid_variances = [v["ftf_variance_m2"] for v in track_stability.values() if v["ftf_variance_m2"] is not None]

        report["temporal_stability"] = {
            "per_track": track_stability,
            "total_unique_track_ids": len(self.eval_dist_histories),
            "mean_track_lifespan_frames": round(float(np.mean([v["lifespan_frames"] for v in track_stability.values()])), 2) if track_stability else 0.0,
            "mean_ftf_variance_m2": round(float(np.mean(valid_variances)), 6) if valid_variances else None,
        }

        report_path = os.path.join(self.run_dir, f"{self.run_name}_evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        print(f"\n--- Evaluation Report Saved: {report_path} ---")

def calculate_calibrated_distance(footpoint_x, footpoint_y, current_h, current_tilt):
    pixel_point = np.array([[[float(footpoint_x), float(footpoint_y)]]], dtype=np.float32)
    undistorted_point = cv2.undistortPoints(pixel_point, CAMERA_MATRIX, DIST_COEFFS, P=CAMERA_MATRIX)
    v_prime = undistorted_point[0][0][1]
    y_norm = (v_prime - CY) / FY
    theta_pixel = math.atan(y_norm)
    total_angle = math.radians(current_tilt) + theta_pixel
    if total_angle <= 0:
        return float('inf')
    return current_h / math.tan(total_angle)

def draw_outlined_text(img, text, pos, scale, color):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

# --- LIVE EXECUTION SETUP ---
print("Loading highly-optimized NCNN model for Raspberry Pi...")
# [OPTIMIZATION] Load the NCNN model exported earlier
model = YOLO('best_ncnn_model') 

# [OPTIMIZATION] Read from the live USB Camera (usually index 0 on a Pi)
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("Error: Could not open the physical camera feed. Check your USB connection!")
    exit()

print("\n--- Starting Optimized Ubuntu Live Feed ---")
print("Press 'q' in the video window to stop the run and save logs.")

run_timestamp = time.strftime("%Y%m%d-%H%M%S")
sys_logger = SystemLogger(f"live_run_{run_timestamp}")

# [OPTIMIZATION] Save the live video feed straight into the Pi's rapid RAM partition (/dev/shm)
ram_video_path = f"/dev/shm/{sys_logger.run_name}_recording.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(ram_video_path, fourcc, 30.0, (TARGET_WIDTH, TARGET_HEIGHT))

tracker = ObjectTracker()
active_trackers = {} 

current_state = State.FOLLOW
state_start_time = 0
overtake_direction = "NONE"
overtake_angle = 0.0          
last_obstacle_seen_time = time.time()
frame_count = 0
run_start_time = time.time()
prev_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        break
        
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    current_time = time.time()
    dt = current_time - prev_time
    if dt <= 0:
        dt = 1.0 / 30.0
    current_fps = 1.0 / dt          

    h, w, _ = frame.shape
    frame_count += 1
    
    critical_obstacle = None
    min_dist = float('inf')
    all_detections = []

    if frame_count % SKIP_FRAMES == 0 or frame_count == 1:
        results = model.track(frame, persist=True, stream=False, verbose=False)
        active_trackers.clear()
        
        for r in results:
            if r.boxes.id is not None:
                track_ids = r.boxes.id.int().cpu().tolist()
                boxes = r.boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    oid = track_ids[i]
                    label_name = model.names[int(box.cls[0])]
                    
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    
                    try:
                        kcf_tracker = cv2.TrackerKCF.create()
                    except AttributeError:
                        try:
                            kcf_tracker = cv2.legacy.TrackerKCF_create()
                        except AttributeError:
                            kcf_tracker = cv2.TrackerKCF_create()
                            
                    kcf_tracker.init(frame, bbox)
                    active_trackers[oid] = {'tracker': kcf_tracker, 'label': label_name}
                    
                    fp_x = int((x1 + x2) / 2)
                    fp_y = int(y2)
                    raw_dist = calculate_calibrated_distance(fp_x, fp_y, camera_height_m, camera_tilt_deg)

                    if raw_dist != float('inf'):
                        dist, closing_speed, ttc, lat_speed = tracker.update(oid, raw_dist, dt, current_state, fp_x)
                        is_dynamic = lat_speed > 40.0 or (current_state in [State.OBSERVE, State.STOP_DYNAMIC] and closing_speed > 0.03)
                        
                        detection_data = {
                            'id': oid, 'label': label_name, 'box': (x1, y1, x2, y2),
                            'fp': (fp_x, fp_y), 'dist': dist, 'speed': closing_speed,
                            'ttc': ttc, 'lat_speed': lat_speed, 'center_x': fp_x,
                            'is_dynamic': is_dynamic
                        }
                        all_detections.append(detection_data)
    else:
        for oid, track_info in list(active_trackers.items()):
            kcf_tracker = track_info['tracker']
            label_name = track_info['label']
            success_track, bbox = kcf_tracker.update(frame)
            
            if success_track:
                x, y, w_box, h_box = map(int, bbox)
                x1, y1, x2, y2 = x, y, x + w_box, y + h_box
                fp_x = int((x1 + x2) / 2)
                fp_y = int(y2)
                
                raw_dist = calculate_calibrated_distance(fp_x, fp_y, camera_height_m, camera_tilt_deg)
                if raw_dist != float('inf'):
                    dist, closing_speed, ttc, lat_speed = tracker.update(oid, raw_dist, dt, current_state, fp_x)
                    is_dynamic = lat_speed > 40.0 or (current_state in [State.OBSERVE, State.STOP_DYNAMIC] and closing_speed > 0.03)
                    
                    detection_data = {
                        'id': oid, 'label': label_name, 'box': (x1, y1, x2, y2),
                        'fp': (fp_x, fp_y), 'dist': dist, 'speed': closing_speed,
                        'ttc': ttc, 'lat_speed': lat_speed, 'center_x': fp_x,
                        'is_dynamic': is_dynamic
                    }
                    all_detections.append(detection_data)
            else:
                del active_trackers[oid]

    for d in all_detections:
        sys_logger.log_object(frame_count, d)
        if d['dist'] < min_dist:
            min_dist = d['dist']
            critical_obstacle = d

    if critical_obstacle is not None:
        last_obstacle_seen_time = current_time

    action_text = "GO"
    hud_color   = (0, 255, 0)
    global_action = "GO"
    show_speed = False
    display_speed_cm_s = 0.0
    speed_label = ""

    if current_state == State.OVERTAKE:
        elapsed = time.time() - state_start_time
        hud_color     = (255, 165, 0)
        global_action = "OVERTAKE"
        if elapsed < TIME_TURN:
            action_text = f"TURN {overtake_direction} {overtake_angle:.1f} DEG"
        elif elapsed < (TIME_TURN + TIME_PASS):
            action_text = "DRIVE STRAIGHT"
        elif elapsed < (TIME_TURN + TIME_PASS + TIME_RETURN):
            return_dir  = "RIGHT" if overtake_direction == "LEFT" else "LEFT"
            action_text = f"RETURN {return_dir} {overtake_angle:.1f} DEG"
        else:
            current_state    = State.REJOIN
            state_start_time = time.time()
    elif current_state == State.REJOIN:
        action_text   = "REALIGNING"
        hud_color     = (255, 255, 0)
        global_action = "OVERTAKE"
        if time.time() - state_start_time > 1.0:
            current_state = State.FOLLOW
    elif critical_obstacle:
        dist          = critical_obstacle['dist']
        ttc           = critical_obstacle['ttc']
        closing_speed = critical_obstacle['speed']
        is_dynamic    = critical_obstacle['is_dynamic']
        show_speed = True
        display_speed_cm_s = max(0.0, closing_speed * 100)
        speed_label = "Relative Closing Speed"

        if current_state in [State.FOLLOW, State.SLOW]:
            if ttc <= TTC_THRESHOLD or dist <= D_STOP:
                current_state    = State.OBSERVE
                state_start_time = time.time()
                action_text      = "BRAKING TO OBSERVE"
                hud_color        = (0, 0, 255)
                global_action    = "STOP"
            elif dist <= D_SLOW:
                current_state = State.SLOW
                action_text   = "SLOW DOWN"
                hud_color     = (0, 255, 255)
                global_action = "SLOW"
        elif current_state == State.OBSERVE:
            action_text   = "WAITING TO CONFIRM STATIC"
            hud_color     = (0, 0, 255)
            global_action = "STOP"
            if is_dynamic:
                current_state = State.STOP_DYNAMIC
                action_text   = "STOP (DYNAMIC THREAT)"
            elif time.time() - state_start_time > TIME_OBSERVE:
                current_state    = State.DECIDE
                state_start_time = time.time()
        elif current_state == State.STOP_DYNAMIC:
            action_text   = "STOP (WAITING FOR CLEAR PATH)"
            hud_color     = (0, 0, 255)
            global_action = "STOP"
            if not is_dynamic and closing_speed < 0.02:
                current_state = State.FOLLOW
        elif current_state == State.DECIDE:
            overtake_direction = "LEFT" if critical_obstacle['center_x'] > CX else "RIGHT"
            x1, y1, x2, y2        = critical_obstacle['box']
            obstacle_width_meters  = (abs(x2 - x1) * critical_obstacle['dist']) / FX
            dynamic_lateral_offset = (obstacle_width_meters / 2) + 0.20
            bumper_dist  = max(0.01, critical_obstacle['dist'] - CAMERA_BUMPER_OFFSET_M)
            overtake_angle = min(math.degrees(math.atan2(dynamic_lateral_offset, bumper_dist)), MAX_STEERING_ANGLE)
            current_state    = State.OVERTAKE
            state_start_time = time.time()
            action_text      = "DECIDING DIRECTION"
            global_action    = "STOP"
    else:
        if current_state in [State.STOP_DYNAMIC, State.OBSERVE]:
            current_state = State.FOLLOW
        elif current_state == State.DECIDE:
            if current_time - last_obstacle_seen_time > BLIND_SPOT_TIMEOUT:
                current_state = State.FOLLOW

    sys_logger.log_frame(frame_count, current_fps, len(all_detections), global_action)

    for d in all_detections:
        x1, y1, x2, y2 = d['box']
        is_crit = (critical_obstacle and d['id'] == critical_obstacle['id'])
        if frame_count % SKIP_FRAMES == 0:
            color = hud_color if is_crit else (200, 200, 200) 
        else:
            color = (0, 255, 255) if is_crit else (0, 100, 100) 

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, d['fp'], 5, (0, 0, 255), -1)

        dist_cm = d['dist'] * 100
        state_str = "DYN" if d['is_dynamic'] else "STAT"
        data_str = f"Dist:{dist_cm:.1f}cm | TTC:{d['ttc']:.1f}s | {state_str}"
        draw_outlined_text(frame, data_str, (x1, y1 - 5), 0.4, color) 

    draw_outlined_text(frame, "LIVE DIORAMA TEST (UBUNTU)", (10, 20), 0.6, (255, 255, 255))
    draw_outlined_text(frame, f"STATE: {current_state}", (10, 45), 0.6, (255, 255, 255))
    draw_outlined_text(frame, action_text, (10, 70), 0.6, hud_color)
    if show_speed:
        draw_outlined_text(frame, f"{speed_label}: {display_speed_cm_s:.1f} cm/s", (10, 95), 0.5, (0, 255, 255))
    draw_outlined_text(frame, f"FPS: {current_fps:.1f} | dt: {dt*1000:.1f}ms", (10, 120), 0.5, (200, 200, 200))

    # Write directly to RAM file instead of a Python array
    video_writer.write(frame)
    cv2.imshow("Optimized Feed Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    prev_time = current_time

# Teardown logic
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"\nRun finished. The video has been saved temporarily to the Pi's RAM.")
print(f"You can manually copy it from this location: {ram_video_path}")
print("NOTE: Make sure to copy it to your SD card before turning off the Pi, as RAM is wiped on shutdown.")

total_elapsed = time.time() - run_start_time
avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0

sys_logger.close(frame_count, avg_fps)
print("Log files generated successfully.")