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

# Seconds before aborting DECIDE if the obstacle stays invisible (blind-spot guard).
BLIND_SPOT_TIMEOUT = 2.0

<<<<<<< HEAD
# --- HARDWARE GEOMETRY CONSTANTS ---
CAMERA_BUMPER_OFFSET_M = 0.16   # Distance from camera lens to front bumper (m)
MAX_STEERING_ANGLE     = 40.0   # Hard cap on computed steering angle (degrees)

=======
>>>>>>> main
# --- EVALUATION CONFIGURATION ---
# Set to the known ground-truth distance (m) for calibration clips.
# Set to None for dynamic/live runs to skip accuracy metrics.
EVAL_GROUND_TRUTH_M = 0.50

class State:
    FOLLOW = "FOLLOW_GLOBAL_PATH"
    SLOW = "LOCAL_SLOW"
    OBSERVE = "LOCAL_AVOID_STATIC (OBSERVE)"
    DECIDE = "LOCAL_AVOID_STATIC (DECIDE)"
    OVERTAKE = "LOCAL_AVOID_STATIC (OVERTAKE)"
    REJOIN = "REJOIN_PATH"
    STOP_DYNAMIC = "LOCAL_AVOID_DYNAMIC (STOP)"


# --- CLASSES ---

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
        
        # Dictionary to store distance histories for evaluation metrics
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

        # Save distance history for evaluation
        tid = obj_data['id']
        if tid not in self.eval_dist_histories:
            self.eval_dist_histories[tid] = []
        self.eval_dist_histories[tid].append(obj_data['dist'])

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

        with open(os.path.join(self.run_dir, f"{self.run_name}_decision_log.json"), 'w') as f:
            json.dump(self.decision_events, f, indent=4)

        summary = {
            "total_frames": total_frames,
            "average_fps": round(avg_fps, 2),
            "action_distribution": self.action_counts
        }
        with open(os.path.join(self.run_dir, f"{self.run_name}_clip_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)

        # --- EVALUATION REPORT METRICS ---
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
print("Loading custom weights: best.pt")
model = YOLO('best.pt')

<<<<<<< HEAD
cap = cv2.VideoCapture('for_testing.mp4') # Ensure this matches your camera index
=======
cap = cv2.VideoCapture(1) # Ensure this matches your camera index
>>>>>>> main
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
        print("Camera feed interrupted.")
        break

    current_time = time.time()
    dt = current_time - prev_time
    if dt <= 0:
        dt = 1.0 / 30.0
    current_fps = 1.0 / dt          

    h, w, _ = frame.shape
    frame_count += 1

    results = model.track(frame, persist=True, stream=True, verbose=False)

    critical_obstacle = None
    min_dist = float('inf')
    all_detections = []
    
    display_speed_cm_s = 0.0
    speed_label = ""
    show_speed = False

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
                    dist, closing_speed, ttc, lat_speed = tracker.update(
                        oid, raw_dist, dt, current_state, fp_x
                    )

                    is_dynamic = lat_speed > 40.0 or (
                        current_state in [State.OBSERVE, State.STOP_DYNAMIC]
                        and closing_speed > 0.03
                    )
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

    if critical_obstacle is not None:
        last_obstacle_seen_time = current_time

    action_text = "GO"
    hud_color   = (0, 255, 0)
    global_action = "GO"

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
        display_speed_cm_s = closing_speed * 100
        if display_speed_cm_s < 0:
            display_speed_cm_s = 0.0

        if current_state in [State.FOLLOW, State.SLOW] and not is_dynamic:
            speed_label = "RC Car Speed"
        elif current_state not in [State.FOLLOW, State.SLOW] and is_dynamic:
            speed_label = "Obstacle Approach Speed"
        else:
            speed_label = "Relative Closing Speed"

        if current_state in [State.FOLLOW, State.SLOW]:
            if ttc <= TTC_THRESHOLD or dist <= D_STOP:
                current_state    = State.OBSERVE
                state_start_time = time.time()
                action_text      = "BRAKING TO OBSERVE"
                hud_color        = (0, 0, 255)
                global_action    = "STOP"
                sys_logger.log_decision_event(
                    frame_count, "Distance_or_TTC_Safety", "OBSERVE",
                    critical_obstacle['id']
                )
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
                sys_logger.log_decision_event(
                    frame_count, "Dynamic_Confirmed", "STOP",
                    critical_obstacle['id']
                )
            elif time.time() - state_start_time > TIME_OBSERVE:
                safe_to_overtake = True
                for other_obj in all_detections:
                    if other_obj['id'] != critical_obstacle['id']:
                        if other_obj['is_dynamic'] or other_obj['dist'] < 0.90:
                            safe_to_overtake = False
                            break
                
                if safe_to_overtake:
                    current_state    = State.DECIDE
                    state_start_time = time.time()
                else:
                    action_text = "WAITING: PATH BLOCKED"

        elif current_state == State.STOP_DYNAMIC:
            action_text   = "STOP (WAITING FOR CLEAR PATH)"
            hud_color     = (0, 0, 255)
            global_action = "STOP"
            if not is_dynamic and closing_speed < 0.02:
                current_state = State.FOLLOW

        elif current_state == State.DECIDE:
            overtake_direction = "LEFT" if critical_obstacle['center_x'] > CX else "RIGHT"
            longitudinal_dist  = critical_obstacle['dist']

<<<<<<< HEAD
            # [Groupmate] Dynamic lateral offset from observed pixel width
            x1, y1, x2, y2        = critical_obstacle['box']
            obstacle_width_meters  = (abs(x2 - x1) * longitudinal_dist) / FX
            dynamic_lateral_offset = (obstacle_width_meters / 2) + 0.20

            # [Restored] Measure from bumper, not camera lens, then cap to servo limit
            bumper_dist  = max(0.01, longitudinal_dist - CAMERA_BUMPER_OFFSET_M)
            angle_radians = math.atan2(dynamic_lateral_offset, bumper_dist)
            overtake_angle = min(math.degrees(angle_radians), MAX_STEERING_ANGLE)
=======
            x1, y1, x2, y2       = critical_obstacle['box']
            obstacle_width_meters = (abs(x2 - x1) * longitudinal_dist) / FX
            dynamic_lateral_offset = (obstacle_width_meters / 2) + 0.20

            angle_radians = math.atan2(dynamic_lateral_offset, longitudinal_dist)
            overtake_angle = math.degrees(angle_radians)
>>>>>>> main

            current_state    = State.OVERTAKE
            state_start_time = time.time()
            action_text      = "DECIDING DIRECTION"
            global_action    = "STOP"
            sys_logger.log_decision_event(
                frame_count,
                "Overtake_Planned",
                f"OVERTAKE {overtake_direction} AT {overtake_angle:.1f} DEG "
<<<<<<< HEAD
                f"(offset={dynamic_lateral_offset:.3f}m, bumper_dist={bumper_dist:.3f}m)",
=======
                f"(offset={dynamic_lateral_offset:.3f}m)",
>>>>>>> main
                critical_obstacle['id']
            )

    else:
        if current_state in [State.STOP_DYNAMIC, State.OBSERVE]:
            current_state = State.FOLLOW

        elif current_state == State.DECIDE:
            time_since_seen = current_time - last_obstacle_seen_time
            if time_since_seen > BLIND_SPOT_TIMEOUT:
                sys_logger.log_decision_event(
                    frame_count,
                    "Blind_Spot_Timeout",
                    f"RESET_TO_FOLLOW from {current_state} after "
                    f"{time_since_seen:.1f}s without obstacle",
                    -1
                )
                current_state = State.FOLLOW

    sys_logger.log_frame(frame_count, current_fps, len(all_detections), global_action)

    for d in all_detections:
        x1, y1, x2, y2 = d['box']
        is_crit = (critical_obstacle and d['id'] == critical_obstacle['id'])
        color = hud_color if is_crit else (200, 200, 200)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, d['fp'], 5, (0, 0, 255), -1)

        dist_cm = d['dist'] * 100
        state_str = "DYN" if d['is_dynamic'] else "STAT"
        data_str = f"Dist:{dist_cm:.1f}cm | TTC:{d['ttc']:.1f}s | {state_str}"
        draw_outlined_text(frame, data_str, (x1, y1 - 5), 0.6, color)

    draw_outlined_text(frame, "LIVE DIORAMA TEST (RECORDING)", (20, 40), 1.0, (255, 255, 255))
    draw_outlined_text(frame, f"STATE: {current_state}", (20, 80), 1.0, (255, 255, 255))
    draw_outlined_text(frame, action_text, (20, 120), 1.0, hud_color)
    
    if show_speed:
        draw_outlined_text(frame, f"{speed_label}: {display_speed_cm_s:.1f} cm/s", (20, 160), 0.8, (0, 255, 255))
        
    draw_outlined_text(frame, f"FPS: {current_fps:.1f} | dt: {dt*1000:.1f}ms", (20, 200), 0.7, (200, 200, 200))

    video_writer.write(frame)
    cv2.imshow("Live Diorama Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    prev_time = current_time

cap.release()
video_writer.release()

total_elapsed = time.time() - run_start_time
avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0

sys_logger.close(frame_count, avg_fps)
cv2.destroyAllWindows()
print(f"Live evaluation complete. Video and logs saved in {sys_logger.run_dir}")