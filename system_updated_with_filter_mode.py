import cv2
import math
import time
import numpy as np
import csv
import os
import json
from collections import deque
from ultralytics import YOLO

# =============================================================
# OPTIMIZATION CONFIGURATION
# =============================================================
SKIP_FRAMES = 1    # 1 = YOLO every frame (cleanest for thesis)
                   # 2 = mild FPS boost with KCF on gap frames
TARGET_WIDTH  = 640
TARGET_HEIGHT = 360
TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT   # 1.7778

# Set True when running headless (SSH / no monitor). Skips all imshow overhead.
HEADLESS = False

# Flush CSV to SD card every N frames (avoids per-frame I/O stall).
LOG_FLUSH_INTERVAL = 30

# =============================================================
# FILTER MODE CONFIGURATION
# =============================================================
# Controls which filtering stages are applied to final_dist and closing_speed
# returned by ObjectTracker.update().
#
#   FILTER_MODE = 1  →  Raw GPM distance only (no median, no Kalman)
#   FILTER_MODE = 2  →  Raw GPM distance → Median filter only (no Kalman)
#   FILTER_MODE = 3  →  Raw GPM distance → Median filter → Kalman  (default)
#
# Run the script three times with modes 1, 2, 3 to generate comparative CSVs.
FILTER_MODE = 2

# =============================================================
# CALIBRATION RESOLUTION (must match the NPZ file)
# =============================================================
CALIB_WIDTH  = 1280
CALIB_HEIGHT = 720

# =============================================================
# PHYSICAL CAMERA PARAMETERS
# =============================================================
camera_height_m = 0.23
camera_tilt_deg = 12.0   # Update after running get_tilt.py

try:
    calib_data         = np.load('calibration_params.npz')
    CAMERA_MATRIX_ORIG = calib_data['camera_matrix'].copy()
    DIST_COEFFS        = calib_data['dist_coeffs']
    print("Successfully loaded calibration_params.npz")
except Exception as e:
    print(f"Error loading calibration: {e}")
    exit()

# =============================================================
# FSM / DIORAMA THRESHOLDS
# =============================================================
D_STOP               = 0.52   # Nearest obstacle must be ≤ this to trigger OBSERVE
D_SLOW               = 0.80   # Nearest obstacle ≤ this → SLOW state
TTC_THRESHOLD        = 2.0    # Seconds — triggers OBSERVE if TTC falls below this

# BUG 1 FIX — secondary-obstacle block distance.
# If ANY obstacle (other than the primary) is within this range when the FSM
# is in DECIDE, OVERTAKE is suppressed because the RC car would not physically
# fit through the gap beside the primary obstacle.
D_SECONDARY_BLOCK    = 0.90   # 90 cm

TIME_OBSERVE         = 1.5    # Seconds to watch before deciding static vs dynamic
TIME_TURN            = 1.0
TIME_PASS            = 1.5
TIME_RETURN          = 1.0
MEDIAN_WINDOW        = 5
DIST_ALPHA           = 0.6
SPEED_ALPHA          = 0.3
BLIND_SPOT_TIMEOUT   = 2.0    # Seconds in DECIDE without sighting → abort to FOLLOW

# BUG 3 & 4 FIX — number of consecutive frames an obstacle must read as static
# before it is considered "confirmed static" and safe to overtake.
STATIC_CONFIRM_FRAMES = 5

CAMERA_BUMPER_OFFSET_M = 0.16
MAX_STEERING_ANGLE     = 40.0
EVAL_GROUND_TRUTH_M    = 0.50

# =============================================================
# STATE MACHINE
# =============================================================
class State:
    FOLLOW       = "FOLLOW_GLOBAL_PATH"
    SLOW         = "LOCAL_SLOW"
    OBSERVE      = "LOCAL_AVOID_STATIC (OBSERVE)"
    DECIDE       = "LOCAL_AVOID_STATIC (DECIDE)"
    OVERTAKE     = "LOCAL_AVOID_STATIC (OVERTAKE)"
    REJOIN       = "REJOIN_PATH"
    STOP_DYNAMIC = "LOCAL_AVOID_DYNAMIC (STOP)"

# =============================================================
# KALMAN FILTER
# =============================================================
class KalmanFilter1D:
    def __init__(self, initial_dist):
        self.x = np.array([[initial_dist], [0.0]])
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[0.1]])
        self.Q = np.array([[0.01, 0.0], [0.0, 0.01]])

    def update_and_predict(self, measurement, dt):
        A      = np.array([[1.0, dt], [0.0, 1.0]])
        x_pred = A @ self.x
        P_pred = A @ self.P @ A.T + self.Q
        innov  = measurement - self.H @ x_pred
        S      = self.H @ P_pred @ self.H.T + self.R
        # S is always 1×1 here — scalar reciprocal avoids full LAPACK inv()
        K      = P_pred @ self.H.T * (1.0 / float(S[0, 0]))
        self.x = x_pred + K @ innov
        self.P = P_pred - K @ self.H @ P_pred
        return float(self.x[0][0]), float(self.x[1][0])

# =============================================================
# OBJECT TRACKER  (fixes Bug 3 & Bug 4 + FILTER_MODE support)
# =============================================================
class ObjectTracker:
    """
    Tracks distance, speed, and a rolling dynamic-status history per object ID.

    BUG 3 FIX: 'is_dynamic' from a single frame is unreliable because the
    exponential smoother has momentum; an obstacle that briefly pauses looks
    static for 1–2 frames. We now require STATIC_CONFIRM_FRAMES consecutive
    static readings before declaring an obstacle confirmed-static.

    BUG 4 FIX: When YOLO reassigns a track ID the first call for the new ID
    returns lat_speed=0, making it look static for one frame. The rolling
    history prevents that single frame from poisoning the decision.

    FILTER_MODE behaviour (controls returned final_dist and closing_speed only;
    the full internal pipeline always executes so that track state remains
    consistent regardless of mode):

        FILTER_MODE = 1  →  final_dist = raw GPM distance  |  closing_speed = raw delta
        FILTER_MODE = 2  →  final_dist = median-filtered   |  closing_speed = median delta
        FILTER_MODE = 3  →  final_dist = Kalman-smoothed   |  closing_speed = Kalman speed  (default)
    """
    def __init__(self):
        self.tracks = {}

    def update(self, object_id, raw_distance, dt, current_fsm_state, fp_x):
        current_time = time.time()

        if object_id not in self.tracks:
            # New track — initialise with a dynamic_history that is all-True
            # so the track is treated as "not yet confirmed static" until it
            # has accumulated STATIC_CONFIRM_FRAMES of clean static readings.
            self.tracks[object_id] = {
                'history':         deque(maxlen=MEDIAN_WINDOW),
                'x_history':       deque(maxlen=MEDIAN_WINDOW),
                'exp_dist':        raw_distance,
                'exp_x':           fp_x,
                'speed':           0.0,
                'last_time':       current_time,
                'kalman':          KalmanFilter1D(raw_distance),
                'prev_raw':        raw_distance,    # for FILTER_MODE 1 delta
                'prev_median':     raw_distance,    # for FILTER_MODE 2 delta
                # Rolling window: True = dynamic reading, False = static reading
                'dynamic_history': deque(
                    [True] * STATIC_CONFIRM_FRAMES,
                    maxlen=STATIC_CONFIRM_FRAMES
                ),
            }
            return raw_distance, 0.0, float('inf'), 0.0, False

        track = self.tracks[object_id]

        # ------------------------------------------------------------------
        # Full internal distance pipeline (always runs to keep state fresh)
        # ------------------------------------------------------------------

        # --- Stage 1: Median filter ---
        track['history'].append(raw_distance)
        median_dist = float(np.median(track['history']))

        # --- Stage 2: Exponential smoother (feeds Kalman) ---
        exp_dist    = DIST_ALPHA * median_dist + (1 - DIST_ALPHA) * track['exp_dist']
        track['exp_dist'] = exp_dist

        # --- Stage 3: Kalman filter ---
        kalman_dist, kalman_speed = track['kalman'].update_and_predict(exp_dist, dt)

        # ------------------------------------------------------------------
        # Select final_dist and raw closing_speed based on FILTER_MODE
        # ------------------------------------------------------------------
        if FILTER_MODE == 1:
            # Raw GPM only — no median, no Kalman
            final_dist    = raw_distance
            raw_delta     = (raw_distance - track['prev_raw']) / dt if dt > 0 else 0.0
            new_speed     = SPEED_ALPHA * raw_delta + (1 - SPEED_ALPHA) * track['speed']
            closing_speed = -new_speed
            track['prev_raw'] = raw_distance

        elif FILTER_MODE == 2:
            # Median-filtered only — no Kalman
            final_dist    = median_dist
            median_delta  = (median_dist - track['prev_median']) / dt if dt > 0 else 0.0
            new_speed     = SPEED_ALPHA * median_delta + (1 - SPEED_ALPHA) * track['speed']
            closing_speed = -new_speed
            track['prev_median'] = median_dist

        else:
            # FILTER_MODE == 3 — full pipeline (original behaviour)
            final_dist    = kalman_dist
            new_speed     = SPEED_ALPHA * kalman_speed + (1 - SPEED_ALPHA) * track['speed']
            closing_speed = -new_speed

        # --- Lateral speed ---
        track['x_history'].append(fp_x)
        median_x     = float(np.median(track['x_history']))
        exp_x        = DIST_ALPHA * median_x + (1 - DIST_ALPHA) * track['exp_x']
        lat_speed_px = abs(exp_x - track['exp_x']) / dt if dt > 0 else 0.0
        track['exp_x'] = exp_x

        # --- TTC ---
        ttc = float('inf')
        if closing_speed > 0.01:
            ttc = final_dist / closing_speed

        # --- Raw dynamic flag for this frame ---
        raw_dynamic = (
            lat_speed_px > 40.0
            or (current_fsm_state in [State.OBSERVE, State.STOP_DYNAMIC]
                and closing_speed > 0.03)
        )

        # Append this frame's reading to the rolling history
        track['dynamic_history'].append(raw_dynamic)

        # BUG 3 & 4 FIX:
        # confirmed_dynamic = True if ANY recent frame was dynamic
        confirmed_dynamic = any(track['dynamic_history'])

        track['speed']     = new_speed
        track['last_time'] = current_time

        return final_dist, closing_speed, ttc, lat_speed_px, confirmed_dynamic

# =============================================================
# DISTANCE FILTER LOGGER
# =============================================================
class DistanceFilterLogger:
    """
    Lightweight, standalone logger that records per-detection distance data
    for a single FILTER_MODE run.  Completely independent of SystemLogger so
    the three resulting CSVs can be compared directly without any cross-
    contamination from the main logging pipeline.

    Output file:  logs/distance_filter_log_<run_timestamp>.csv
    Columns:      frame_id, track_id, filter_mode, final_calculated_distance
    """
    def __init__(self, run_timestamp, filter_mode):
        self.filter_mode = filter_mode
        os.makedirs("logs", exist_ok=True)
        filename  = f"distance_filter_log_{run_timestamp}.csv"
        filepath  = os.path.join("logs", filename)
        self._file   = open(filepath, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(["frame_id", "track_id", "filter_mode",
                                "final_calculated_distance"])
        self._flush_counter = 0
        print(f"  Distance filter log → {filepath}")

    def log(self, frame_id, track_id, final_dist):
        self._writer.writerow([
            frame_id,
            track_id,
            self.filter_mode,
            round(final_dist, 6),
        ])
        self._flush_counter += 1
        if self._flush_counter >= LOG_FLUSH_INTERVAL:
            self._file.flush()
            self._flush_counter = 0

    def close(self):
        self._file.flush()
        self._file.close()

# =============================================================
# SYSTEM LOGGER
# =============================================================
class SystemLogger:
    def __init__(self, run_name="live_run"):
        self.run_name     = run_name
        self.run_dir      = os.path.join("logs", run_name)
        os.makedirs("logs",          exist_ok=True)
        os.makedirs(self.run_dir,    exist_ok=True)
        self.frame_file   = open(os.path.join(self.run_dir, f"{run_name}_frame_log.csv"),  'w', newline='')
        self.obj_file     = open(os.path.join(self.run_dir, f"{run_name}_object_log.csv"), 'w', newline='')
        self.frame_writer = csv.writer(self.frame_file)
        self.obj_writer   = csv.writer(self.obj_file)
        self.frame_writer.writerow(["frame_id", "timestamp", "fps", "total_detections", "global_action"])
        self.obj_writer.writerow(  ["frame_id", "track_id",  "class", "dist_m", "speed_ms", "ttc_s", "state"])
        self.decision_events     = []
        self.action_counts       = {"GO": 0, "SLOW": 0, "STOP": 0, "OVERTAKE": 0}
        self.eval_dist_histories = {}
        self._flush_counter      = 0

    def log_frame(self, frame_id, fps, num_detections, global_action):
        self.frame_writer.writerow([frame_id, time.time(), round(fps, 2),
                                    num_detections, global_action])
        if global_action in self.action_counts:
            self.action_counts[global_action] += 1
        self._flush_counter += 1
        if self._flush_counter >= LOG_FLUSH_INTERVAL:
            self.frame_file.flush()
            self.obj_file.flush()
            self._flush_counter = 0

    def log_object(self, frame_id, obj_data):
        state_str = "DYNAMIC" if obj_data['is_dynamic'] else "STATIC"
        self.obj_writer.writerow([
            frame_id, obj_data['id'], obj_data['label'],
            round(obj_data['dist'],  3), round(obj_data['speed'], 3),
            round(obj_data['ttc'],   2), state_str
        ])
        tid = obj_data['id']
        if tid not in self.eval_dist_histories:
            self.eval_dist_histories[tid] = []
        self.eval_dist_histories[tid].append(obj_data['dist'])

    def log_decision_event(self, frame_id, trigger_reason, action, obj_id):
        self.decision_events.append({
            "time": time.time(), "frame": frame_id,
            "trigger": trigger_reason, "action": action,
            "target_track_id": obj_id
        })

    def close(self, total_frames, avg_fps):
        self.frame_file.close()
        self.obj_file.close()
        with open(os.path.join(self.run_dir, f"{self.run_name}_decision_log.json"), 'w') as f:
            json.dump(self.decision_events, f, indent=4)
        summary = {
            "total_frames":        total_frames,
            "average_fps":         round(avg_fps, 2),
            "action_distribution": self.action_counts
        }
        with open(os.path.join(self.run_dir, f"{self.run_name}_clip_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)
        report = {
            "run_name":       self.run_name,
            "filter_mode":    FILTER_MODE,
            "total_frames":   total_frames,
            "average_fps":    round(avg_fps, 2),
            "ground_truth_m": EVAL_GROUND_TRUTH_M,
        }
        EVAL_MAX_RELIABLE_DIST_M = 0.85
        EVAL_PLAUSIBILITY_BAND_M = 0.40
        if EVAL_GROUND_TRUTH_M is not None and self.eval_dist_histories:
            candidate_tracks = {
                tid: dists for tid, dists in self.eval_dist_histories.items()
                if dists
                and float(np.mean(dists)) <= EVAL_MAX_RELIABLE_DIST_M
                and abs(float(np.mean(dists)) - EVAL_GROUND_TRUTH_M) <= EVAL_PLAUSIBILITY_BAND_M
            }
            if candidate_tracks:
                primary_tid   = max(candidate_tracks, key=lambda t: len(candidate_tracks[t]))
                primary_dists = candidate_tracks[primary_tid]
                errors    = [abs(d - EVAL_GROUND_TRUTH_M) for d in primary_dists]
                sq_errors = [e ** 2 for e in errors]

                # Thesis table metrics — computed directly on the primary track array
                # dist_variance_m2 : overall spread of the distance readings (not delta spread)
                # max_ftf_delta_m  : worst-case single-frame jump (key stability indicator for
                #                    a stationary target — any value > 0 is measurement noise)
                primary_ftf_abs  = [abs(primary_dists[i] - primary_dists[i - 1])
                                    for i in range(1, len(primary_dists))]
                dist_variance_m2 = float(np.var(primary_dists))
                max_ftf_delta_m  = float(max(primary_ftf_abs)) if primary_ftf_abs else 0.0

                # --- Thesis-ready summary (copy directly into table) ---
                report["thesis_metrics"] = {
                    "filter_mode":       FILTER_MODE,
                    "MAE_m":             round(float(np.mean(errors)), 6),
                    "dist_variance_m2":  round(dist_variance_m2,       6),
                    "max_ftf_delta_m":   round(max_ftf_delta_m,        6),
                }

                report["distance_accuracy"] = {
                    "primary_track_id":        primary_tid,
                    "candidates_after_filter": len(candidate_tracks),
                    "tracks_rejected":         len(self.eval_dist_histories) - len(candidate_tracks),
                    "sample_count":            len(primary_dists),
                    "mean_dist_m":             round(float(np.mean(primary_dists)), 4),
                    "std_dist_m":              round(float(np.std(primary_dists)),  4),
                    "MAE_m":                   round(float(np.mean(errors)),                    4),
                    "RMSE_m":                  round(float(np.sqrt(np.mean(sq_errors))),        4),
                    "p95_error_m":             round(float(np.percentile(errors, 95)),          4),
                }
            else:
                report["distance_accuracy"] = {
                    "error":                   "no_plausible_track_found",
                    "max_reliable_dist_m": EVAL_MAX_RELIABLE_DIST_M,
                    "plausibility_band_m": EVAL_PLAUSIBILITY_BAND_M,
                    "ground_truth_m":      EVAL_GROUND_TRUTH_M,
                    "track_mean_dists":    {
                        str(tid): round(float(np.mean(d)), 4)
                        for tid, d in self.eval_dist_histories.items() if d
                    },
                }
        else:
            report["distance_accuracy"] = (
                "skipped_no_ground_truth" if EVAL_GROUND_TRUTH_M is None
                else "no_detections_logged"
            )
        track_stability = {}
        for tid, dists in self.eval_dist_histories.items():
            lifespan = len(dists)
            if lifespan >= 2:
                diffs            = [dists[i] - dists[i - 1] for i in range(1, lifespan)]
                ftf_abs          = [abs(d) for d in diffs]
                # dist_variance_m2: variance of the distance readings themselves
                # (how spread the measurements are around their mean)
                dist_variance_m2 = float(np.var(dists))
                ftf_std          = float(np.std(diffs))
                max_ftf_delta_m  = float(max(ftf_abs))
            else:
                dist_variance_m2 = ftf_std = max_ftf_delta_m = None
            track_stability[str(tid)] = {
                "lifespan_frames":  lifespan,
                "dist_variance_m2": round(dist_variance_m2, 6) if dist_variance_m2 is not None else None,
                "ftf_std_m":        round(ftf_std,          6) if ftf_std           is not None else None,
                "max_ftf_delta_m":  round(max_ftf_delta_m,  6) if max_ftf_delta_m   is not None else None,
            }
        valid_vars = [v["dist_variance_m2"] for v in track_stability.values()
                      if v["dist_variance_m2"] is not None]
        report["temporal_stability"] = {
            "per_track":                  track_stability,
            "total_unique_track_ids":     len(self.eval_dist_histories),
            "mean_track_lifespan_frames": round(float(np.mean(
                [v["lifespan_frames"] for v in track_stability.values()])), 2)
                if track_stability else 0.0,
            "mean_dist_variance_m2": round(float(np.mean(valid_vars)), 6) if valid_vars else None,
        }
        report_path = os.path.join(self.run_dir, f"{self.run_name}_evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"\n--- Evaluation Report Saved: {report_path} ---")

# =============================================================
# HARDWARE HELPERS
# =============================================================
def get_hardware_frame_dimensions(cap, num_probe_frames=5):
    widths, heights = [], []
    for _ in range(num_probe_frames):
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            widths.append(w);  heights.append(h)
    if not widths:
        return None, None
    return int(np.median(widths)), int(np.median(heights))

def build_scaled_camera_matrix(cam_matrix_orig, hw_w, hw_h,
                                calib_w, calib_h, target_w, target_h):
    hw_aspect  = hw_w / hw_h
    tgt_aspect = target_w / target_h
    tol        = 0.02
    if abs(hw_aspect - tgt_aspect) < tol:
        crop_x, crop_y, crop_w, crop_h = 0, 0, hw_w, hw_h
        crop_note = "none (aspect ratio matches)"
    elif hw_aspect < tgt_aspect:
        crop_w = hw_w
        crop_h = int(round(hw_w / tgt_aspect));  crop_h -= crop_h % 2
        crop_x = 0;  crop_y = (hw_h - crop_h) // 2
        crop_note = f"vertical crop: {hw_h - crop_h}px removed"
    else:
        crop_h = hw_h
        crop_w = int(round(hw_h * tgt_aspect));  crop_w -= crop_w % 2
        crop_x = (hw_w - crop_w) // 2;  crop_y = 0
        crop_note = f"horizontal crop: {hw_w - crop_w}px removed"
    scale_x = target_w / calib_w
    scale_y = target_h / calib_h
    off_x   = crop_x * (calib_w / hw_w)
    off_y   = crop_y * (calib_h / hw_h)
    M  = cam_matrix_orig.copy()
    fx = M[0,0] * scale_x
    fy = M[1,1] * scale_y
    cx = (M[0,2] - off_x) * scale_x
    cy = (M[1,2] - off_y) * scale_y
    scaled = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float64)
    crop_params = {'x': crop_x, 'y': crop_y, 'w': crop_w, 'h': crop_h}
    diag = {
        "calibration_resolution": f"{calib_w}x{calib_h}",
        "hardware_delivered":     f"{hw_w}x{hw_h}",
        "crop_applied":           crop_note,
        "pipeline_resolution":    f"{target_w}x{target_h}",
        "scale_x": round(scale_x, 6), "scale_y": round(scale_y, 6),
        "FX": round(fx, 4), "FY": round(fy, 4),
        "CX": round(cx, 4), "CY": round(cy, 4),
    }
    return scaled, crop_params, diag

def apply_crop(frame, cp):
    return frame[cp['y']:cp['y']+cp['h'], cp['x']:cp['x']+cp['w']]

def calculate_calibrated_distance(fp_x, fp_y, h, tilt_deg,
                                   cam_mat, dist_coeffs, cx, cy, fy):
    pt  = np.array([[[float(fp_x), float(fp_y)]]], dtype=np.float32)
    upt = cv2.undistortPoints(pt, cam_mat, dist_coeffs, P=cam_mat)
    v   = upt[0][0][1]
    ang = math.radians(tilt_deg) + math.atan((v - cy) / fy)
    return h / math.tan(ang) if ang > 0 else float('inf')

def draw_outlined_text(img, text, pos, scale, color):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,   2)

# =============================================================
# STARTUP
# =============================================================
print("Loading NCNN model: best_ncnn_model")
model = YOLO(r'C:\Users\Paul\thesis\AV-Robots-Distance_Detection-Mandap (1)\AV-Robots-Distance_Detection-Mandap\best_ncnn_model')

cap = cv2.VideoCapture(0) # 1, cv2.CAP_V4L2 kapag nasa raspi na
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)  # Ignored by MJPEG stream; set on Pi side
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Ignored by MJPEG stream; set on Pi side
cap.set(cv2.CAP_PROP_FPS,          30)    # Ignored by MJPEG stream; set on Pi side
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)     # Prevents stale-frame drain

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("\n--- Hardware Verification ---")
HW_WIDTH, HW_HEIGHT = get_hardware_frame_dimensions(cap, num_probe_frames=5)
if HW_WIDTH is None:
    print("FATAL: No frames from camera.")
    cap.release();  exit()

rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"  Requested : {rw}x{rh}")
print(f"  Delivered : {HW_WIDTH}x{HW_HEIGHT}")
if HW_WIDTH != rw or HW_HEIGHT != rh:
    print("  ⚠  Mismatch — using delivered dimensions for matrix scaling.")

CAMERA_MATRIX, CROP_PARAMS, diag = build_scaled_camera_matrix(
    CAMERA_MATRIX_ORIG, HW_WIDTH, HW_HEIGHT,
    CALIB_WIDTH, CALIB_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT
)
CX = diag["CX"];  CY = diag["CY"]
FX = diag["FX"];  FY = diag["FY"]

print("\n--- Camera Matrix Diagnostic ---")
for k, v in diag.items():
    print(f"  {k:<30}: {v}")

# NOTE: hw_diagnostic JSON file intentionally removed — diagnostic data is
# fully visible in the terminal output above. File I/O on every run was
# unnecessary overhead.

needs_crop = (CROP_PARAMS['x'] != 0 or CROP_PARAMS['y'] != 0
              or CROP_PARAMS['w'] != HW_WIDTH or CROP_PARAMS['h'] != HW_HEIGHT)

print(f"\n  Crop     : {'ENABLED — ' + diag['crop_applied'] if needs_crop else 'not needed'}")
print(f"\n--- Starting Live Feed ---")
print(f"  FILTER_MODE={FILTER_MODE} | SKIP_FRAMES={SKIP_FRAMES} | HEADLESS={HEADLESS} | BUFFERSIZE=1")
print(f"  D_STOP={D_STOP}m | D_SLOW={D_SLOW}m | D_SECONDARY_BLOCK={D_SECONDARY_BLOCK}m")
print(f"  STATIC_CONFIRM_FRAMES={STATIC_CONFIRM_FRAMES}")
print("  Press 'q' to stop.\n")

# =============================================================
# RUNTIME STATE
# =============================================================
run_timestamp = time.strftime("%Y%m%d-%H%M%S")
os.makedirs("logs", exist_ok=True)

sys_logger      = SystemLogger(f"live_run_{run_timestamp}")
dist_filter_log = DistanceFilterLogger(run_timestamp, FILTER_MODE)
tracker         = ObjectTracker()
active_trackers = {}

current_state           = State.FOLLOW
state_start_time        = 0.0
overtake_direction      = "NONE"
overtake_angle          = 0.0
last_obstacle_seen_time = time.time()
frame_count             = 0
run_start_time          = time.time()
prev_time               = time.time()

# =============================================================
# MAIN LOOP
# =============================================================
while True:
    success, raw_frame = cap.read()
    if not success:
        break

    if needs_crop:
        raw_frame = apply_crop(raw_frame, CROP_PARAMS)

    frame = cv2.resize(raw_frame, (TARGET_WIDTH, TARGET_HEIGHT),
                       interpolation=cv2.INTER_NEAREST)

    current_time = time.time()
    dt  = current_time - prev_time
    if dt <= 0:
        dt = 1.0 / 30.0
    current_fps = 1.0 / dt

    frame_count   += 1
    all_detections = []

    # ----------------------------------------------------------
    # DETECTION — YOLO or KCF depending on SKIP_FRAMES
    # ----------------------------------------------------------
    if frame_count % SKIP_FRAMES == 0 or frame_count == 1:
        results = model.track(frame, persist=True, stream=True, verbose=False)
        active_trackers.clear()
        for r in results:
            if r.boxes.id is None:
                continue
            track_ids = r.boxes.id.int().cpu().tolist()
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                oid        = track_ids[i]
                label_name = model.names[int(box.cls[0])]
                bbox       = (x1, y1, x2 - x1, y2 - y1)
                # KCF GUARD — only pay the init cost when KCF frames will
                # actually occur (SKIP_FRAMES > 1).
                if SKIP_FRAMES > 1:
                    try:
                        kcf = cv2.TrackerKCF.create()
                    except AttributeError:
                        try:
                            kcf = cv2.legacy.TrackerKCF_create()
                        except AttributeError:
                            kcf = cv2.TrackerKCF_create()
                    kcf.init(frame, bbox)
                    active_trackers[oid] = {'tracker': kcf, 'label': label_name}
                fp_x = int((x1 + x2) / 2);  fp_y = int(y2)
                raw_dist = calculate_calibrated_distance(
                    fp_x, fp_y, camera_height_m, camera_tilt_deg,
                    CAMERA_MATRIX, DIST_COEFFS, CX, CY, FY)
                if raw_dist != float('inf'):
                    dist, closing_speed, ttc, lat_speed, is_dynamic = tracker.update(
                        oid, raw_dist, dt, current_state, fp_x)
                    # Log this detection's filtered distance independently
                    dist_filter_log.log(frame_count, oid, dist)
                    all_detections.append({
                        'id': oid, 'label': label_name, 'box': (x1, y1, x2, y2),
                        'fp': (fp_x, fp_y), 'dist': dist, 'speed': closing_speed,
                        'ttc': ttc, 'lat_speed': lat_speed, 'center_x': fp_x,
                        'is_dynamic': is_dynamic
                    })
    else:
        for oid, info in list(active_trackers.items()):
            ok, bbox = info['tracker'].update(frame)
            if not ok:
                del active_trackers[oid]
                continue
            x, y, wb, hb   = map(int, bbox)
            x1, y1, x2, y2 = x, y, x+wb, y+hb
            fp_x = int((x1+x2)/2);  fp_y = int(y2)
            raw_dist = calculate_calibrated_distance(
                fp_x, fp_y, camera_height_m, camera_tilt_deg,
                CAMERA_MATRIX, DIST_COEFFS, CX, CY, FY)
            if raw_dist != float('inf'):
                dist, closing_speed, ttc, lat_speed, is_dynamic = tracker.update(
                    oid, raw_dist, dt, current_state, fp_x)
                # Log this detection's filtered distance independently
                dist_filter_log.log(frame_count, oid, dist)
                all_detections.append({
                    'id': oid, 'label': info['label'], 'box': (x1, y1, x2, y2),
                    'fp': (fp_x, fp_y), 'dist': dist, 'speed': closing_speed,
                    'ttc': ttc, 'lat_speed': lat_speed, 'center_x': fp_x,
                    'is_dynamic': is_dynamic
                })

    # ----------------------------------------------------------
    # FIND CRITICAL OBSTACLE (nearest)
    # ----------------------------------------------------------
    critical_obstacle = None
    min_dist          = float('inf')
    for d in all_detections:
        sys_logger.log_object(frame_count, d)
        if d['dist'] < min_dist:
            min_dist = d['dist']
            critical_obstacle = d

    if critical_obstacle is not None:
        last_obstacle_seen_time = current_time

    # ----------------------------------------------------------
    # 3-TIER FSM  (all 5 bugs addressed)
    # ----------------------------------------------------------
    action_text        = "GO"
    hud_color          = (0, 255, 0)
    global_action      = "GO"
    show_speed         = False
    display_speed_cm_s = 0.0

    if current_state == State.OVERTAKE:
        elapsed       = time.time() - state_start_time
        hud_color     = (255, 165, 0)
        global_action = "OVERTAKE"
        if elapsed < TIME_TURN:
            action_text = f"TURN {overtake_direction} {overtake_angle:.1f} DEG"
        elif elapsed < TIME_TURN + TIME_PASS:
            action_text = "DRIVE STRAIGHT"
        elif elapsed < TIME_TURN + TIME_PASS + TIME_RETURN:
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
        show_speed    = True
        display_speed_cm_s = max(0.0, closing_speed * 100)

        # --- FOLLOW / SLOW ---
        if current_state in [State.FOLLOW, State.SLOW]:
            if ttc <= TTC_THRESHOLD or dist <= D_STOP:
                current_state    = State.OBSERVE
                state_start_time = time.time()
                action_text      = "BRAKING TO OBSERVE"
                hud_color        = (0, 0, 255)
                global_action    = "STOP"
                sys_logger.log_decision_event(
                    frame_count, "Distance_or_TTC_Safety", "OBSERVE",
                    critical_obstacle['id'])
            elif dist <= D_SLOW:
                current_state = State.SLOW
                action_text   = "SLOW DOWN"
                hud_color     = (0, 255, 255)
                global_action = "SLOW"

        # --- OBSERVE ---
        # BUG 3 & 4 FIX: is_dynamic now uses confirmed_dynamic (rolling window).
        elif current_state == State.OBSERVE:
            action_text   = "WAITING TO CONFIRM STATIC"
            hud_color     = (0, 0, 255)
            global_action = "STOP"
            if is_dynamic:
                current_state = State.STOP_DYNAMIC
                action_text   = "STOP (DYNAMIC THREAT)"
                sys_logger.log_decision_event(
                    frame_count, "Dynamic_Confirmed", "STOP_DYNAMIC",
                    critical_obstacle['id'])
            elif time.time() - state_start_time > TIME_OBSERVE:
                current_state    = State.DECIDE
                state_start_time = time.time()
                sys_logger.log_decision_event(
                    frame_count, "Static_Confirmed_EnterDecide", "DECIDE",
                    critical_obstacle['id'])

        # --- STOP_DYNAMIC ---
        elif current_state == State.STOP_DYNAMIC:
            action_text   = "STOP (WAITING FOR CLEAR PATH)"
            hud_color     = (0, 0, 255)
            global_action = "STOP"
            if not is_dynamic and closing_speed < 0.02:
                current_state = State.FOLLOW
                sys_logger.log_decision_event(
                    frame_count, "Dynamic_Cleared", "FOLLOW",
                    critical_obstacle['id'])

        # --- DECIDE ---
        # BUG 1 FIX: Check every obstacle in all_detections before allowing OVERTAKE.
        # BUG 2 FIX: DECIDE is now a persistent state — re-evaluates every frame.
        # BUG 5 FIX: Re-confirmation of static state happens here on every frame.
        elif current_state == State.DECIDE:
            hud_color     = (0, 0, 255)
            global_action = "STOP"
            any_dynamic = any(d['is_dynamic'] for d in all_detections)
            if any_dynamic:
                current_state = State.STOP_DYNAMIC
                action_text   = "STOP (DYNAMIC APPEARED IN DECIDE)"
                sys_logger.log_decision_event(
                    frame_count, "Dynamic_In_Decide_Abort", "STOP_DYNAMIC",
                    critical_obstacle['id'])
            else:
                secondary_too_close = any(
                    d['id'] != critical_obstacle['id']
                    and d['dist'] <= D_SECONDARY_BLOCK
                    for d in all_detections
                )
                if secondary_too_close:
                    action_text = "WAITING: SECONDARY OBSTACLE BLOCKING GAP"
                    sys_logger.log_decision_event(
                        frame_count, "Secondary_Block_Waiting", "HOLD_DECIDE",
                        critical_obstacle['id'])
                else:
                    overtake_direction     = "LEFT" if critical_obstacle['center_x'] > CX else "RIGHT"
                    x1, y1, x2, y2        = critical_obstacle['box']
                    obs_width_m            = (abs(x2 - x1) * critical_obstacle['dist']) / FX
                    lateral_offset         = (obs_width_m / 2) + 0.20
                    bumper_dist            = max(0.01, critical_obstacle['dist'] - CAMERA_BUMPER_OFFSET_M)
                    overtake_angle         = min(
                        math.degrees(math.atan2(lateral_offset, bumper_dist)),
                        MAX_STEERING_ANGLE)
                    current_state    = State.OVERTAKE
                    state_start_time = time.time()
                    action_text      = "DECIDING DIRECTION"
                    sys_logger.log_decision_event(
                        frame_count,
                        "Overtake_Planned",
                        f"OVERTAKE {overtake_direction} AT {overtake_angle:.1f} DEG "
                        f"(lateral_offset={lateral_offset:.3f}m)",
                        critical_obstacle['id'])

    # --- No obstacle visible ---
    else:
        if current_state in [State.STOP_DYNAMIC, State.OBSERVE]:
            current_state = State.FOLLOW
        elif current_state == State.DECIDE:
            if current_time - last_obstacle_seen_time > BLIND_SPOT_TIMEOUT:
                sys_logger.log_decision_event(
                    frame_count, "Blind_Spot_Timeout", "RESET_TO_FOLLOW", -1)
                current_state = State.FOLLOW

    sys_logger.log_frame(frame_count, current_fps, len(all_detections), global_action)

    # ----------------------------------------------------------
    # HUD
    # ----------------------------------------------------------
    if not HEADLESS:
        for d in all_detections:
            x1, y1, x2, y2 = d['box']
            is_crit = critical_obstacle and d['id'] == critical_obstacle['id']
            color   = hud_color if is_crit else (200, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, d['fp'], 5, (0, 0, 255), -1)
            state_str = "DYN" if d['is_dynamic'] else "STAT"
            draw_outlined_text(
                frame,
                f"Dist:{d['dist']*100:.1f}cm | TTC:{d['ttc']:.1f}s | {state_str}",
                (x1, y1 - 5), 0.4, color)
        draw_outlined_text(frame, "LIVE DIORAMA TEST",           (10, 20),  0.6, (255, 255, 255))
        draw_outlined_text(frame, f"STATE: {current_state}",     (10, 45),  0.6, (255, 255, 255))
        draw_outlined_text(frame, action_text,                   (10, 70),  0.6, hud_color)
        draw_outlined_text(frame, f"FILTER_MODE: {FILTER_MODE}", (10, 95),  0.5, (200, 200, 255))
        if show_speed:
            draw_outlined_text(frame,
                               f"Closing Speed: {display_speed_cm_s:.1f} cm/s",
                               (10, 120), 0.5, (0, 255, 255))
        draw_outlined_text(frame,
                           f"FPS: {current_fps:.1f} | dt: {dt*1000:.1f}ms",
                           (10, 145), 0.5, (200, 200, 200))
        draw_outlined_text(frame,
                           f"HW:{HW_WIDTH}x{HW_HEIGHT} | Crop:{needs_crop}",
                           (10, TARGET_HEIGHT - 10), 0.4, (180, 180, 180))
        cv2.imshow("Optimized Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count:>5} | FPS:{current_fps:5.1f} | "
                  f"Det:{len(all_detections)} | State:{current_state} | FilterMode:{FILTER_MODE}")

    prev_time = current_time

# =============================================================
# SHUTDOWN
# =============================================================
cap.release()
if not HEADLESS:
    cv2.destroyAllWindows()

total_elapsed = time.time() - run_start_time
avg_fps       = frame_count / total_elapsed if total_elapsed > 0 else 0.0

dist_filter_log.close()
sys_logger.close(frame_count, avg_fps)