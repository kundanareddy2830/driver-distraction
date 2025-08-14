import cv2
import numpy as np
import mediapipe as mp
import time
import winsound
import threading
from scipy.spatial import distance as dist

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- TUNABLE PARAMETERS ---
# Head Pose
HEAD_PITCH_UP_THRESHOLD = -15
HEAD_PITCH_DOWN_THRESHOLD = 25
HEAD_YAW_THRESHOLD = 25
POSE_TIME_THRESHOLD = 1.5
# Eye/Cognitive
GAZE_SIDE_THRESHOLD = 0.4
GAZE_DOWN_THRESHOLD = 0.65  
COGNITIVE_TIME_THRESHOLD = 8
DROWSY_TIME_THRESHOLD = 1.5
GAZE_TIME_THRESHOLD = 2.0
# Alerting
WARNING_DURATION = 5.0 # --- CHANGED: Universal 5-second warning delay ---
BEEP_COOLDOWN = 5.0

# --- Landmark Indices ---
HEAD_POSE_LANDMARKS = [1, 152, 226, 446, 57, 287]
LEFT_EYE, RIGHT_EYE = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]
LEFT_IRIS, RIGHT_IRIS = [468, 469, 470, 471], [473, 474, 475, 476]

# --- Helper Functions ---
def euclidean(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))
def calculate_ear(eye): return (euclidean(eye[1], eye[5]) + euclidean(eye[2], eye[4])) / (2.0 * euclidean(eye[0], eye[3]))

def get_head_pose(face_landmarks_in_pixels, cam_matrix, dist_coeffs):
    model_points = np.array([(0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)], dtype=np.float64)
    success, rvec, _ = cv2.solvePnP(model_points, face_landmarks_in_pixels, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if success:
        R, _ = cv2.Rodrigues(rvec)
        # --- BUG FIX: Changed *2 to **2 for correct squaring ---
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        if sy >= 1e-6:
            pitch, yaw = np.rad2deg(np.arctan2(R[2, 1], R[2, 2])), np.rad2deg(np.arctan2(-R[2, 0], sy))
            return pitch, yaw
    return None, None

def get_gaze_direction(left_eye, right_eye, left_iris, right_iris):
    def get_norm_pos(eye, iris):
        x_min, x_max = min(pt[0] for pt in eye), max(pt[0] for pt in eye)
        y_min, y_max = min(pt[1] for pt in eye), max(pt[1] for pt in eye)
        iris_center = np.mean(iris, axis=0)
        norm_x = (iris_center[0] - x_min) / (x_max - x_min + 1e-6)
        norm_y = (iris_center[1] - y_min) / (y_max - y_min + 1e-6)
        return norm_x, norm_y
    norm_x_left, norm_y_left = get_norm_pos(left_eye, left_iris)
    norm_x_right, norm_y_right = get_norm_pos(right_eye, right_iris)
    avg_norm_x, avg_norm_y = (norm_x_left + norm_x_right) / 2.0, (norm_y_left + norm_y_right) / 2.0
    if avg_norm_y > GAZE_DOWN_THRESHOLD: return "DOWN"
    elif avg_norm_x < GAZE_SIDE_THRESHOLD: return "LEFT"
    elif avg_norm_x > 1 - GAZE_SIDE_THRESHOLD: return "RIGHT"
    else: return "CENTER"

def play_alert_sound(freq=1200):
    global last_alert_time
    current_time = time.time()
    if (current_time - last_alert_time) > BEEP_COOLDOWN:
        last_alert_time = current_time
        threading.Thread(target=winsound.Beep, args=(freq, 700), daemon=True).start()

# --- Main Program ---
cap = cv2.VideoCapture(0)
camera_matrix, dist_coeffs = None, None
is_calibrated = False
forward_pose_offset = None
distraction_timers = { "HEAD": 0, "DROWSY": 0, "COGNITIVE": 0, "GAZE": 0 }
last_alert_time = 0

# --- State management for the universal warning system ---
current_warning_state = None
warning_start_time = 0
alert_triggered = False

# --- Variables for Unified Calibration ---
calibration_start_time = 0
calibration_duration = 7.0
ear_samples = []
head_pose_samples = []
EAR_THRESHOLD = 0.21 # Default fallback

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    status_text, status_color = "ATTENTIVE", (0, 255, 0)
    
    if results.multi_face_landmarks:
        mesh_points = np.array([(p.x * w, p.y * h) for p in results.multi_face_landmarks[0].landmark])
        
        if camera_matrix is None:
            focal_length = w; center = (w/2, h/2)
            camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # --- PHASE 1: UNIFIED CALIBRATION ---
        if not is_calibrated:
            if calibration_start_time == 0:
                print("Starting unified calibration... Please blink normally for 7 seconds.")
                calibration_start_time = time.time()

            elapsed_time = time.time() - calibration_start_time
            if elapsed_time < calibration_duration:
                # Collect samples for both EAR and Head Pose
                ear_samples.append((calculate_ear(mesh_points[LEFT_EYE]) + calculate_ear(mesh_points[RIGHT_EYE])) / 2.0)
                pitch, yaw = get_head_pose(mesh_points[HEAD_POSE_LANDMARKS].astype(np.float64), camera_matrix, dist_coeffs)
                if pitch is not None:
                    head_pose_samples.append([pitch, yaw])
                
                remaining_time = int(calibration_duration - elapsed_time) + 1
                status_text, status_color = f"CALIBRATING... ({remaining_time}s)", (0, 255, 255)
            else:
                # Calculate final thresholds
                if ear_samples:
                    mean_ear, std_ear = np.mean(ear_samples), np.std(ear_samples)
                    EAR_THRESHOLD = mean_ear - (1.75 * std_ear)
                    print(f"EAR Threshold calibrated to: {EAR_THRESHOLD:.3f}")
                if head_pose_samples:
                    forward_pose_offset = np.mean(head_pose_samples, axis=0)
                    print(f"Forward Pose Offset calibrated to: {forward_pose_offset.round(1)}")
                is_calibrated = True
        else:
            # --- PHASE 2: NORMAL DETECTION ---
            final_distraction = None
            
            avg_ear = (calculate_ear(mesh_points[LEFT_EYE]) + calculate_ear(mesh_points[RIGHT_EYE])) / 2.0
            gaze_direction = get_gaze_direction(mesh_points[LEFT_EYE], mesh_points[RIGHT_EYE], mesh_points[LEFT_IRIS], mesh_points[RIGHT_IRIS])
            pitch, yaw = get_head_pose(mesh_points[HEAD_POSE_LANDMARKS].astype(np.float64), camera_matrix, dist_coeffs)
            
            if avg_ear < EAR_THRESHOLD:
                if distraction_timers["DROWSY"] == 0: distraction_timers["DROWSY"] = time.time()
                if time.time() - distraction_timers["DROWSY"] > DROWSY_TIME_THRESHOLD:
                    final_distraction = "DROWSY"
            else: distraction_timers["DROWSY"] = 0

            if final_distraction is None and pitch is not None and forward_pose_offset is not None:
                relative_pitch = pitch - forward_pose_offset[0]
                relative_yaw = yaw - forward_pose_offset[1]
                if relative_pitch < HEAD_PITCH_UP_THRESHOLD or relative_pitch > HEAD_PITCH_DOWN_THRESHOLD or abs(relative_yaw) > HEAD_YAW_THRESHOLD:
                    if distraction_timers["HEAD"] == 0: distraction_timers["HEAD"] = time.time()
                    if time.time() - distraction_timers["HEAD"] > POSE_TIME_THRESHOLD:
                        final_distraction = "HEAD TURNED AWAY"
                else: distraction_timers["HEAD"] = 0
            
            if final_distraction is None:
                if avg_ear < EAR_THRESHOLD + 0.04: distraction_timers["COGNITIVE"] = 0
                else:
                    if distraction_timers["COGNITIVE"] == 0: distraction_timers["COGNITIVE"] = time.time()
                    if time.time() - distraction_timers["COGNITIVE"] > COGNITIVE_TIME_THRESHOLD:
                        final_distraction = "COGNITIVE (Staring)"

            if final_distraction is None and gaze_direction != "CENTER":
                if distraction_timers["GAZE"] == 0: distraction_timers["GAZE"] = time.time()
                if time.time() - distraction_timers["GAZE"] > GAZE_TIME_THRESHOLD:
                    final_distraction = f"GAZE ({gaze_direction})"
            else: distraction_timers["GAZE"] = 0

            # Universal Warning and Alert Logic
            if final_distraction:
                if final_distraction != current_warning_state:
                    current_warning_state = final_distraction
                    warning_start_time = time.time()
                    alert_triggered = False
                
                elapsed_time = time.time() - warning_start_time
                if elapsed_time >= WARNING_DURATION:
                    status_text, status_color = f"ALERT: {current_warning_state}!", (0, 0, 255)
                    if not alert_triggered:
                        play_alert_sound()
                        alert_triggered = True
                else:
                    remaining_time = WARNING_DURATION - elapsed_time
                    status_text, status_color = f"Warning: {current_warning_state} ({remaining_time:.1f}s)", (0, 255, 255)
            else:
                current_warning_state, warning_start_time, alert_triggered = None, 0, False

    else:
        status_text, status_color = "NO FACE DETECTED", (0, 0, 255)

    cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.imshow("Driver Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()