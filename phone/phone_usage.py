import cv2
import mediapipe as mp
import time
import numpy as np
import torch
import winsound
import threading
from enum import Enum

# =================================================================================
# --- Parameters & Optimizations ---
# =================================================================================

# --- Hand Tracking Parameters ---
CALIBRATION_DURATION = 10.0
MIN_CALIBRATION_SAMPLES = 50
ROI_FIXED_PIXEL_PADDING = 50
NARROW_THRESHOLD = 0.15
WIDE_THRESHOLD = 0.3

# --- Phone Detection Parameters ---
YOLO_CONF_THRESHOLD = 0.25
PHONE_PERSISTENCE_FRAMES = 10

# --- Alerting Timers and Parameters ---
ONE_HAND_OFF_THRESHOLD = 15.0 
BOTH_HANDS_OFF_THRESHOLD = 5.0 
PHONE_AND_ONE_HAND_OFF_THRESHOLD = 4.0 
BEEP_COOLDOWN = 3.0

# --- PERFORMANCE OPTIMIZATIONS ---
YOLO_FRAME_INTERVAL = 5 
HOLISTIC_MODEL_COMPLEXITY = 0
YOLO_MODEL_SIZE = 'yolov5s'

EXTRA_TOP_MARGIN = 100 # Extra margin for the top of the ROI
EXTRA_LEFT_MARGIN = -20  # Decrease left margin by 20 pixels
EXTRA_RIGHT_MARGIN = 30  # Increase right margin by 30 pixels

# =================================================================================
# --- System Setup ---
# =================================================================================

class AlertLevel(Enum):
    SAFE = 0
    CAUTION = 1
    WARNING = 2
    SEVERE = 3

def calculate_3d_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def is_hand_near_box(hand_landmarks, box, image_width, image_height):
    if not hand_landmarks:
        return False
    x1, y1, x2, y2 = box
    for lm in hand_landmarks.landmark:
        hand_x, hand_y = int(lm.x * image_width), int(lm.y * image_height)
        if x1 < hand_x < x2 and y1 < hand_y < y2:
            return True
    return False

def play_alert_sound(last_alert_time, freq=1000, duration=700):
    current_time = time.time()
    if (current_time - last_alert_time) > BEEP_COOLDOWN:
        threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()
        return current_time
    return last_alert_time

def run_integrated_system():
    # --- Model and Camera Initialization ---
    print("Loading models...")
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=HOLISTIC_MODEL_COMPLEXITY)
    
    yolo_model = torch.hub.load('ultralytics/yolov5', YOLO_MODEL_SIZE, pretrained=True)
    yolo_model.conf = YOLO_CONF_THRESHOLD
    yolo_model.classes = [67]
    print("Models loaded successfully.")

    cap = cv2.VideoCapture(0)

    # --- State and Timer Variables ---
    is_calibrating = True
    calibration_failed = False
    calibrated_roi = None
    calibrated_hand_distance = None 
    calibration_samples = []
    last_time = time.time()
    app_start_time = time.time()
    one_hand_off_timer = 0.0
    both_hands_off_timer = 0.0
    phone_persistence_counter = 0
    last_alert_sound_time = 0
    
    frame_counter = 0
    phone_is_held = False 
    phone_is_visible_stable = False

    while cap.isOpened():
        # --- FPS Counter Start ---
        fps_start_time = time.time()
        
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        ### NEW OPTIMIZATION: Resize frame to reduce workload for all models ###
        image = cv2.resize(image, (640, 480))
        
        h, w, c = image.shape
        
        # --- Run Models ---
        results_holistic = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        frame_counter += 1
        if frame_counter % YOLO_FRAME_INTERVAL == 0:
            results_yolo = yolo_model(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            detections = results_yolo.xyxy[0].cpu().numpy()
            
            phone_found_this_frame = False
            for *box, conf, cls in detections:
                if int(cls) == 67:
                    phone_box = list(map(int, box))
                    left_hand_near = is_hand_near_box(results_holistic.left_hand_landmarks, phone_box, w, h)
                    right_hand_near = is_hand_near_box(results_holistic.right_hand_landmarks, phone_box, w, h)
                    
                    if left_hand_near or right_hand_near:
                        phone_found_this_frame = True
                        phone_is_held = True
                        cv2.rectangle(image, (phone_box[0], phone_box[1]), (phone_box[2], phone_box[3]), (0, 0, 255), 2)
                        cv2.putText(image, f"Phone Held {conf:.2f}", (phone_box[0], phone_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        break
            
            if not phone_found_this_frame:
                 phone_is_held = False

            if phone_is_held:
                phone_persistence_counter = PHONE_PERSISTENCE_FRAMES
            elif phone_persistence_counter > 0:
                phone_persistence_counter -= 1
            
            phone_is_visible_stable = phone_persistence_counter > 0
        
        delta_time = time.time() - last_time
        last_time = time.time()

        if is_calibrating:
            # --- PHASE 1: CALIBRATION ---
            elapsed_time = time.time() - app_start_time
            progress = elapsed_time / CALIBRATION_DURATION
            cv2.putText(image, "CALIBRATING: Hold wheel with both hands", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.rectangle(image, (50, 80), (int(50 + 300 * progress), 100), (0, 255, 255), -1)
            cv2.rectangle(image, (50, 80), (350, 100), (255, 255, 255), 2)
    
            if results_holistic.left_hand_landmarks and results_holistic.right_hand_landmarks:
                left_wrist = results_holistic.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
                right_wrist = results_holistic.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
                
                calibration_samples.append({
                    "left_pos": (int(left_wrist.x * w), int(left_wrist.y * h)),
                    "right_pos": (int(right_wrist.x * w), int(right_wrist.y * h)),
                    "distance": calculate_3d_distance(left_wrist, right_wrist)
                })

            if elapsed_time >= CALIBRATION_DURATION:
                is_calibrating = False
                if len(calibration_samples) > MIN_CALIBRATION_SAMPLES:
                    all_points = [s['left_pos'] for s in calibration_samples] + [s['right_pos'] for s in calibration_samples]
                    points_arr = np.array(all_points)
                    x_min, y_min = np.min(points_arr, axis=0)
                    x_max, y_max = np.max(points_arr, axis=0)
                    calibrated_roi = (x_min - ROI_FIXED_PIXEL_PADDING, y_min - ROI_FIXED_PIXEL_PADDING - EXTRA_TOP_MARGIN, 
                                      x_max + ROI_FIXED_PIXEL_PADDING, y_max + ROI_FIXED_PIXEL_PADDING)
                    all_distances = [s['distance'] for s in calibration_samples]
                    calibrated_hand_distance = np.mean(all_distances)
                    print(f"Calibration successful. Calibrated Distance: {calibrated_hand_distance:.3f}")
                else:
                    calibration_failed = True
        else:
            # --- PHASE 2: INTEGRATED MONITORING ---
            if calibration_failed:
                cv2.putText(image, "Calibration Failed. Please Restart.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue
            
            display_text, display_color = "STATUS: SAFE", (0, 255, 0)
            hands_on_wheel_count = 0
            
            if calibrated_roi and calibrated_hand_distance:
                cv2.rectangle(image, (calibrated_roi[0], calibrated_roi[1]), (calibrated_roi[2], calibrated_roi[3]), (0, 255, 0), 1)
                left_in_roi = False
                if results_holistic.left_hand_landmarks:
                    wrist = results_holistic.left_hand_landmarks.landmark[0]
                    if calibrated_roi[0] < int(wrist.x*w) < calibrated_roi[2] and calibrated_roi[1] < int(wrist.y*h) < calibrated_roi[3]:
                        left_in_roi = True
                right_in_roi = False
                if results_holistic.right_hand_landmarks:
                    wrist = results_holistic.right_hand_landmarks.landmark[0]
                    if calibrated_roi[0] < int(wrist.x*w) < calibrated_roi[2] and calibrated_roi[1] < int(wrist.y*h) < calibrated_roi[3]:
                        right_in_roi = True
                hands_in_roi_count = left_in_roi + right_in_roi
                if hands_in_roi_count == 2:
                    left_wrist = results_holistic.left_hand_landmarks.landmark[0]
                    right_wrist = results_holistic.right_hand_landmarks.landmark[0]
                    current_distance = calculate_3d_distance(left_wrist, right_wrist)
                    lower_bound = calibrated_hand_distance - NARROW_THRESHOLD
                    upper_bound = calibrated_hand_distance + WIDE_THRESHOLD
                    if lower_bound < current_distance < upper_bound:
                        hands_on_wheel_count = 2
                    else:
                        hands_on_wheel_count = 1
                else:
                    hands_on_wheel_count = hands_in_roi_count

            if hands_on_wheel_count == 0:
                both_hands_off_timer += delta_time
                one_hand_off_timer = 0
                if both_hands_off_timer > BOTH_HANDS_OFF_THRESHOLD:
                    display_text, display_color = "ALERT! BOTH HANDS OFF WHEEL!", (0, 0, 255)
                    last_alert_sound_time = play_alert_sound(last_alert_sound_time, 1200, 800)
                else:
                    display_text, display_color = "WARNING: BOTH HANDS OFF", (0, 165, 255)
            elif hands_on_wheel_count == 1 and phone_is_visible_stable:
                both_hands_off_timer += delta_time
                one_hand_off_timer = 0
                if both_hands_off_timer > PHONE_AND_ONE_HAND_OFF_THRESHOLD:
                    display_text, display_color = "SEVERE: PHONE USE DETECTED!", (0, 0, 255)
                    last_alert_sound_time = play_alert_sound(last_alert_sound_time, 1500, 800)
                else:
                    display_text, display_color = "WARNING: Phone and One Hand Off", (0, 165, 255)
            elif hands_on_wheel_count == 1:
                one_hand_off_timer += delta_time
                both_hands_off_timer = 0
                if one_hand_off_timer > ONE_HAND_OFF_THRESHOLD:
                    display_text, display_color = "ALERT: ONE HAND OFF TOO LONG!", (0, 255, 255)
                    last_alert_sound_time = play_alert_sound(last_alert_sound_time, 800, 500)
                else:
                    display_text, display_color = f"CAUTION: ONE HAND OFF ({int(ONE_HAND_OFF_THRESHOLD - one_hand_off_timer)}s)", (0, 255, 255)
            else:
                one_hand_off_timer, both_hands_off_timer = 0.0, 0.0
                if phone_is_visible_stable:
                    display_text, display_color = "CAUTION: Phone Detected", (0, 255, 255)
                    last_alert_sound_time = play_alert_sound(last_alert_sound_time, 800, 400)
                else:
                    display_text, display_color = "STATUS: SAFE", (0, 255, 0)
            
            cv2.putText(image, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2)
        
        # --- Drawing landmarks ---
        if results_holistic.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results_holistic.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results_holistic.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results_holistic.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # --- FPS Counter Display ---
        fps_end_time = time.time()
        fps = 1 / (fps_end_time - fps_start_time)
        cv2.putText(image, f"FPS: {int(fps)}", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Integrated Driver Monitoring System', image)
        if cv2.waitKey(5) & 0xFF == 27: break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

if __name__ == '__main__':
    run_integrated_system()