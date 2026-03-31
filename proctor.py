# Phase 6 - MediaPipe Eye Gaze + Head Pose Proctoring Engine

import cv2
import time
import os
import numpy as np
from datetime import datetime

# MediaPipe imports
import mediapipe as mp

# -----------------------------
# Configuration
# -----------------------------
MAX_WARNINGS = 5
WARNING_COOLDOWN = 5
NO_FACE_THRESHOLD = 2.5
MULTI_FACE_THRESHOLD = 2
FRAME_DELAY = 0.02

# Gaze thresholds (tune these if needed)
GAZE_THRESHOLD = 0.32      # balanced gaze threshold
HEAD_TURN_THRESHOLD = 22   # tighter head turn threshold

# -----------------------------
# Paths Setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_FOLDER = os.path.join(BASE_DIR, "evidence")
LOG_FILE = os.path.join(BASE_DIR, "log.txt")

os.makedirs(EVIDENCE_FOLDER, exist_ok=True)

# -----------------------------
# Global Warning Count (for frontend polling)
# -----------------------------
_warning_count = 0

def get_warning_count():
    return _warning_count


def increment_warning(reason="Tab_Switch"):
    global _warning_count
    _warning_count += 1
    save_evidence_log(reason)


def save_evidence_log(reason):
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} - {reason}\n")
    except Exception as e:
        print("Log error:", e)


# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

# Iris landmark indices (MediaPipe Face Mesh)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eye corner indices
LEFT_EYE_CORNERS  = [33, 133]   # left corner, right corner of left eye
RIGHT_EYE_CORNERS = [362, 263]  # left corner, right corner of right eye

# Nose tip and chin for head pose
NOSE_TIP   = 4
CHIN       = 152
LEFT_EAR   = 234
RIGHT_EAR  = 454


# -----------------------------
# Save Evidence
# -----------------------------
def save_evidence(frame, reason):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(EVIDENCE_FOLDER, f"{reason}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} - {reason}\n")
    except Exception as e:
        print("Evidence save error:", e)


# -----------------------------
# Camera Initialization
# -----------------------------
def initialize_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be accessed.")
    return cap


# -----------------------------
# Get Iris Gaze Direction
# Returns: "Left", "Right", or "Center"
# -----------------------------
def get_gaze_direction(landmarks, frame_w, frame_h):
    def iris_ratio(iris_indices, corner_indices):
        iris_pts = [landmarks[i] for i in iris_indices]
        iris_x = np.mean([p.x for p in iris_pts]) * frame_w

        left_corner_x  = landmarks[corner_indices[0]].x * frame_w
        right_corner_x = landmarks[corner_indices[1]].x * frame_w

        eye_width = right_corner_x - left_corner_x
        if eye_width == 0:
            return 0.5

        ratio = (iris_x - left_corner_x) / eye_width
        return ratio

    left_ratio  = iris_ratio(LEFT_IRIS,  LEFT_EYE_CORNERS)
    right_ratio = iris_ratio(RIGHT_IRIS, RIGHT_EYE_CORNERS)

    avg_ratio = (left_ratio + right_ratio) / 2.0

    if avg_ratio < GAZE_THRESHOLD:
        return "Looking Left"
    elif avg_ratio > (1 - GAZE_THRESHOLD):
        return "Looking Right"
    else:
        return "Center"


# -----------------------------
# Get Head Pose Direction
# Returns: "Left", "Right", or "Center"
# -----------------------------
def get_head_pose(landmarks, frame_w):
    nose_x      = landmarks[NOSE_TIP].x * frame_w
    left_ear_x  = landmarks[LEFT_EAR].x * frame_w
    right_ear_x = landmarks[RIGHT_EAR].x * frame_w

    face_width = right_ear_x - left_ear_x
    if face_width == 0:
        return "Center"

    # Nose offset from face center
    face_center = (left_ear_x + right_ear_x) / 2.0
    offset = (nose_x - face_center) / face_width * 100  # in % of face width

    if offset < -HEAD_TURN_THRESHOLD:
        return "Head Left"
    elif offset > HEAD_TURN_THRESHOLD:
        return "Head Right"
    else:
        return "Center"


# -----------------------------
# Flask Streaming Generator
# -----------------------------
def generate_frames():

    cap = initialize_camera()

    global _warning_count
    _warning_count     = 0
    warning_count      = 0
    last_warning_time  = 0
    no_face_start      = None
    gaze_warning_start = None

    with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,       # enables iris landmarks
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Frame capture failed.")
                    break

                frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = face_mesh.process(rgb_frame)

                current_time = time.time()
                alert_text   = ""

                # -------------------------
                # No Face Detection
                # -------------------------
                if not results.multi_face_landmarks:
                    if no_face_start is None:
                        no_face_start = current_time
                    elif current_time - no_face_start > NO_FACE_THRESHOLD:
                        if current_time - last_warning_time > WARNING_COOLDOWN:
                            if warning_count < MAX_WARNINGS:
                                warning_count += 1
                                save_evidence(frame, "No_Face")
                            last_warning_time = current_time
                        no_face_start = None

                    cv2.putText(frame, "No Face Detected!",
                                (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2)

                else:
                    no_face_start = None

                    # -------------------------
                    # Multiple Faces
                    # -------------------------
                    if len(results.multi_face_landmarks) > 1:
                        cv2.putText(frame, "Multiple Faces Detected!",
                                    (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (0, 0, 255), 2)

                        if current_time - last_warning_time > WARNING_COOLDOWN:
                            if warning_count < MAX_WARNINGS:
                                warning_count += 1
                                save_evidence(frame, "Multiple_Faces")
                            last_warning_time = current_time

                    # -------------------------
                    # Eye Gaze + Head Pose
                    # (use first/main face only)
                    # -------------------------
                    face_landmarks = results.multi_face_landmarks[0].landmark

                    # -------------------------
                    # Draw Green Face Box
                    # -------------------------
                    x_coords = [lm.x * frame_w for lm in face_landmarks]
                    y_coords = [lm.y * frame_h for lm in face_landmarks]
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    gaze      = get_gaze_direction(face_landmarks, frame_w, frame_h)
                    head_pose = get_head_pose(face_landmarks, frame_w)

                    # Only suspicious if BOTH eyes AND head turn same direction
                    # Eyes alone = normal reading. Head alone extreme = suspicious.
                    gaze_dir = gaze.replace("Looking ", "")
                    head_dir = head_pose.replace("Head ", "")

                    both_same = (
                        gaze_dir != "Center" and
                        head_dir != "Center" and
                        gaze_dir == head_dir
                    )
                    head_extreme = (head_dir != "Center" and gaze_dir == "Center")
                    # flag: extreme gaze, head turn, or both together
                    gaze_extreme = (gaze_dir != "Center")
                    suspicious   = both_same or head_extreme or gaze_extreme

                    if suspicious:
                        label = f"{gaze} + {head_pose}" if both_same else (gaze if gaze_extreme else head_pose)
                        cv2.putText(frame, "SUSPICIOUS!",
                                    (40, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.85, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Monitoring...",
                                    (40, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)

                    if suspicious:

                        if gaze_warning_start is None:
                            gaze_warning_start = current_time
                        elif current_time - gaze_warning_start > 2.0:
                            if current_time - last_warning_time > WARNING_COOLDOWN:
                                if warning_count < MAX_WARNINGS:
                                    warning_count += 1
                                    _warning_count = warning_count
                                    reason = label.replace(" ", "_").replace("+", "and")
                                    save_evidence(frame, reason)
                                last_warning_time = current_time
                            gaze_warning_start = None
                    else:
                        gaze_warning_start = None

                # Sync global warning count every frame
                _warning_count = warning_count

                # -------------------------
                # Warnings Display
                # -------------------------
                cv2.putText(frame,
                            f"Warnings: {warning_count}/{MAX_WARNINGS}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

                # -------------------------
                # Exam End Conditions
                # -------------------------
                if warning_count >= MAX_WARNINGS:
                    cv2.putText(frame, "Exam Terminated!",
                                (frame_w//2 - 130, frame_h//2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')
                    break

                # -------------------------
                # Encode and Stream
                # -------------------------
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')

                time.sleep(FRAME_DELAY)

        except Exception as e:
            print("Streaming error:", e)

        finally:
            cap.release()
            cv2.destroyAllWindows()