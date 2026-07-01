import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import csv
import os
from bmi_model import predict_bmi
from bmi_categories import classify_bmi, category_color_bgr
from body_features import estimate_reliability, landmarks_to_features

DEBUG_FEATURES = os.environ.get("BMI_DEBUG_FEATURES") == "1"
MIN_RELIABILITY = 0.75


def camera_pose_ready(landmarks):
    top_y = landmarks[0].y
    foot_y = max(landmarks[27].y, landmarks[28].y)
    body_span = foot_y - top_y

    feet_visible = min(landmarks[27].visibility, landmarks[28].visibility) >= 0.65
    shoulders_visible = min(landmarks[11].visibility, landmarks[12].visibility) >= 0.65
    hips_visible = min(landmarks[23].visibility, landmarks[24].visibility) >= 0.65

    return (
        feet_visible and
        shoulders_visible and
        hips_visible and
        0.55 <= body_span <= 0.98 and
        top_y >= -0.02 and
        foot_y <= 1.02
    )


def measurement_ready(features):
    shoulder, hip, height, torso, arm, leg, ratio, shoulder_h, hip_h = features

    # Live video often under-measures hips when the lower body is cropped,
    # dark, or segmented poorly. In that case, skip instead of showing underweight.
    if hip_h < 0.24 and ratio > 1.35:
        return False

    if shoulder_h < 0.18 or hip_h < 0.18:
        return False

    return True


def draw_message(frame, message, y=60):
    cv2.putText(frame, message, (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# ===== CREATE HISTORY FILE =====
if not os.path.exists("history.csv"):
    with open("history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["BMI", "Category"])

# ===== MEDIAPIPE =====
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

# ===== VARIABLES =====
bmi_history = deque(maxlen=10)
frame_count = 0
previous_bmi = None

# ===== MAIN LOOP =====
with mp_pose.Pose(enable_segmentation=True) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            if not camera_pose_ready(lm):
                bmi_history.clear()
                draw_message(frame, "Step back: show full body", 60)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                cv2.imshow("Live BMI Detection", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            features = landmarks_to_features(
                lm,
                results.segmentation_mask,
                debug=DEBUG_FEATURES and frame_count % 15 == 0
            )

            if features is None:
                bmi_history.clear()
                draw_message(frame, "Stand fully in frame", 60)
            elif not measurement_ready(features):
                bmi_history.clear()
                if DEBUG_FEATURES:
                    print("Skipped camera features:", features)
                draw_message(frame, "Adjust camera: body width unclear", 60)
            else:
                # ===== STABILIZATION =====
                if frame_count < 10:
                    cv2.putText(frame, "Stabilizing...", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                else:
                    # ===== PREDICTION =====
                    bmi = predict_bmi(features)

                    if bmi is None:
                        cv2.putText(frame, "Prediction Error", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    else:
                        reliability = estimate_reliability(features)
                        if reliability is None or reliability < MIN_RELIABILITY:
                            bmi_history.clear()
                            draw_message(frame, "Adjust lighting / pose", 60)
                            bmi = None

                    if bmi is not None:
                        # ===== SMOOTHING =====
                        bmi_history.append(bmi)
                        smooth_bmi = round(float(np.median(bmi_history)), 2)

                        # ===== CATEGORY =====
                        category = classify_bmi(smooth_bmi)
                        color = category_color_bgr(category)

                        # ===== SAVE HISTORY =====
                        if frame_count % 20 == 0:
                            with open("history.csv", "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([smooth_bmi, category])

                        # ===== CHANGE =====
                        if previous_bmi is not None:
                            diff = round(smooth_bmi - previous_bmi, 2)
                            cv2.putText(frame, f"Change: {diff}", (30, 260),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                        previous_bmi = smooth_bmi

                        # ===== UI =====
                        cv2.rectangle(frame, (20,20), (380,280), (255,255,255), 2)

                        cv2.putText(frame, f"BMI: {smooth_bmi}", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                        cv2.putText(frame, category, (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        cv2.putText(frame, f"Reliability: {reliability}", (30, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

                        cv2.putText(frame, "Normal: 18.5 - 25.4", (30, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

                        cv2.putText(frame, "AI Estimation", (30, 220),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # ===== DRAW SKELETON =====
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

        else:
            cv2.putText(frame, "No person detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Live BMI Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# ===== CLEANUP =====
cap.release()
cv2.destroyAllWindows()
