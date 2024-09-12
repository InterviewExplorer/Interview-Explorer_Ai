import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def draw_human_silhouette(frame, left_offset=200, right_offset=200, vertical_offset=100, head_vertical_offset=-75):
    h, w, _ = frame.shape
    head_center = (w//2, int(h*0.25) + vertical_offset + head_vertical_offset)
    head_radius = int(h*0.1 * 2)
    cv2.circle(frame, head_center, head_radius, (0, 255, 0), 2)
    
    top_left = (int(w*0.4) - left_offset, int(h*0.35) + vertical_offset)
    top_right = (int(w*0.6) + right_offset, int(h*0.35) + vertical_offset)
    height = int(h*0.09 * 4) * 2
    
    expanded_top_left = (top_left[0], top_left[1])
    expanded_top_right = (top_right[0], top_right[1])
    
    cv2.rectangle(frame, expanded_top_left, (expanded_top_right[0], expanded_top_left[1] + height), (0, 255, 0), 2)
    
    return (expanded_top_left, expanded_top_right, height, head_center, head_radius)

def is_within_area(point, top_left, top_right, height, head_center, head_radius):
    x, y = point
    dist_to_head = np.sqrt((x - head_center[0])**2 + (y - head_center[1])**2)
    if dist_to_head <= head_radius:
        return True
    if top_left[0] < x < top_right[0] and top_left[1] < y < top_left[1] + height:
        return True
    return False

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    success_flag = False
    top_left, top_right, height, head_center, head_radius = draw_human_silhouette(frame)

    if results.pose_landmarks:
        h, w, _ = frame.shape

        def to_pixel_coordinates(landmark):
            return int(landmark.x * w), int(landmark.y * h)

        nose = to_pixel_coordinates(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
        left_shoulder = to_pixel_coordinates(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = to_pixel_coordinates(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER])

        if (is_within_area(nose, top_left, top_right, height, head_center, head_radius) and
            is_within_area(left_shoulder, top_left, top_right, height, head_center, head_radius) and
            is_within_area(right_shoulder, top_left, top_right, height, head_center, head_radius)):
            success_flag = True

    return frame, success_flag