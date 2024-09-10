import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 카메라 시작
cap = cv2.VideoCapture(0)

def draw_human_silhouette(frame, left_offset=200, right_offset=200, vertical_offset=180, head_vertical_offset=-75):
    h, w, _ = frame.shape
    
    # 얼굴 (머리) 중심과 반지름
    head_center = (w//2, int(h*0.25) + vertical_offset + head_vertical_offset)
    head_radius = int(h*0.1 * 2)  # 얼굴 크기 1.7배 키우기
    cv2.circle(frame, head_center, head_radius, (0, 255, 0), 2)
    
    # 사각형의 기준 좌표
    top_left = (int(w*0.4) - left_offset, int(h*0.35) + vertical_offset)
    top_right = (int(w*0.6) + right_offset, int(h*0.35) + vertical_offset)
    
    # 사각형의 세로 길이 설정
    height = int(h*0.09 * 4) * 2

    # 가로 길이 조정
    expanded_top_left = (top_left[0], top_left[1])
    expanded_top_right = (top_right[0], top_right[1])
    
    # 사각형 그리기
    cv2.rectangle(frame, expanded_top_left, (expanded_top_right[0], expanded_top_left[1] + height), (0, 255, 0), 2)  # 확대된 사각형

    return (expanded_top_left, expanded_top_right, height, head_center, head_radius)

def is_within_area(point, top_left, top_right, height, head_center, head_radius):
    x, y = point
    
    # 얼굴 (머리) 영역 확인
    dist_to_head = np.sqrt((x - head_center[0])**2 + (y - head_center[1])**2)
    if dist_to_head <= head_radius:
        return True
    
    # 사각형 영역 확인
    if top_left[0] < x < top_right[0] and top_left[1] < y < top_left[1] + height:
        return True

    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR을 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    # 랜드마크가 감지되었는지 확인
    if result.pose_landmarks:
        # 랜드마크 추출
        landmarks = result.pose_landmarks.landmark
        h, w, _ = frame.shape

        # 랜드마크 좌표를 픽셀 단위로 변환
        def to_pixel_coordinates(landmark):
            return int(landmark.x * w), int(landmark.y * h)

        # 랜드마크 좌표 추출
        left_shoulder = to_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = to_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
        nose = to_pixel_coordinates(landmarks[mp_pose.PoseLandmark.NOSE])

        # 얼굴과 상반신을 포함하는 영역 (사람 모형으로 설정)
        top_left, top_right, height, head_center, head_radius = draw_human_silhouette(frame)

        # 얼굴 및 어깨가 영역 안에 들어왔는지 확인
        if (is_within_area(nose, top_left, top_right, height, head_center, head_radius) and
            is_within_area(left_shoulder, top_left, top_right, height, head_center, head_radius) and
            is_within_area(right_shoulder, top_left, top_right, height, head_center, head_radius)):
            success_flag = True
        else:
            success_flag = False
            
        # 성공 메시지 표시
        if success_flag:
            cv2.putText(frame, "Success", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Align your face and shoulder", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
