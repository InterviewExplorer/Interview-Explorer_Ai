import math

def analyze_landmarks(pose_landmarks):

    # 얼굴
    nose = pose_landmarks.landmark[0]
    left_eye = pose_landmarks.landmark[2]
    right_eye = pose_landmarks.landmark[5]
    left_ear = pose_landmarks.landmark[7]
    right_ear = pose_landmarks.landmark[8]
    mouth_left = pose_landmarks.landmark[9]
    mouth_right = pose_landmarks.landmark[10]

    # 왼쪽 상체
    left_shoulder = pose_landmarks.landmark[11]
    left_elbow = pose_landmarks.landmark[13]
    left_wrist = pose_landmarks.landmark[15]
    left_pinky = pose_landmarks.landmark[17]
    left_index = pose_landmarks.landmark[19]
    left_thumb = pose_landmarks.landmark[21]

    # 오른쪽 상체
    right_shoulder = pose_landmarks.landmark[12]
    right_elbow = pose_landmarks.landmark[14]
    right_wrist = pose_landmarks.landmark[16]
    right_pinky = pose_landmarks.landmark[18]
    right_index = pose_landmarks.landmark[20]
    right_thumb = pose_landmarks.landmark[22]

    # 얼굴 중심 좌표
    center_x = (left_ear.x + right_ear.x + nose.x) / 3
    center_z = (left_ear.z + right_ear.z + left_eye.z + right_eye.z + mouth_left.z + mouth_right.z + nose.z) / 7
    eyes_y = (left_eye.y + right_eye.y) / 2
    mouth_y = (mouth_left.y + mouth_right.y) / 2
    center_y = (eyes_y + mouth_y + nose.y) / 3
    visibility = (left_ear.visibility + right_ear.visibility +
                  left_eye.visibility + right_eye.visibility +
                  mouth_left.visibility + mouth_right.visibility + 
                  nose.visibility) / 7

    face_center = (center_x, center_y, center_z, visibility)
    
    feedback_list = []

    # touch_face = 0
    # hand_move = 0
    # not_front = 0


    # 유클리드 거리 계산 2D
    def euclidean_distance_2d(point1, point2):
        return math.sqrt((point1.x - point2[0]) ** 2 + (point1.y - point2[1]) ** 2)

    # 손이 얼굴 범위안에 들어올 경우
    if left_wrist.visibility > 0.5 and face_center[3] > 0.9 or \
        right_wrist.visibility > 0.5 and face_center[3] > 0.9:
        if left_wrist.visibility > 0.5 and face_center[3] > 0.9 and left_wrist.z > -2:
            distance = euclidean_distance_2d(left_wrist, face_center)
            if distance < 0.25:
                feedback_list.append("얼굴 만짐")
                # touch_face = 1
            else:
                feedback_list.append("산만한 손의 움직임")
                # hand_move = 1

        if right_wrist.visibility > 0.5 and face_center[3] > 0.9 and right_wrist.z > -2.3:
            distance = euclidean_distance_2d(right_wrist, face_center)
            if distance < 0.25:
                feedback_list.append("얼굴 만짐")
                # touch_face = 1
            else:
                feedback_list.append("산만한 손의 움직임")
                # hand_move = 1

    # 상체가 정면을 바라보지 않는 경우
    if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
        shoulder_z_diff = left_shoulder.z - right_shoulder.z
        if abs(shoulder_z_diff) > 0.3:
            if shoulder_z_diff > 0:
                feedback_list.append("정면을 보지않는 자세")
                # not_front = 1
            elif shoulder_z_diff < 0:
                feedback_list.append("정면을 보지않는 자세")
                # not_front = 1

    # return feedback_list, touch_face, hand_move, not_front
    return feedback_list

def analyze_video_landmarks(pose_landmarks_sequence):
    feedback_set = set()
    frame_count = len(pose_landmarks_sequence)
    # face_touch_total = 0
    # hand_move_total = 0
    # not_front_total = 0

    for frame, pose_landmarks in enumerate(pose_landmarks_sequence):
        # frame_feedback, touch_face, hand_move, not_front = analyze_landmarks(pose_landmarks)
        frame_feedback = analyze_landmarks(pose_landmarks)
        feedback_set.update(frame_feedback)
        # face_touch_total = max(face_touch_total, touch_face)
        # hand_move_total = max(hand_move_total, hand_move)
        # not_front_total = max(not_front_total, not_front)
        
    # return list(feedback_set), face_touch_total, hand_move_total, not_front_total
    return list(feedback_set)

# 이 파일의 끝에 다음 줄 추가
__all__ = ['analyze_landmarks', 'analyze_video_landmarks']
    