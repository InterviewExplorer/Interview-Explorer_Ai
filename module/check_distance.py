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
    center_x = (left_ear.x + right_ear.x) / 2
    center_z = (left_ear.z + right_ear.z + left_eye.z + right_eye.z + mouth_left.z + mouth_right.z) / 6
    eyes_y = (left_eye.y + right_eye.y) / 2
    mouth_y = (mouth_left.y + mouth_right.y) / 2
    center_y = (eyes_y + mouth_y) / 2
    visibility = (left_ear.visibility + right_ear.visibility +
                  left_eye.visibility + right_eye.visibility +
                  mouth_left.visibility + mouth_right.visibility) / 6

    face_center = (center_x, center_y, center_z, visibility)
    
    feedback_list = []
    
    # 유클리드 거리 계산 2D
    def euclidean_distance_2d(point1, point2):
        return math.sqrt((point1.x - point2[0]) ** 2 + (point1.y - point2[1]) ** 2)

    # 손이 얼굴 범위안에 들어올 경우
    if left_wrist.visibility > 0.5 and face_center[3] > 0.9 or \
        right_wrist.visibility > 0.5 and face_center[3] > 0.9:
        if left_wrist.visibility > 0.5 and face_center[3] > 0.9 and left_wrist.z > -2:
            distance = euclidean_distance_2d(left_wrist, face_center)
            if distance < 0.25:
                feedback_list.append("왼손이 얼굴을 만지는 습관이 있을 수 있습니다.")
        elif left_wrist.visibility > 0.5 and face_center[3] > 0.9 and left_wrist.z < -2:
            feedback_list.append("왼손이 산만하게 움직이고 있습니다.")

        if right_wrist.visibility > 0.5 and face_center[3] > 0.9 and right_wrist.z > -2.3:
            distance = euclidean_distance_2d(right_wrist, face_center)
            if distance < 0.25:
                feedback_list.append("오른손이 얼굴을 만지는 습관이 있을 수 있습니다.")
        elif right_wrist.visibility > 0.5 and face_center[3] > 0.9 and right_wrist.z < -2.3:
            feedback_list.append("오른손이 산만하게 움직이고 있습니다.")

    # 상체가 정면을 바라보지 않는 경우
    if left_shoulder.visibility > 0.5 and right_shoulder.visibility:
        z_diff = left_shoulder.z - right_shoulder.z
        x_diff = left_shoulder.x - right_shoulder.x
        # shoulder_diff = abs(z_diff) + abs(x_diff) / 2

        # print(f"어깨 z 좌표 차이: {z_diff:.2f}")
        # print(f"어깨 x 좌표 차이: {x_diff:.2f}")
        # print(f"어깨 좌표 차이: {shoulder_diff:.2f}")

        if abs(z_diff) > 0.05:
            if z_diff > 0.5:
                feedback_list.append("상체가 오른쪽으로 돌아간 자세입니다.")
            elif z_diff < 0:
                feedback_list.append("상체가 왼쪽으로 돌아간 자세입니다.")
        if abs(x_diff) > 0.05:
            if x_diff > 0:
                feedback_list.append("상체가 오른쪽으로 돌아간 자세입니다.")
            else:
                feedback_list.append("상체가 왼쪽으로 돌아간 자세입니다.")

    # # 팔짱을 낀 경우
    # if pose_landmarks.landmark[19].visibility > 0.5 and pose_landmarks.landmark[14].visibility > 0.5:
    #     distance = euclidean_distance(pose_landmarks.landmark[19], pose_landmarks.landmark[14])
    #     print("왼손과 오른쪽 팔꿈치 사이의 거리: ", distance)
    #     if distance < 1:
    #         feedback_list.append("왼손이 팔짱을 낀 습관이 있습니다.")

    # if pose_landmarks.landmark[20].visibility > 0.5 and pose_landmarks.landmark[13].visibility > 0.5:
    #     distance = euclidean_distance(pose_landmarks.landmark[20], pose_landmarks.landmark[13])
    #     print("오른손과 왼쪽 팔꿈치 사이의 거리: ", distance)
    #     if distance < 1:
    #         feedback_list.append("오른손이 팔짱을 낀 습관이 있습니다.")

    # # 긴장한 자세(오른쪽)
    # if pose_landmarks.landmark[12].visibility > 0.5 and pose_landmarks.landmark[12].y < 0.7:
    #     feedback_list.append("오른쪽 어깨가 너무 올라가있어 긴장한 습관이 있습니다.")

    # # 긴장한 자세(왼쪽)
    # if pose_landmarks.landmark[11].visibility > 0.5 and pose_landmarks.landmark[11].y > 1:
    #     feedback_list.append("왼쪽 어깨가 너무 올라가있어 긴장한 습관이 있습니다.")

    # # 긴장한 자세(양쪽)
    # if pose_landmarks.landmark[12].visibility > 0.5 and pose_landmarks.landmark[11].visibility > 0.5:
    #     distance = euclidean_distance(pose_landmarks.landmark[12], pose_landmarks.landmark[11])
    #     print("오른쪽 어깨와 왼쪽 어깨 위치: ", distance)
    #     if distance < 0.5:
    #         feedback_list.append("양쪽 어깨가 너무 올라가있어 긴장한 습관이 있습니다.")

    # # 자신감이 없어 보이는 자세
    # if pose_landmarks.landmark[12].visibility > 0.5 and pose_landmarks.landmark[12].y < 0.7:
    #     feedback_list.append("오른쪽 어깨가 너무 내려가있어 자신감이 없어보이는 습관이 있습니다.")

    # if pose_landmarks.landmark[11].visibility > 0.5 and pose_landmarks.landmark[11].y < 0.7:
    #     feedback_list.append("왼쪽 어깨가 너무 내려가있어 자신감이 없어보이는 습관이 있습니다.")

    # # 자신감이 없어보이거나 피곤해 보이는 자세
    # if pose_landmarks.landmark[0].visibility > 0.5 and pose_landmarks.landmark[0].y < 0.3:
    #     feedback_list.append("고개가 너무 숙여져있어 피곤해 보이는 습관이 있습니다.")

    # # 오만하거나 무례해 보이는 자세
    # if pose_landmarks.landmark[0].visibility > 0.5 and pose_landmarks.landmark[0].y > 0.7:
    #     feedback_list.append("고개가 너무 들어있어 오만하거나 무례해 보이는 습관이 있습니다.")

    # # 상체가 앞으로 쏠린 경우
    # if (pose_landmarks.landmark[11].z > 0.3) and \
    #    (pose_landmarks.landmark[12].z > 0.3) and \
    #    (pose_landmarks.landmark[0].z > 0.3):
    #     feedback_list.append("상체가 앞으로 쏠려 피곤해 보이거나 집중하지 않는 상태입니다.")

    return feedback_list

def analyze_video_landmarks(pose_landmarks_sequence):
    feedback_set = set()
    frame_count = len(pose_landmarks_sequence)

    for frame, pose_landmarks in enumerate(pose_landmarks_sequence):
        frame_feedback = analyze_landmarks(pose_landmarks)
        feedback_set.update(frame_feedback)

    return list(feedback_set)

# 이 파일의 끝에 다음 줄 추가
__all__ = ['analyze_landmarks', 'analyze_video_landmarks']
    