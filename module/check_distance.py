import math

def analyze_landmarks(pose_landmarks):
    
    print("포즈 랜드마크 실행 중...")
    feedback_list = [""]

    # 유클리드 거리 계산
    def euclidean_distance(point1, point2):
        print("유클리드 거리 계산 중...")
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)
    
    # 손을 턱에 기댄 경우
    if pose_landmarks.landmark[19].visibility > 0.5 and pose_landmarks.landmark[9].visibility > 0.5:
        distance = euclidean_distance(pose_landmarks.landmark[19], pose_landmarks.landmark[9])
        print("왼손과 턱 사이의 거리: ", distance)
        if distance < 0.7:
            print("왼손이 턱에 기댄 습관이 있습니다.")
            feedback_list.append("왼손이 턱에 기댄 습관이 있습니다.")
        else:
            print("왼손이 턱에 기댄 습관이 없습니다.")

    if pose_landmarks.landmark[20].visibility > 0.5 and pose_landmarks.landmark[10].visibility > 0.5:
        distance = euclidean_distance(pose_landmarks.landmark[20], pose_landmarks.landmark[10])
        print("오른손과 턱 사이의 거리: ", distance)
        if distance < 1.5:
            print("오른손이 턱에 기댄 습관이 있습니다.")
            feedback_list.append("오른손이 턱에 기댄 습관이 있습니다.")
        else:
            print("오른손이 턱에 기댄 습관이 없습니다.")

    # 손으로 코를 만지는 경우
    if pose_landmarks.landmark[19].visibility > 0.5 and pose_landmarks.landmark[0].visibility > 0.5:
        distance = euclidean_distance(pose_landmarks.landmark[19], pose_landmarks.landmark[0])
        print("왼손과 코 사이의 거리: ", distance)
        if distance < 0.5:
            print("왼손으로 코를 만지는 습관이 있습니다.")
            feedback_list.append("왼손으로 코를 만지는 습관이 있습니다.")
        else:
            print("왼손으로 코를 만지는 습관이 없습니다.")

    if pose_landmarks.landmark[20].visibility > 0.5 and pose_landmarks.landmark[0].visibility > 0.5:
        distance = euclidean_distance(pose_landmarks.landmark[20], pose_landmarks.landmark[0])
        print("오른손과 코 사이의 거리: ", distance)
        if distance < 1.5:
            print("오른손으로 코를 만지는 습관이 있습니다.")
            feedback_list.append("오른손으로 코를 만지는 습관이 있습니다.")
        else:
            print("오른손으로 코를 만지는 습관이 없습니다.")

    # 손으로 눈을 만지는 경우
    if pose_landmarks.landmark[19].visibility > 0.5 and pose_landmarks.landmark[1].visibility > 0.5:
        distance = euclidean_distance(pose_landmarks.landmark[19], pose_landmarks.landmark[1])
        print("왼손과 왼쪽 눈 사이의 거리: ", distance)
        if distance < 0.5:
            print("왼손으로 눈을 만지는 습관이 있습니다.")
            feedback_list.append("왼손으로 눈을 만지는 습관이 있습니다.")
        else:
            print("왼손으로 눈을 만지는 습관이 없습니다.")

    if pose_landmarks.landmark[20].visibility > 0.5 and pose_landmarks.landmark[1].visibility > 0.5:
        distance = euclidean_distance(pose_landmarks.landmark[20], pose_landmarks.landmark[1])
        print("오른손과 오른쪽 눈 사이의 거리: ", distance)
        if distance < 1.5:
            print("오른손으로 눈을 만지는 습관이 있습니다.")
            feedback_list.append("오른손으로 눈을 만지는 습관이 있습니다.")
        else:
            print("오른손으로 눈을 만지는 습관이 없습니다.")

    # 손으로 입을 만지는 경우
    if pose_landmarks.landmark[19].visibility > 0.5 and (pose_landmarks.landmark[9].visibility > 0.5 or pose_landmarks.landmark[10].visibility > 0.5):
        distance = euclidean_distance(pose_landmarks.landmark[19], pose_landmarks.landmark[9])
        print("왼손과 입 사이의 거리: ", distance)
        if distance < 0.7:
            print("왼손으로 입을 만지는 습관이 있습니다.")
            feedback_list.append("왼손으로 입을 만지는 습관이 있습니다.")
        else:
            print("왼손으로 입을 만지는 습관이 없습니다.")

    if pose_landmarks.landmark[20].visibility > 0.5 and (pose_landmarks.landmark[9].visibility > 0.5 or pose_landmarks.landmark[10].visibility > 0.5):
        distance = euclidean_distance(pose_landmarks.landmark[20], pose_landmarks.landmark[9])
        print("오른손과 입 사이의 거리: ", distance)
        if distance < 1.5:
            print("오른손으로 입을 만지는 습관이 있습니다.")
            feedback_list.append("오른손으로 입을 만지는 습관이 있습니다.")
        else:
            print("오른손으로 입을 만지는 습관이 없습니다.")

    # 팔짱을 낀 경우
    if pose_landmarks.landmark[19].visibility > 0.5 and pose_landmarks.landmark[14].visibility > 0.5:
        distance = euclidean_distance(pose_landmarks.landmark[19], pose_landmarks.landmark[14])
        print("왼손과 오른쪽 팔꿈치 사이의 거리: ", distance)
        if distance < 1:
            print("왼손이 팔짱을 낀 습관이 있습니다.")
            feedback_list.append("왼손이 팔짱을 낀 습관이 있습니다.")
        else:
            print("왼손이 팔짱을 낀 습관이 없습니다.")

    if pose_landmarks.landmark[20].visibility > 0.5 and pose_landmarks.landmark[13].visibility > 0.5:
        distance = euclidean_distance(pose_landmarks.landmark[20], pose_landmarks.landmark[13])
        print("오른손과 왼쪽 팔꿈치 사이의 거리: ", distance)
        if distance < 1:
            print("오른손이 팔짱을 낀 습관이 있습니다.")
            feedback_list.append("오른손이 팔짱을 낀 습관이 있습니다.")
        else:
            print("오른손이 팔짱을 낀 습관이 없습니다.")

    # 긴장한 자세(오른쪽)
    if pose_landmarks.landmark[12].visibility > 0.5 and pose_landmarks.landmark[12].y < 0.7:
        print("오른쪽 어깨가 너무 올라가있어 긴장한 습관이 있습니다.")
        feedback_list.append("오른쪽 어깨가 너무 올라가있어 긴장한 습관이 있습니다.")

    # 긴장한 자세(왼쪽)
    if pose_landmarks.landmark[11].visibility > 0.5 and pose_landmarks.landmark[11].y > 1:
        print("왼쪽 어깨가 너무 올라가있어 긴장한 습관이 있습니다.")
        feedback_list.append("왼쪽 어깨가 너무 올라가있어 긴장한 습관이 있습니다.")

    # 긴장한 자세(양쪽)
    if pose_landmarks.landmark[12].visibility > 0.5 and pose_landmarks.landmark[11].visibility > 0.5:
        distance = euclidean_distance(pose_landmarks.landmark[12], pose_landmarks.landmark[11])
        print("오른쪽 어깨와 왼쪽 어깨 위치: ", distance)
        if distance < 0.5:
            print("양쪽 어깨가 너무 올라가있어 긴장한 습관이 있습니다.")
            feedback_list.append("양쪽 어깨가 너무 올라가있어 긴장한 습관이 있습니다.")
        else:
            print("양쪽 어깨가 너무 올라가있어 긴장한 습관이 없습니다.")

    # 자신감이 없어 보이는 자세
    if pose_landmarks.landmark[12].visibility > 0.5 and pose_landmarks.landmark[12].y < 0.7:
        print("오른쪽 어깨가 너무 내려가있어 자신감이 없어보이는 습관이 있습니다.")
        feedback_list.append("오른쪽 어깨가 너무 내려가있어 자신감이 없어보이는 습관이 있습니다.")

    if pose_landmarks.landmark[11].visibility > 0.5 and pose_landmarks.landmark[11].y < 0.7:
        print("왼쪽 어깨가 너무 내려가있어 자신감이 없어보이는 습관이 있습니다.")
        feedback_list.append("왼쪽 어깨가 너무 내려가있어 자신감이 없어보이는 습관이 있습니다.")

    # 자신감이 없어보이거나 피곤해 보이는 자세
    if pose_landmarks.landmark[0].visibility > 0.5 and pose_landmarks.landmark[0].y < 0.3:
        print("고개가 너무 숙여져있어 피곤해 보이는 습관이 있습니다.")
        feedback_list.append("고개가 너무 숙여져있어 피곤해 보이는 습관이 있습니다.")

    # 오만하거나 무례해 보이는 자세
    if pose_landmarks.landmark[0].visibility > 0.5 and pose_landmarks.landmark[0].y > 0.7:
        print("고개가 너무 들어있어 오만하거나 무례해 보이는 습관이 있습니다.")
        feedback_list.append("고개가 너무 들어있어 오만하거나 무례해 보이는 습관이 있습니다.")

    # 상체가 앞으로 쏠린 경우
    if (pose_landmarks.landmark[11].z > 0.3) and \
       (pose_landmarks.landmark[12].z > 0.3) and \
       (pose_landmarks.landmark[0].z > 0.3):
        feedback_list.append("상체가 앞으로 쏠려 피곤해 보이거나 집중하지 않는 상태입니다.")

    # 손 랜드마크 분석
    # if hands_landmarks:
    #     for hand_landmarks in hands_landmarks:
    #         # 손을 턱에 기댄 경우
    #         if hand_landmarks.landmark[12].visibility > 0.5 and pose_landmarks.landmark[9].visibility > 0.5:
    #             distance = euclidean_distance(hand_landmarks.landmark[12], pose_landmarks.landmark[9])
    #             print("왼손과 턱 사이의 거리: ", distance)
    #             if distance < 1.3:
    #                 print("왼손이 턱에 기댄 습관이 있습니다.")
    #                 feedback_list.append("왼손이 턱에 기댄 습관이 있습니다.")
    #             else:
    #                 print("왼손이 턱에 기댄 습관이 없습니다.")

    #         # 손으로 코를 만지는 경우
    #         if hand_landmarks.landmark[12].visibility > 0.5 and pose_landmarks.landmark[0].visibility > 0.5:
    #             distance = euclidean_distance(hand_landmarks.landmark[12], pose_landmarks.landmark[0])
    #             print("손가락과 코 사이의 거리: ", distance)
    #             if distance < 1:
    #                 print("손가락으로 코를 만지는 습관이 있습니다.")
    #                 feedback_list.append("손가락으로 코를 만지는 습관이 있습니다.")
    #             else:
    #                 print("손가락으로 코를 만지는 습관이 없습니다.")

    #         # 손으로 눈을 만지는 경우
    #         if hand_landmarks.landmark[12].visibility > 0.5 and pose_landmarks.landmark[1].visibility > 0.5:
    #             distance = euclidean_distance(hand_landmarks.landmark[12], pose_landmarks.landmark[1])
    #             print("손가락과 눈 사이의 거리: ", distance)
    #             if distance < 0.8:
    #                 print("손가락으로 눈을 만지는 습관이 있습니다.")
    #                 feedback_list.append("손가락으로 눈을 만지는 습관이 있습니다.")
    #             else:
    #                 print("손가락으로 눈을 만지는 습관이 없습니다.")

    #         # 손으로 입을 만지는 경우
    #         if hand_landmarks.landmark[12].visibility > 0.5 and (pose_landmarks.landmark[9].visibility > 0.5 or pose_landmarks.landmark[10].visibility > 0.5):
    #             distance = euclidean_distance(hand_landmarks.landmark[12], pose_landmarks.landmark[9])
    #             print("손가락과 입 사이의 거리: ", distance)
    #             if distance < 1.2:
    #                 print("손가락으로 입을 만지는 습관이 있습니다.")
    #                 feedback_list.append("손가락으로 입을 만지는 습관이 있습니다.")
    #             else:
    #                 print("손가락으로 입을 만지는 습관이 없습니다.")

    return feedback_list
    