import math

def analyze_landmarks(landmarks):
    feedback_list = [""]

    def landmark_exists(key):
        return key in landmarks and landmarks[key] is not None
    
    print("랜드마크 부위별: ", landmarks)

    # 유클리드 거리 계산
    def euclidean_distance(point1, point2):
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)

    # 손을 턱에 기댄 경우
    if landmark_exists("left_wrist") and landmark_exists("mouth_left"):
        distance = euclidean_distance(landmarks["left_wrist"], landmarks["mouth_left"])
        print("왼손과 턱 사이의 거리: ", distance)
        if distance < 1:
            print("왼손이 턱에 기댄 습관이 있습니다.")
            feedback_list.append("왼손이 턱에 기댄 습관이 있습니다.")
        else:
            print("왼손이 턱에 기댄 습관이 없습니다.")

    if landmark_exists("right_wrist") and landmark_exists["mouth_right"]:
        distance = euclidean_distance(landmarks["right_wrist"], landmarks["mouth_right"])
        print("오른손과 턱 사이의 거리: ", distance)
        if distance < 1:
            print("오른손이 턱에 기댄 습관이 있습니다.")
            feedback_list.append("오른손이 턱에 기댄 습관이 있습니다.")
        else:
            print("오른손이 턱에 기댄 습관이 없습니다.")

    # 팔짱을 낀 경우
    if landmark_exists("left_wrist") and landmark_exists("right_elbow"):
        distance = euclidean_distance(landmarks["left_wrist"], landmarks["right_elbow"])
        print("왼손과 오른쪽 팔꿈치 사이의 거리: ", distance)
        if distance < 0.7:  # 특정 거리 임계값 설정
            print("왼손이 팔짱을 낀 습관이 있습니다.")
            feedback_list.append("왼손이 팔짱을 낀 습관이 있습니다.")
        else:
            print("왼손이 팔짱을 낀 습관이 없습니다.")

    if landmark_exists("right_wrist") and landmark_exists("left_elbow"):
        distance = euclidean_distance(landmarks["right_wrist"], landmarks["left_elbow"])
        print("오른손과 왼쪽 팔꿈치 사이의 거리: ", distance)
        if distance < 0.7:
            print("오른손이 팔짱을 낀 습관이 있습니다.")
            feedback_list.append("오른손이 팔짱을 낀 습관이 있습니다.")
        else:
            print("오른손이 팔짱을 낀 습관이 없습니다.")

    # 긴장한 자세
    if landmarks["right_shoulder"].y < 1:
        print("오른쪽 어깨가 너무 올라가있어 긴장한 습관이 있습니다.")
        feedback_list.append("오른쪽 어깨가 너무 올라가있어 긴장한 습관이 있습니다.")

    if landmarks["left_shoulder"].y > 1:
        print("왼쪽 어깨가 너무 올라가있어 긴장한 습관이 있습니다.")
        feedback_list.append("왼쪽 어깨가 너무 올라가있어 긴장한 습관이 있습니다.")

    # 자신감이 없어 보이는 자세
    if landmarks["right_shoulder"].y < 0.7:
        print("오른쪽 어깨가 너무 내려가있어 자신감이 없어보이는 습관이 있습니다.")
        feedback_list.append("오른쪽 어깨가 너무 내려가있어 자신감이 없어보이는 습관이 있습니다.")

    if landmarks["left_shoulder"].y < 0.7:
        print("왼쪽 어깨가 너무 내려가있어 자신감이 없어보이는 습관이 있습니다.")
        feedback_list.append("왼쪽 어깨가 너무 내려가있어 자신감이 없어보이는 습관이 있습니다.")

    # 자신감이 없어보이거나 피곤해 보이는 자세
    if landmarks["nose"].y < 0.3:
        print("고개가 너무 숙여져있어 피곤해 보이는 습관이 있습니다.")
        feedback_list.append("고개가 너무 숙여져있어 피곤해 보이는 습관이 있습니다.")

    # 오만하거나 무례해 보이는 자세
    if landmarks["nose"].y > 0.8:
        print("고개가 너무 들어있어 오만하거나 무례해 보이는 습관이 있습니다.")
        feedback_list.append("고개가 너무 들어있어 오만하거나 무례해 보이는 습관이 있습니다.")

    # # 양쪽 어깨와 코가 앞으로 쏠릴 때
    # if landmarks["left_shoulder"].z > 0.3 and landmarks["right_shoulder"].z > 0.3 and landmarks["nose"].z > 0.3:
    #     print("상체가 앞으로 쏠려 피곤해 보이거나 집중하지 않는 상태입니다.")
    #     feedback_list.append("상체가 앞으로 쏠려 피곤해 보이거나 집중하지 않는 상태입니다.")

        # 상체가 앞으로 쏠린 경우
    if (landmark_exists("left_shoulder") and landmarks["left_shoulder"].z > 0.3) and \
       (landmark_exists("right_shoulder") and landmarks["right_shoulder"].z > 0.3) and \
       (landmark_exists("nose") and landmarks["nose"].z > 0.3):
        feedback_list.append("상체가 앞으로 쏠려 피곤해 보이거나 집중하지 않는 상태입니다.")

    # 손으로 코를 만지는 경우
    if landmark_exists("left_wrist") and landmark_exists("nose"):
        distance = euclidean_distance(landmarks["left_wrist"], landmarks["nose"])
        print("왼손과 코 사이의 거리: ", distance)
        if distance < 0.2:
            feedback_list.append("왼손으로 코를 만지는 습관이 있습니다.")
        else:
            print("왼손으로 코를 만지는 습관이 없습니다.")

    if landmark_exists("right_wrist") and landmark_exists("nose"):
        distance = euclidean_distance(landmarks["right_wrist"], landmarks["nose"])
        print("오른손과 코 사이의 거리: ", distance)
        if distance < 0.2:
            feedback_list.append("오른손으로 코를 만지는 습관이 있습니다.")
        else:
            print("오른손으로 코를 만지는 습관이 없습니다.")

    # 손으로 눈을 만지는 경우
    if landmark_exists("left_wrist") and landmark_exists("left_eye"):
        distance = euclidean_distance(landmarks["left_wrist"], landmarks["left_eye"])
        print("왼손과 왼쪽 눈 사이의 거리: ", distance)
        if distance < 0.2:
            feedback_list.append("왼손으로 눈을 만지는 습관이 있습니다.")
        else:
            print("왼손으로 눈을 만지는 습관이 없습니다.")

    if landmark_exists("right_wrist") and landmark_exists("right_eye"):
        distance = euclidean_distance(landmarks["right_wrist"], landmarks["right_eye"])
        print("오른손과 오른쪽 눈 사이의 거리: ", distance)
        if distance < 0.2:
            feedback_list.append("오른손으로 눈을 만지는 습관이 있습니다.")
        else:
            print("오른손으로 눈을 만지는 습관이 없습니다.")

    # 손으로 입을 만지는 경우
    if landmark_exists("left_wrist") and (landmark_exists("mouth_left") or landmark_exists("mouth_right")):
        distance = euclidean_distance(landmarks["left_wrist"], landmarks["mouth_left"])
        print("왼손과 입 사이의 거리: ", distance)
        if distance < 0.2:
            feedback_list.append("왼손으로 입을 만지는 습관이 있습니다.")
        else:
            print("왼손으로 입을 만지는 습관이 없습니다.")

    if landmark_exists("right_wrist") and (landmark_exists("mouth_left") or landmark_exists("mouth_right")):
        distance = euclidean_distance(landmarks["right_wrist"], landmarks["mouth_left"])
        if distance < 0.2:
            feedback_list.append("오른손으로 입을 만지는 습관이 있습니다.")
        else:
            print("오른손으로 입을 만지는 습관이 없습니다.")
    return feedback_list