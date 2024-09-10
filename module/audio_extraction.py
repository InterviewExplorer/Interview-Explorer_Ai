import subprocess
import io
import os
import cv2
import mediapipe as mp
from module.check_distance import analyze_video_landmarks
from module.pose_feedback import analyze_pose_movement
def convert_webm_to_mp3(webm_file: io.BytesIO, mp3_path: str):
    """
    메모리에서 webm 파일을 mp3 형식으로 변환하고 포즈를 분석합니다.
    :param webm_file: 메모리에서의 webm 파일
    :param mp3_path: 변환할 mp3 파일의 경로
    :return: 중복이 제거된 포즈 분석 피드백
    """
    # 웹엠 파일을 임시로 저장할 폴더 확인
    temp_webm_path = 'audio/temp_video.webm'
    # 웹엠 파일을 임시로 저장
    with open(temp_webm_path, 'wb') as temp_file:
        temp_file.write(webm_file.read())

    # 변환 명령어 실행
    command = [
        'ffmpeg', '-i', temp_webm_path, '-y',
        '-q:a', '0',  # 오디오 품질을 최적화
        '-map', 'a',  # 오디오 스트림만 추출
        mp3_path
    ]
    subprocess.run(command, check=True)

    # BlazePose 복잡도 선택, 명시하지 않으면 디폴트 값 1
    MODEL_COMPLEXITY = {
        "LITE": 0,
        "FULL": 1,
        "HEAVY": 2
    }

    # MediaPipe Pose 모듈 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=MODEL_COMPLEXITY["FULL"],
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 비디오 파일을 열기
    cap = cv2.VideoCapture(temp_webm_path)

    # 비디오 파일이 없을 시
    if not cap.isOpened():
        raise ValueError("비디오 파일을 열 수 없습니다. 파일 경로를 확인해주세요.")

    all_pose_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR 이미지를 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 성능 향상을 위한 편집기능 끄기
        rgb_frame.flags.writeable = False

        # 포즈 추정 수행
        pose_results = pose.process(rgb_frame)
        
        # 포즈 결과 저장 (랜드마크가 감지된 경우에만)
        if pose_results.pose_landmarks:
            all_pose_results.append(pose_results.pose_landmarks)

    # 자원 해제
    cap.release()

    # 임시 파일 삭제
    os.remove(temp_webm_path)

    # 전체 비디오에 대한 포즈 분석 및 중복 제거된 피드백 수집
    feedback_set = set(analyze_video_landmarks(all_pose_results))

    # List로 리턴 받은 feedback_set을 넘겨주며 개별 피드백 생성
    feedback = analyze_pose_movement(list(feedback_set))

    # 최종 피드백 출력 (중복 제거됨)
    # final_feedback = "\n".join(feedback_set)

    # print("지적 목록(audio_extraction.py): ", "".join(feedback_set))
    return feedback