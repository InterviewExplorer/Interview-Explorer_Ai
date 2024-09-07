import subprocess
import io
import os
import cv2
import mediapipe as mp
from module.pose_feedback import analyze_pose_movement

def convert_webm_to_mp3(webm_file: io.BytesIO, mp3_path: str):
    """
    메모리에서 webm 파일을 mp3 형식으로 변환합니다.
    
    :param webm_file: 메모리에서의 webm 파일
    :param mp3_path: 변환할 mp3 파일의 경로
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

    # MediaPipe 포즈 모듈 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # 비디오 파일을 열기
    cap = cv2.VideoCapture(temp_webm_path)

    # 비디오 파일이 없을 시
    if not cap.isOpened():
        raise ValueError("비디오 파일을 열 수 없습니다. 파일 경로를 확인해주세요.")

    feedback_list = []

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

        # 편집기능 다시 켜기
        rgb_frame.flags.writeable = True

        # RGB 이미지를 BGR로 다시 변환 (OpenCV 사용을 위해)
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # 랜드마크가 감지되었는지 확인 후 그리기
        if pose_results.pose_landmarks:
            # 랜드마크와 연결선 그리기
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        # 결과를 화면에 표시
        # cv2.imshow('Pose Detection', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # 포즈 분석 및 피드백 수집
    feedback = analyze_pose_movement(pose_results)

    if feedback:
        feedback_list.extend(feedback)

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    
    # 임시 파일 삭제
    os.remove(temp_webm_path)

    # 최종 피드백 출력
    final_feedback = "".join(feedback_list)

    return final_feedback