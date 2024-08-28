import subprocess
import io
import os

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
        'ffmpeg', '-i', temp_webm_path,
        '-q:a', '0',  # 오디오 품질을 최적화
        '-map', 'a',  # 오디오 스트림만 추출
        mp3_path
    ]
    subprocess.run(command, check=True)
    
    # 임시 파일 삭제
    os.remove(temp_webm_path)
