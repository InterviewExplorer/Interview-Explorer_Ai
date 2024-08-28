# whisper_medium.py

import whisper

def transcribe_audio(file_path: str) -> str:
    """
    주어진 MP3 파일을 텍스트로 변환합니다.

    :param file_path: 변환할 MP3 파일의 경로
    :return: 변환된 텍스트
    """
    # Whisper-medium 모델 로드
    model = whisper.load_model("medium")

    # 모델을 사용하여 오디오 파일을 텍스트로 변환
    result = model.transcribe(file_path)
    
    return result["text"]
