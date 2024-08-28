import whisper
import tempfile
import shutil
import os

def transcribe_audio(file_stream) -> str:
    """
    주어진 MP3 파일 스트림을 텍스트로 변환합니다.
    
    :param file_stream: 변환할 MP3 파일의 스트림
    :return: 변환된 텍스트
    """
    # Whisper-medium 모델 로드
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("medium")

    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        shutil.copyfileobj(file_stream, temp_file)
        temp_file_path = temp_file.name

    try:
        # 모델을 사용하여 오디오 파일을 텍스트로 변환
        result = model.transcribe(temp_file_path)
        return result["text"]
    finally:
        # 임시 파일 삭제
        os.remove(temp_file_path)
