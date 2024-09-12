import openai
import tempfile
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

def transcribe_audio(file_stream) -> str:
    """
    OpenAI API를 사용하여 주어진 MP3 파일 스트림을 텍스트로 변환합니다.

    :param file_stream: 변환할 MP3 파일의 스트림
    :return: 변환된 텍스트
    """

    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise ValueError("API_KEY가 없습니다.")

    # OpenAI API 키 설정
    openai.api_key = api_key

    # 파일 스트림을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        shutil.copyfileobj(file_stream, temp_file)
        temp_file_path = temp_file.name

    try:
        # OpenAI API를 사용하여 텍스트로 변환
        with open(temp_file_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language="ko",  # 한국어로 설정
                response_format="text"
            )
            return response
    finally:
        # 임시 파일 삭제
        os.remove(temp_file_path)
