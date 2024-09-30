import openai
import tempfile
import shutil
import os
import re
from langdetect import detect
from dotenv import load_dotenv

load_dotenv()

def transcribe_audio(file_stream, language="ko") -> str:

    if language not in ["ko", "en"]:
        raise ValueError("지원되지 않는 언어입니다. 'ko' 또는 'en'만 사용 가능합니다.")

    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise ValueError("API_KEY가 없습니다.")

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
                language=language,
                response_format="text"
            )
            
            # 언어 감지 및 필터링
            detected_lang = detect(response)
            if detected_lang not in ['ko', 'en']:
                # 한국어나 영어가 아닌 경우 빈 문자열 반환
                return ""
            
            # 한국어나 영어 문자만 허용
            filtered_response = re.sub(r'[^가-힣a-zA-Z\s]', '', response)

            # 반환된 텍스트가 "MBC 뉴스 이덕영입니다."인 경우 빈 문자열 반환
            return "" if response.strip() == "MBC 뉴스 이덕영입니다." else response
    finally:
        # 임시 파일 삭제
        os.remove(temp_file_path)
