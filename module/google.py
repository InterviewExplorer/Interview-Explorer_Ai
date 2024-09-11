import os
from google.cloud import speech_v1p1beta1 as speech
from dotenv import load_dotenv
import tempfile
import shutil

# .env 파일 로드
load_dotenv()

def transcribe_audio(file_stream) -> str:
    print("@@@@@@@@transcribe_audio 실행")
    """
    주어진 FLAC 파일 스트림을 텍스트로 변환합니다.
    
    :param file_stream: 변환할 FLAC 파일의 스트림
    :return: 변환된 텍스트
    """
    # Google Cloud 인증 환경 변수 설정
    google_application_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not google_application_credentials:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.")

    # Google Cloud Speech-to-Text 클라이언트 생성
    client = speech.SpeechClient()

    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as temp_file:
        shutil.copyfileobj(file_stream, temp_file)
        temp_file_path = temp_file.name

    try:
        # FLAC 파일 읽기
        with open(temp_file_path, "rb") as audio_file:
            content = audio_file.read()

        # 오디오 파일과 구성 설정
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,  # FLAC 파일 인코딩 설정
            sample_rate_hertz=44100,  # FLAC의 일반 샘플링 레이트
            language_code="ko-KR",  # 한국어 설정
        )

        # 텍스트로 변환
        print("Sending audio to Google Cloud Speech-to-Text...")
        response = client.recognize(config=config, audio=audio)

        # 텍스트 추출
        transcript = ''
        for result in response.results:
            transcript += result.alternatives[0].transcript + '\n'

        return transcript.strip()
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise
    finally:
        # 임시 파일 삭제
        os.remove(temp_file_path)
