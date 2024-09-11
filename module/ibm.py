import os
import tempfile
import shutil
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from dotenv import load_dotenv  # .env 파일 로드

# .env 파일 로드
load_dotenv()

def transcribe_audio(file_stream) -> str:
    """
    주어진 MP3 파일 스트림을 IBM Speech to Text API를 사용하여 텍스트로 변환합니다.
    
    :param file_stream: 변환할 MP3 파일의 스트림
    :return: 변환된 텍스트
    """
    # .env 파일에서 IBM API 키와 URL을 불러오기
    ibm_api_key = os.getenv('IBM_KEY')
    ibm_service_url = os.getenv('IBM_URL')
    
    if not ibm_api_key or not ibm_service_url:
        raise ValueError("IBM API 키 또는 URL이 설정되지 않았습니다.")

    # IBM Speech to Text 서비스에 연결
    authenticator = IAMAuthenticator(ibm_api_key)
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url(ibm_service_url)

    # 임시 파일 생성 (MP3 파일로 저장)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        shutil.copyfileobj(file_stream, temp_file)
        temp_file_path = temp_file.name

    try:
        # MP3 파일을 열어 IBM Speech to Text API로 변환 요청
        with open(temp_file_path, 'rb') as audio_file:
            speech_recognition_results = speech_to_text.recognize(
                audio=audio_file,
                content_type='audio/mp3',
                # model='ko-KR_BroadbandModel'  #고음질 오디오 파일에 적합
                model='ko-KR_NarrowbandModel'   #낮은 비트레이트 음성파일
            ).get_result()

        # 텍스트 추출
        transcript = ''
        for result in speech_recognition_results['results']:
            transcript += result['alternatives'][0]['transcript'] + '\n'

        return transcript.strip()
    finally:
        # 임시 파일 삭제
        os.remove(temp_file_path)
