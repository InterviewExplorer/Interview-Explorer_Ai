from typing import Dict, Any
import requests
import asyncio
import os

POST_URL = "https://api.d-id.com/talks"
GET_URL_TEMPLATE = "https://api.d-id.com/talks/"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": os.getenv("did") # 인증 토큰을 추가하세요
}

async def fetch_result_url(key: str, question: str) -> Dict[str, Any]:
    payload = {
        "source_url": "https://i.ibb.co/Y7hCJ2Z/51ee3aa1-c0e5-45e8-8fb9-1c47bceb8c8d-1.jpg", #소스 얼굴
    "script": {
        "type": "text",
        "subtitles": "false",
        "provider": {
            "type": "microsoft",
            "voice_id": "ko-KR-BongJinNeural"
        },
        "input": question
    },
    "config": {
        "fluent": "false",
        "stitch": "true"

    }
}
    response = requests.post(POST_URL, json=payload, headers=HEADERS)
    response_data = response.json()

    # 'id' 값 가져오기
    clip_id = response_data.get('id')

    # Retry until we have a clip_id or reach max attempts
    attempts = 0
    max_attempts = 10
    while not clip_id and attempts < max_attempts:
        # print("No clip_id found, retrying...")
        await asyncio.sleep(5)  # Wait before retrying
        response = requests.post(POST_URL, json=payload, headers=HEADERS)
        response_data = response.json()
        clip_id = response_data.get('id')
        print(clip_id)
        attempts += 1

    if not clip_id:
        return {"question": question, "error": "No 'id' found in response after retries"}

    # GET 요청을 통해 결과 URL 확인
    get_url = f"https://api.d-id.com/talks/{clip_id}"

    # Retry until we have a result_url or reach max attempts
    attempts = 0
    max_attempts = 10
    while attempts < max_attempts:
        response = requests.get(get_url, headers=HEADERS)
        response_data = response.json()
        result_url = response_data.get('result_url')

        if result_url:
            return {key: result_url}

        print("Result URL not ready, retrying...")
        await asyncio.sleep(5)  # Wait before retrying
        attempts += 1

    return {key: "No 'result_url' found after retries"}