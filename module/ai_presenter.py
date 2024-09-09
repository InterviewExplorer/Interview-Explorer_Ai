from typing import Dict, Any
import requests
import asyncio
import os

POST_URL = "https://api.d-id.com/talks"
GET_URL_TEMPLATE = "https://api.d-id.com/talks/"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Basic IFpIVnlhV1IxY21rek9UTTVRR2R0WVdsc0xtTnZiUTpHT0x3SUlaa0Y5WmFramhnVkNzcDY="  # 인증 토큰을 추가하세요
}

async def fetch_result_url(key: str, question: str) -> Dict[str, Any]:
    payload = {
        "source_url": "https://i.ibb.co/zm3fBjp/test.jpg", #소스 얼굴
    "script": {
        "type": "text",
        "subtitles": "false",
        "provider": {
            "type": "microsoft",
            "voice_id": "ko-KR-JiMinNeural"
        },
        "input": question
    },
    "config": {
        "fluent": "true",
        "pad_audio": "0.0"
    }
}
    
    # POST 요청 보내기
    # 여기 토큰 소모 때문에 주석처리
    # response = requests.post(POST_URL, json=payload, headers=HEADERS)
    # response_data = response.json()
    
    # 'id' 값 가져오기
    # 여기 토큰 소모 때문에 주석처리
    # clip_id = response_data.get('id')
    clip_id = "tlk_OspBjf2NVpGEw2T3nreVF"
    # print(clip_id)
    
    
    if not clip_id:
        return {"question": question, "error": "No 'id' found in response"}
    
    # GET 요청을 통해 결과 URL 확인
    get_url = "https://api.d-id.com/talks/{}".format(clip_id)
    # print(get_url)
    # 폴링 로직을 사용하여 결과가 준비될 때까지 기다림
    # for _ in range(10):  # 최대 20번 재시도
    #     await asyncio.sleep(5)  # 5초 대기 후 다시 요청 (필요에 따라 조정)

    response = requests.get(get_url, headers=HEADERS)
    response_data = response.json()
    
    # 'result_url' 가져오기
    result_url = response_data.get('result_url')
    
    if result_url:
        return {key: result_url}
    else:
        return {key: "No 'result_url' found"}