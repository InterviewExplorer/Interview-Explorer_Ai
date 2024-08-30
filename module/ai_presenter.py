from typing import Dict, Any
import requests
import asyncio
POST_URL = "https://api.d-id.com/clips"
GET_URL_TEMPLATE = "https://api.d-id.com/clips/{}"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Basic YOUR_AUTHORIZATION_TOKEN"  # 인증 토큰을 추가하세요
}

async def fetch_result_url(question: str) -> Dict[str, Any]:
    payload = {
        "presenter_id": "sophia-ndjDZ_Osqg",
        "script": {
            "type": "text",
            "subtitles": "false",
            "provider": {
                "type": "microsoft",
                "voice_id": "ko-KR-JiMinNeural"
            },
            "input": question,
            "ssml": "false",
            "audio_url": "https://path.to/audio.mp3"
        },
        "config": { "result_format": "mp4" },
        "presenter_config": { "crop": { "type": "wide" } }
    }
    
    # POST 요청 보내기
    response = requests.post(POST_URL, json=payload, headers=HEADERS)
    response_data = response.json()
    
    # 'id' 값 가져오기
    clip_id = response_data.get('id')
    if not clip_id:
        return {"question": question, "error": "No 'id' found in response"}
    
    # GET 요청을 통해 결과 URL 확인
    get_url = GET_URL_TEMPLATE.format(clip_id)
    
    # 기다려야 할 수도 있음 (실제 상황에서는 적절한 대기 시간을 설정하거나 폴링 로직을 구현해야 할 수 있습니다)
    await asyncio.sleep(10)  # 10초 대기 (필요에 따라 조정)

    response = requests.get(get_url, headers=HEADERS)
    response_data = response.json()
    
    # 'result_url' 가져오기
    result_url = response_data.get('clips', [{}])[0].get('result_url')
    
    if result_url:
        return {question: result_url}
    else:
        return {question: "No 'result_url' found"}