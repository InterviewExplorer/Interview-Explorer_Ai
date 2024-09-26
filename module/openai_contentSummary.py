from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import time

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 키 가져오기
api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY가 없습니다.")

# GPT 모델 가져오기
gpt_model = os.getenv("gpt")
if gpt_model is None:
    raise ValueError("GPT_Model이 없습니다.")

# OpenAI 클라이언트 초기화 및 api키 등록
client = OpenAI(api_key=api_key)

def summaryOfContent(content: str) -> dict:
    prompt = f"""
    # Role
    You are an expert in summarizing.

    # Task
    - Summarize {content} into 700 bytes or less.
    
    # Policy
    - Never add anything that is not in {content}.
    - Generate the value in English.
    - You must strictly adhere to the following JSON format.
    - Only include the values corresponding to the value in the output format.

    # Output Format
    {{
        "Summary":""
    }}
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": "You are an expert in summarizing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            response_content = completion.choices[0].message.content
            result = json.loads(response_content)
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패, 재시도 중... (시도 {attempt + 1}/{max_retries})")
            time.sleep(2)  # 짧은 대기 후 재시도

    # 모든 재시도 실패 시 기본 구조 반환
    return {"error": "JSONDecodeError"}
