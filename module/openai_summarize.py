from openai import OpenAI
import os
from dotenv import load_dotenv
import json

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

def summarize_text(evaluations):
    # 평가 내용을 문자열로 변환
    evaluation_items = [f"평가 {i+1}: {evaluation}" for i, evaluation in enumerate(evaluations.values())]
    
    # 요약을 위한 프롬프트 생성
    prompt = (
        "다음은 기술 면접에 대한 평가 내용입니다:\n\n" +
        "\n".join(evaluation_items) +
        "\n\n위의 평가 내용을 바탕으로 5줄 내외로 요약, 정리해 주세요."
    )
    
    # OpenAI API를 사용하여 요약 생성
    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are an expert interviewer evaluating technical responses."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,  # 5줄 내외로 요약
        temperature=0,
    )
        
    try:
        response_content = completion.choices[0].message.content
        return response_content
    except json.JSONDecodeError:
        return "JSONDecodeError 문제가 발생했습니다."
    except Exception as e:
        # 에러 발생 시 기본 구조 반환
        return "평가 내용을 요약하는 데 문제가 발생했습니다."

