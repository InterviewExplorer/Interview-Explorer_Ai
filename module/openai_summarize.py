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
    evaluation_items_text = '\n'.join(evaluation_items)
    
    # 요약을 위한 프롬프트 생성
    prompt = f"""
    # Role
    You are an expert interviewer evaluating technical responses. The interviewee is the candidate whose responses you are evaluating.

    # Task
    - Review the following technical evaluation content:

    {evaluation_items_text}

    - Based on the evaluation content, write a technical interview assessment in 5 lines or less.
    - The assessment should be concise and provide a summary of the candidate's performance.

    # Policy
    - Ensure the assessment is clear and focused on the technical content.
    - Do not include personal opinions or unrelated information.
    - Keep the response within 5 lines, providing a summary of the evaluation.
    - Provide all information and responses only in Korean.

    # Output Format
    {{
        "총평": "the assessment is here"
    }}
    """
    
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

