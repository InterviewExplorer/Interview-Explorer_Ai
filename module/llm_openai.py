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

# OpenAI 클라이언트 초기화 및 API 키 등록
client = OpenAI(api_key=api_key)

def follow_Q(answer: str, years: str, job: str, questions: dict) -> str:
    # 기존 질문들을 리스트 형태로 변환
    existing_questions = "\n".join(f"- {q}" for q in questions.values())

    prompt = f"""
    # Role
    You are an interviewer evaluating technical responses.

    # Task
    - Generate exactly one relevant question based on ({answer}).
    - The question should be appropriate for the candidate's job role ({job}) and years of experience ({years} years).
    - Focus on technical content, adjusting the depth and difficulty according to the job role and experience level.
    - Questions should be constructed to accurately assess technical content.
    - Adjust the difficulty and depth according to the job role and experience level.


    # Policy
    - The output should include only the question in the specified format.
    - Do not include any additional content or explanations.
    - Ensure that the generated question does not repeat any of the existing questions provided below.
    - All questions should be clear and specific, and should assess the candidate's technical skills.
    - Responses must be written in Korean.
    - Responses must be in JSON format.

    # Context
    - Here are the existing questions:
    {existing_questions}

    # Output Format
    {{
        "question":""
    }}
    """

    # API 호출
    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are an expert interviewer evaluating technical responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    response_content = completion.choices[0].message.content

    return json.loads(response_content)
