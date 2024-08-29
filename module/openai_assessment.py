# openai_assessment.py
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

def evaluate_answer(question: str, answer: str, years: str, job: str) -> str:
    prompt = f"""
    # Role
    You are an expert interviewer evaluating technical responses.

    # Task
    - Evaluate the provided answer based on the given question.
    - Consider the candidate's job role ({job}) and years of experience ({years} years) when assessing the answer.
    - Focus exclusively on the technical content of the answer.
    - Provide a score from 0 to 100 on how well the answer fits the question, considering the technical depth appropriate for the role and experience level.
    - Explain why the answer is rated as such, focusing solely on its technical content and relevance to the role and experience level.
    - Provide a model answer that would be considered ideal for this question given the candidate's role and experience level.

    # Policy
    - Ensure the evaluation is detailed and justifiable.
    - Only assess the technical aspects of the answer. Do not evaluate personality, job fit, or organizational fit.
    - Provide clear explanations for the rating given, including how the job role and experience level influenced the assessment.
    - Include a model answer tailored to the candidate’s role and experience level to set a standard for ideal responses.
    - Provide all information and responses only in Korean.

    # Output Format
    {{
        "질문": "{question}",
        "답변": "{answer}",
        "평가": "Evaluation score here",
        "설명": "explanation here",
        "모범답안": "Model answer here"
    }}
    """

    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are an expert interviewer evaluating technical responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    try:
        response_content = completion.choices[0].message.content
        result = json.loads(response_content)
        
        # Ensure the evaluation score is a number
        if isinstance(result.get("평가"), str) and result["평가"].isdigit():
            result["평가"] = int(result["평가"])
        
        return result
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 기본 구조 반환
        return {
            "질문": question,
            "답변": answer,
            "평가": "응답을 평가하는 데 문제가 발생했습니다.",
            "설명": "설명을 생성하는 데 문제가 발생했습니다.",
            "모범답안": "모범답안을 생성하는 데 문제가 발생했습니다."
        }
