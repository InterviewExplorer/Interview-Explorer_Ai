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

def evaluate_answer(question: str, answer: str, years: str, job: str, type: str) -> dict:
    if type == "technical":
        prompt = f"""
        # Role
        You are a technical interviewer with expertise in conducting interviews.

        # Task
        1. Check if the answer ({answer}) contains any technical content. If there is no technical content, score it as 0.
        2. If the answer contains technical content, check whether it is directly related to the technology mentioned in the question ({question}). If the answer is not directly related to the technology in the question, assign a score of 30 or lower, depending on the relevance of the answer.
        3. If the answer is related to the technology in the question, evaluate the appropriateness of the answer based on the candidate's job role ({job}) and {years} years of experience. Assign a score from 0 to 100.
        4. Explain the reason for the assigned score, focusing only on the technical content. The explanation must be provided in Korean.
        5. Provide a model answer that would be considered ideal for this question, tailored to the candidate's job role and experience level. The model answer must be provided in Korean.

        # Policy
        - Ensure the evaluation is detailed and justifiable.
        - Evaluate only the technical aspects of the answer; do not consider personality, job fit, or organizational fit.
        - Clearly explain the reasoning behind the assigned score, including how the job role and experience level influenced the evaluation. The explanation must be in Korean.
        - Provide a model answer that reflects the appropriate depth for the job role and experience level, in Korean.
        - Responses must be in JSON format.
        - Place the score in the `score` value of the JSON output.
        - Place the explanation in the `explanation` value of the JSON output.
        - Place the model answer in the `model` value of the JSON output.
        - Do not include any additional explanations beyond the specified output format.

        # Output Format
        {{
            "score": "",
            "explanation": "",
            "model": ""
        }}
        """
    elif type == "behavioral":
        prompt = f"""
        # Role
        You are an expert interviewer specializing in evaluating interpersonal aspects.

        # Task    
        1. From an interpersonal interviewer’s perspective, explain the intention of ({question}) in 3 sentences or less.
        2. Determine whether ({answer}) is an appropriate response to ({question}). If the answer is not related to the question, assign a score of 0.
        3. Evaluate whether ({answer}) aligns with the intention of ({question}) and assign a score from 0 to 100.
        4. Explain the reason for the assigned score. The explanation should focus solely on interpersonal aspects and be provided in Korean.

        # Policy
        - Ensure the evaluation is detailed and justifiable.
        - Evaluate only the interpersonal aspects of the answer.
        - Do not consider the technical aspects.
        - Clearly explain the reasoning behind the assigned score, including how the job role and experience level influenced the evaluation. The explanation must be in Korean.
        - Responses must be in JSON format.
        - Place the score in the `score` value of the JSON output.
        - Place the explanation in the `explanation` value of the JSON output.
        - Place the intention of the question in the `intention` value of the JSON output.
        - Do not include any additional explanations beyond the specified output format.
        
        # Output Format
        {{
            "score": "",
            "explanation": "",
            "intention": ""
        }}
        """
    else:
        raise ValueError("Invalid type provided. Must be 'technical' or 'behavioral'.")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": "You are a professional interviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            response_content = completion.choices[0].message.content
            result = json.loads(response_content)
            
            # Ensure the evaluation score is a number
            if isinstance(result.get("score"), str) and result["score"].isdigit():
                result["score"] = int(result["score"])
            
            return result
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패, 재시도 중... (시도 {attempt + 1}/{max_retries})")
            time.sleep(2)  # 짧은 대기 후 재시도

    # 모든 재시도 실패 시 기본 구조 반환
    return {"error": "JSONDecodeError"}
