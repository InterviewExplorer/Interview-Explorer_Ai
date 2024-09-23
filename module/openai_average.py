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

def calculate_average(years: str, job: str, type: str) -> dict:
    if type == "technical":
        prompt = f"""
        # Role
        You are a technical interviewer with expertise in conducting interviews.

        # Task
        - Calculate the average values for each element for a ({job}) with ({years}) years of experience.
        - Measure these elements on a scale from 1 to 100.

        # Policy
        - Responses must be in JSON format.
        - Place scores for each element according to the corresponding keys in the output format.
        - Do not include any additional explanations beyond the specified output format.

        # Output Format
        {{
            "problem_solving": null,
            "technical_understanding": null,
            "logical_thinking": null,
            "learning_ability": null,
            "collaboration_communication": null
        }}
        """
    elif type == "behavioral":
        prompt = f"""
        # Role
        You are an expert interviewer specializing in evaluating interpersonal aspects.

        # Task
        - Calculate the average values for the elements of honesty reliability, interpersonal skills, self motivation passion, adaptability, self awareness for a ({job}) with ({years}) years of experience.
        - Measure these elements on a scale from 1 to 100.
        
        # Policy
        - Responses must be in JSON format.
        - Place scores for each element according to the corresponding keys in the output format.
        - Do not include any additional explanations beyond the specified output format.
        
        # Output Format
        {{
            "honesty_reliability": null,
            "interpersonal_skills": null,
            "self_motivation_passion": null,
            "adaptability": null,
            "self_awareness": null
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
