from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import time
from typing import Dict

load_dotenv()

api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY가 없습니다.")

gpt_model = os.getenv("gpt")
if gpt_model is None:
    raise ValueError("GPT_Model이 없습니다.")

client = OpenAI(api_key=api_key)

def evaluate_speaking(answers: Dict[str, str]) -> dict:
    # 모든 답변을 하나의 문자열로 결합
    combined_text = " ".join(answers.values())
    print("combined_text", combined_text)

    prompt = f"""
    # Role
    You are an interviewer specializing in evaluating language habits and speaking style.

    # Task
    - Evaluate the language-related aspects in the ({combined_text}).
    - Check for the use of meaningless words (e.g., "um," "uh") and the presence of repeated words.
    - Provide specific feedback on the interviewee's language habits and speaking style.
    - Ensure that all instances of meaningless words (e.g., "um") are accurately identified.
    - Do not evaluate the content of the response, only focus on language habits and expressions.
    - Clearly state any positive and negative aspects observed.

    # Policy
    - Keep the response within 5 lines, providing a summary of the evaluation.
    - Provide all information and responses only in Korean.
    - Responses must be in JSON format.
    - Place the evaluation in the `speaking` value of the JSON output.
    - Do not include titles or other additional descriptions.
    - Refer to users as '면접자'.

    # Output Format
    {{
        "speaking": ""
    }}
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": "You are an interviewer specializing in evaluating language habits and speaking style."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,  # 5줄 내외로 요약
                temperature=0,
            )
            
            response_content = completion.choices[0].message.content
            result = json.loads(response_content)
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패, 재시도 중... (시도 {attempt + 1}/{max_retries})")
            time.sleep(2)
        
        except Exception as e:
            print(f"평가내용 요약 실패, 재시도 중... (시도 {attempt + 1}/{max_retries})")
            time.sleep(2)

    # 모든 재시도 실패 시 기본 구조 반환
    return {"error": "오류가 발생했습니다."}
