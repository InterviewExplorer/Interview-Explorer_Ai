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

def summarize_text(evaluations, type):
    evaluation_items = [f"평가 {i+1}: {evaluation}" for i, evaluation in enumerate(evaluations.values())]
    evaluation_items_text = '\n'.join(evaluation_items)

    if type == "technical":    
        prompt = f"""
        # Role
        You are an expert interviewer evaluating technical responses. The interviewee is the candidate whose responses you are evaluating.

        # Task
        - Review the following technical evaluation content: {evaluation_items_text}
        - Based on the technical evaluation content, write a summary of the technical interview assessment in 5 lines or less.
        - The evaluation should be concise while providing clear and specific reasons.
        - Provide the summary as a comprehensive evaluation, without dividing it into individual assessments.

        # Policy
        - Ensure the assessment is clear and focused on the technical content.
        - Do not include personal opinions or unrelated information.
        - Keep the response within 5 lines, providing a summary of the evaluation.
        - Provide all information and responses only in Korean.
        - The evaluation content must be assigned only within the "" of the output format. Do not include titles or other additional descriptions.
        - The summary should be provided as a single cohesive assessment, without breaking it down into individual evaluations.
    
        # Output Format
        {{
            ""
        }}
        """
    elif type == "behavioral":
        prompt = f"""
        # Role
        You are an expert interviewer specializing in evaluating interpersonal aspects.

        # Task
        - Review the following interpersonal evaluation content: {evaluation_items_text}
        - Based on the interpersonal evaluation content, write a summary of the interpersonal interview assessment in 5 lines or less.
        - The evaluation should be concise while providing clear and specific reasons.
        - Provide the summary as a comprehensive evaluation, without dividing it into individual assessments.
        
        # Policy
        - The evaluation should be clear and focused on interpersonal aspects.
        - Do not include personal opinions or unrelated information.
        - Keep the response within 5 lines, providing a summary of the evaluation.
        - Provide all information and responses only in Korean.
        - The evaluation content must be assigned only within the "" of the output format. Do not include titles or other additional descriptions.
        - The summary should be provided as a single cohesive assessment, without breaking it down into individual evaluations.
    

        # Output Format
        {{
            ""
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
                max_tokens=150,  # 5줄 내외로 요약
                temperature=0,
            )
            
            response_content = completion.choices[0].message.content
            return response_content
        
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패, 재시도 중... (시도 {attempt + 1}/{max_retries})")
            time.sleep(2)
        
        except Exception as e:
            print(f"평가내용 요약 실패, 재시도 중... (시도 {attempt + 1}/{max_retries})")
            time.sleep(2)

    # 모든 재시도 실패 시 기본 구조 반환
    return {"error": "오류가 발생했습니다."}
