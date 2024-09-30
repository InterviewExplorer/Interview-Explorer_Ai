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

def generate_model(question: str, years: str, job: str, type: str) -> dict:
    if type == "technical":
        prompt = f"""
        # Role
        You are a technical interview expert.

        # Task
        - Considering {job} and {years}, generate a ideal answer suitable for {question}.
        
        # Policy
        - The ideal answer must include only content that can be orally described.
        - Provide the answer in Korean.
        - Place the score in the `model` value of the JSON output.
        - Do not include any additional explanations beyond the specified output format.

        # Output Format
        {{
            "model": ""
        }}
        """
    elif type == "behavioral":
        prompt = f"""
        # Role
        You are a behavioral interview expert.

        # Task
        - Generate the intent behind the question {question} for a behavioral interview with {years} of experience as a {job}
        
        # Policy
        - Provide the answer in Korean.
        - Place the intent in the `intention` value of the JSON output.
        - Do not include any additional explanations beyond the specified output format.
        - Refer to users as '면접자'.

        # Output Format
        {{
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
            
            return result
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패, 재시도 중... (시도 {attempt + 1}/{max_retries})")
            time.sleep(2)

    return {"error": "JSONDecodeError"}

    # 1. Check if the answer contains any technical content. If there is no technical content, score it as 0.
    # 2. If the answer contains technical content, check whether it is directly related to the technology mentioned in the question. If the answer is not directly related to the technology in the question, assign a score of 30 or lower, depending on the content of the answer.
    # 3. If the answer is directly related to the skills mentioned in the question, evaluate the level and appropriateness of the answer based on the interviewee's job and experience level, and assign a score between 0 and 100.
    # 4. Assign a higher score the closer the interviewer's response is to the model answer.
    # 5. Explain the reason for the assigned score, focusing only on the technical content. The explanation must be provided in Korean.
    # 6. Do not mention any content related to the model answer in the explanation.
    # 7. Generate the ideal answer for the question, considering the interviewer's role and experience, and reflecting the explained reason.
    # 8. Evaluate the answer based on the following five criteria: problem-solving, technical understanding, logical thinking, learning ability, and collaboration/communication. Assign a score between 1 and 100 for each criterion.
    # 9. If a criterion is not present in the answer, assign a null value, and only assign a score if the criterion is included.

def generate_assessment(question: str, answer: str, years: str, job: str, type: str, result: str) -> dict:
    if type == "technical":
        prompt = f"""
        # Role
        You are a technical interview expert.

        # Task
        Evaluate the answer based on the following criteria:
        - Interviewer's role: {job}
        - Interviewer's experience level: {years} years
        - Interviewer's answer: {answer}
        - Technical interview questions : {question}
        - model answer : {result}

        # Instructions
        - Evaluate the answers to the questions on a scale of 1 to 100, considering the interviewee's job role and level of experience.
        - Assess only the technical aspects.
        - Explain the reasoning behind the score given.
        - Generate the ideal answer for the question, considering the interviewer's role and experience, and reflecting the explained reason.
        - Evaluate the answer based on the following five criteria: problem-solving, technical understanding, logical thinking, learning ability, and collaboration/communication. Assign a score between 1 and 100 for each criterion.
        - If a criterion is not present in the answer, assign a null value, and only assign a score if the criterion is included.

        # Policy
        - Ensure the evaluation is detailed and justifiable.
        - Evaluate only the technical aspects of the answer. Do not consider personality, job fit, or organizational fit.
        - Clearly explain the reasoning behind the assigned score, including how the job role and experience level influenced the evaluation. The explanation must be in Korean.
        - Responses must be in JSON format.
        - Place the score in the `score` value of the JSON output.
        - Place the explanation in the `explanation` value of the JSON output.
        - Place the ideal answer in the `ideal` value of the JSON output.
        - Do not include any additional explanations beyond the specified output format.
        - Refer to users as '면접자'.

        # Output Format
        {{
            "score": "",
            "explanation": "",
            "ideal": "",
            "criteria_scores": {{
                "problem_solving": null,
                "technical_understanding": null,
                "logical_thinking": null,
                "learning_ability": null,
                "collaboration_communication": null
            }}
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
        5. Evaluate the answer based on the following five criteria: honesty (reliability), interpersonal skills, self-motivation (passion), adaptability, and self-awareness. Assign a score between 1 and 100 for each criterion.
        6. If a criterion is not present in the answer, assign a null value, and only assign a score if the criterion is included.

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
        - Refer to users as '면접자'.
        
        # Output Format
        {{
            "score": "",
            "explanation": "",
            "criteria_scores": {{
                "honesty_reliability": null,
                "interpersonal_skills": null,
                "self_motivation_passion": null,
                "adaptability": null,
                "self_awareness": null
            }}
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

def assessment_each(question: str, answer: str, years: str, job: str, type: str) -> dict:
    # 모범 답안 또는 질문의 의도 생성
    modelData = generate_model(question, years, job, type)
    print("@@@modelData", modelData)

    # value값만 추출
    result = ""
    if type == "technical":
        result = modelData.get("model", "")
    elif type == "behavioral":
        result = modelData.get("intention", "")

    # 평가 생성
    assessmentData = generate_assessment(question, answer, years, job, type, result)
    print("@@@assessmentData", assessmentData)

    # 모범 답안 또는 질문의 의도와 평가 데이터 결합
    combined_data = {**assessmentData, **modelData}
    
    return combined_data
