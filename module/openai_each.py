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

def generate_assessment(question: str, answer: str, years: str, job: str, type: str) -> dict:
    if type == "technical":
        prompt = f"""
        # Role
        You are a technical interview expert.

        # Task
        Evaluate the answer based on the following criteria:
        - Interviewer's role: {job}
        - Interviewer's experience level: {years} years
        - Interviewer's answer: {answer}
        - Question : {question}

        # Scoring Scale
        A: Excellent
        B: Good
        C: Satisfactory
        D: Poor
        E: Unsatisfactory or No relevant content
        F: No answer or No technical content

        # Instructions
        - Score according to `Scoring Scale`.
        - When explaining the reasons fr the points awarded, focus only on the technical content. The description must be provided in Korean.
        - Do not include any contents related to 'Scoring Scale' or score in the explanation.
        - Provide an ideal answer to the question, considering the interviewee's role and experience, reflecting the reason you explained.
        - The ideal answer must consist only of content that can be verbally expressed. Do not include special characters such as hyphens or colons.
        - Evaluate the answer based on the following five criteria: problem-solving, technical understanding, logical thinking, learning ability, and collaboration/communication. Assign a score between 1 and 100 for each criterion.
        - If a criterion is not present in the answer, assign a null value, and only assign a score if the criterion is included.

        # Policy
        - Ensure the evaluation is detailed and justifiable.
        - Evaluate only the technical aspects of the answer. Do not consider personality, job fit, or organizational fit.
        - The score must be evaluated strictly according to the 'Scoring Scale' and expressed as an alphabetical letter.
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
        You are an expert on personality interviews.

        # Task
        Evaluate the answer based on the following criteria:
        - Interviewer's role: {job}
        - Interviewer's experience level: {years} years
        - Interviewer's answer: {answer}
        - Question : {question}

        # Scoring Scale
        A: Excellent
        B: Good
        C: Satisfactory
        D: Poor
        E: Unsatisfactory or No relevant content
        F: No answer or No technical content

        # Task
        - Score according to `Scoring Scale`.
        - When explaining the reasons for the points awarded, focus only on the personality aspects. The description must be provided in Korean.
        - Do not include any contents related to 'Scoring Scale' or score in the explanation.
        - Provide an ideal answer to the question, considering the interviewee's role and experience, reflecting the reason you explained.
        - The ideal answer must consist only of content that can be verbally expressed. Do not include special characters such as hyphens or colons.
        - Evaluate the answer based on the following five criteria: honesty reliability, interpersonal skills, self motivation passion, adaptability, and self awareness. Assign a score between 1 and 100 for each criterion.
        - If a criterion is not present in the answer, assign a null value, and only assign a score if the criterion is included.

        # Policy
        - Ensure the evaluation is detailed and justifiable.
        - Evaluate only the interpersonal aspects of the answer. Do not consider the technical aspects.
        - The score must be evaluated strictly according to the 'Scoring Scale' and expressed as an alphabetical letter.
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
            "intention": "",
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
    # 평가 생성
    assessmentData = generate_assessment(question, answer, years, job, type)
    print("@@@assessmentData", assessmentData)

    return assessmentData
