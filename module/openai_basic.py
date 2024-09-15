from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()


api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY가 없습니다.")

gpt_model = os.getenv("gpt")
if gpt_model is None:
    raise ValueError("GPT_Model이 없습니다.")

client = OpenAI(api_key=api_key)

def create_basic_question(job, year, interviewType):
    if interviewType == "technical":
        prompt = f"""
        # Role
        You are the interviewer who creates technical questions.

        # Task
        Create a set of five technical interview questions for a candidate applying for the position of {job}. 
        The candidate has {year} years of experience in this field. 
        Ensure that the questions are challenging and relevant to the {job} role, 
        and cover both fundamental and advanced topics that someone with {year} years of experience should know.
        You must create your question in Korean.
        Each question should assess one of the following areas:
        - The first question should assess "how the problem is solved."
        - The second question should assess "technical literacy."
        - The third question should assess "logical thinking."
        - The fourth question should assess "learning ability."
        - The fifth question should assess "collaboration and communication."

        # Output Format
        You must strictly adhere to the following JSON format:
        {{
            "Q1": "Your first question based on role and experience",
            "Q2": "Your second question based on role and experience",
            "Q3": "Your third question based on role and experience",
            "Q4": "Your fourth question based on role and experience",
            "Q5": "Your fifth question based on role and experience"
        }}
        """
    elif interviewType == "behavioral":
        prompt = f"""
        # Role
        You are the interviewer who creates behavioral questions.

        # Task
        Create a set of five behavioral interview questions for a candidate. 
        Ensure that the questions are challenging and relevant to assess the candidate's personality and soft skills.
        You must create your question in Korean.
        Each question should assess one of the following areas:
        - The first question should assess "honesty (reliability)."
        - The second question should assess "interpersonal skills."
        - The third question should assess "self-motivation (passion)."
        - The fourth question should assess "adaptability."
        - The fifth question should assess "self-awareness."

        # Output Format
        You must strictly adhere to the following JSON format:
        {{
            "Q1": "Your first question based on the specified area",
            "Q2": "Your second question based on the specified area",
            "Q3": "Your third question based on the specified area",
            "Q4": "Your fourth question based on the specified area",
            "Q5": "Your fifth question based on the specified area"
        }}
        """
    else:
        raise ValueError("인터뷰 유형이 잘못되었습니다. 다시 선택해 주세요.")

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
    except Exception as e:
        raise ValueError(f"질문 생성 실패: {e}")
