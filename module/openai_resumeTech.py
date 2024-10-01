from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
# from elasticsearch import Elasticsearch
import json

def technical_resume(job, years, pdf_file=None, basic_questions=None, max_retries=3):
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise ValueError("API_KEY가 없습니다.")

    client = OpenAI(api_key=api_key)

    resume_content = ""
    if pdf_file:
        loader = PyPDFLoader(pdf_file)
        document = loader.load()
        resume_content = "\n".join([page.page_content for page in document])

    # 기본 질문들을 문자열로 변환
    basic_questions_str = "\n".join([f"{key}: {value}" for key, value in basic_questions.items() if value])

    # 이력서가 없는 경우의 프롬프트
    prompt_without_resume = f"""
    # Role
    You are the interviewer.
    # Task
    Create technical questions based on the following criteria:
    - User experience level: {years} years
    - User role: {job}

    # Instructions
    - Generate 50 unique (non-duplicate) technical questions based on the user's role and experience level.
    - Ensure that questions are relevant, clear, and focused on assessing technical knowledge.
    - Questions should be answerable through verbal explanation.
    - Write your questions in Korean only.
    - Construct questions at a level appropriate for the years of experience provided.
    - Do not ask for code examples.
    - Randomly choose 2 out of 50 questions you created
    - Output only the selected 2 questions.
    - Do not duplicate or closely resemble any of the following basic questions:
    {basic_questions_str}

    # Output Format
    You must **strictly** adhere to the following JSON format:
    {{
        "Q6": "Your sixth question based on role and experience",
        "Q7": "Your seventh question based on role and experience"
    }}
    """

    # 이력서가 있는 경우의 프롬프트
    prompt_with_resume = f"""
    # Role
    You are the interviewer.

    # Task
    Create technical questions based on the following criteria:
    - User experience level: {years} years
    - User role: {job}

    # Instructions
    - Generate 50 unique (non-duplicate) technical questions based on the resume content.
    - Focus these questions on specific projects, technologies, or experiences mentioned in the resume.
    - Select a project from the resume that is related to the user's role ({job}) and create two technical questions based on that project.
    - Randomly choose 2 questions from the 50 you created.
    - If the resume includes experiences that do not match the user's role ({job}), ignore them and choose projects and technologies that align with the given role.
    - Questions should be relevant, clear, and focused on assessing technical knowledge.
    - Questions should be answerable through verbal explanation.
    - Write your questions in Korean only.
    - Construct questions at a level appropriate for the years of experience provided.
    - Do not ask for code examples.
    - Output only the selected 2 questions.
    - Do not duplicate or closely resemble any of the following basic questions:
    {basic_questions_str}

    # Output Format
    You must **strictly** adhere to the following JSON format:
    {{
        "Q6": "Your sixth question based on role and experience",
        "Q7": "Your seventh question based on role and experience"
    }}
    
    Resume content:
    {resume_content}
    """

    def get_questions(prompt):
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=os.getenv("gpt"),
                    messages=[
                        {"role": "system", "content": "You are the interviewer, you are a professional developer. You must always respond in the specified JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5
                )
                response_content = completion.choices[0].message.content
                questions = json.loads(response_content)
                
                # 응답을 Q6와 Q7 형식으로 강제 변환
                formatted_questions = {}
                question_keys = list(questions.keys())
                if len(question_keys) >= 2:
                    formatted_questions["Q1"] = questions[question_keys[0]]
                    formatted_questions["Q2"] = questions[question_keys[1]]
                else:
                    raise ValueError("Not enough questions in the response")
                
                return formatted_questions
            except (json.JSONDecodeError, ValueError, KeyError):
                if attempt < max_retries - 1:
                    prompt += "\n\nYour previous response was not in the correct format. Please ensure you respond with a valid JSON object containing exactly two questions."
                else:
                    return None
        return None

    # 이력서 유무에 따라 적절한 프롬프트 선택
    selected_prompt = prompt_with_resume if pdf_file else prompt_without_resume

    questions = get_questions(selected_prompt)

    if questions is None:
        # If all attempts fail, return a default structure
        default_questions = {
            "Q1": "여섯 번째 질문을 생성하는 데 문제가 발생했습니다.",
            "Q2": "일곱 번째 질문을 생성하는 데 문제가 발생했습니다."
        }
        return default_questions
    
    return questions