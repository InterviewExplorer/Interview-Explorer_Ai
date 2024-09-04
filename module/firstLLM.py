from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from elasticsearch import Elasticsearch
import json

def generateQ(job, years, pdf_file=None, max_retries=3):
    load_dotenv()
    api_key = os.getenv("API_KEY2")
    if api_key is None:
        raise ValueError("API_KEY가 없습니다.")

    client = OpenAI(api_key=api_key)

    resume_content = ""
    if pdf_file:
        loader = PyPDFLoader(pdf_file)
        document = loader.load()
        resume_content = "\n".join([page.page_content for page in document])

    # 이력서가 없는 경우의 프롬프트
    prompt_without_resume = f"""
    # Role
    You are the interviewer.

    # Task
    Create technical questions based on the following criteria:
    - User experience level: {years} years
    - User role: {job}

    # Instructions
    - Generate 2 technical questions based on the user's role and experience level.
    - Ensure that questions are relevant, clear, and focused on assessing technical knowledge.
    - Questions should be answerable through verbal explanation.
    - Write your questions in Korean only.
    - Construct questions at a level appropriate for the years of experience provided.
    - Do not ask for code examples.

    # Output Format
    You must strictly adhere to the following JSON format:
    {{
        "Q1": "Your first question based on role and experience",
        "Q2": "Your second question based on role and experience"
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

    There are two main tasks:

    ## Task 1: General Technical Questions
    1. Create 2 technical questions based on the user's role ({job}) and experience level ({years} years).
    2. These questions should be independent of the resume content.

    ## Task 2: Resume-Based Technical Questions
    1. Create 2 additional technical questions based on the resume content.
    2. Focus these questions on specific projects, technologies, or experiences mentioned in the resume.

    # Instructions
    - Generate a total of 4 questions:
        - 2 questions based on the user's role and experience level (Task 1).
        - 2 additional questions based on the resume content (Task 2).
    - Ensure that questions from Task 1 are not influenced by the resume content.
    - Questions should be relevant, clear, and focused on assessing technical knowledge.
    - Questions should be answerable through verbal explanation.
    - Write your questions in Korean only.
    - Construct questions at a level appropriate for the years of experience provided.
    - Do not ask for code examples.

    # Output Format
    You must strictly adhere to the following JSON format:
    {{
        "Q1": "Your first question based on role and experience",
        "Q2": "Your second question based on role and experience",
        "Q3": "Your third question based on resume content",
        "Q4": "Your fourth question based on resume content"
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
                    temperature=0.7
                )
                response_content = completion.choices[0].message.content
                questions = json.loads(response_content)
                return questions
            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    prompt += "\n\nYour previous response was not in the correct JSON format. Please ensure you respond with a valid JSON object containing the questions."
                else:
                    return None
        return None

    # 이력서 유무에 따라 적절한 프롬프트 선택
    selected_prompt = prompt_with_resume if pdf_file else prompt_without_resume

    questions = get_questions(selected_prompt)

    if questions is None:
        # If all attempts fail, return a default structure
        default_questions = {
            "Q1": "첫 번째 질문을 생성하는 데 문제가 발생했습니다.",
            "Q2": "두 번째 질문을 생성하는 데 문제가 발생했습니다."
        }
        if pdf_file:
            default_questions.update({
                "Q3": "세 번째 질문을 생성하는 데 문제가 발생했습니다.",
                "Q4": "네 번째 질문을 생성하는 데 문제가 발생했습니다."
            })
        return default_questions
    
    return questions