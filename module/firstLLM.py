from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
import json

def generateQ(job, years, pdf_file=None):
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise ValueError("API_KEY가 없습니다.")

    client = OpenAI(api_key=api_key)
    
    gpt_model = os.getenv("gpt")
    if gpt_model is None:
        raise ValueError("GPT_Model이 없습니다.")

    resume_content = ""
    if pdf_file:
        loader = PyPDFLoader(pdf_file)
        document = loader.load()
        resume_content = "\n".join([page.page_content for page in document])

    prompt = f"""
    # Role
    You are the interviewer

    # Task
    - You will be provided with user experience level: {years}, user role: {job}, and optionally a resume: {resume_content}.
    - Create technical questions based on the provided information.
    - Create 2 technical questions based on the user's role and experience level.
    - If resume content is provided, create 2 additional technical questions based on the resume content.

    # Policy
    - Create a total of:
        - 2 questions if no resume is provided.
        - 4 questions if resume is provided (2 based on the user's role and experience, and 2 based on the resume content).
    - Do not create any other content beyond the specified number of technical questions.
    - Questions should be relevant, clear, and focused on assessing technical knowledge.
    - Questions should be answerable through verbal explanation.
    - You must write your questions in Korean only.
    - You must construct questions at a level appropriate for the years of experience provided.
    - Don't ask for code examples.
    - When creating questions based on your resume, if you have project experience, mention the project and ask questions about technology within the project.

    # Output Format
    {{
        "Q1": "Your first question here",
        "Q2": "Your second question here"
        "Q3": "Your thrid question here"
        "Q4": "Your fourth question here"
    }}

    Resume content (if provided):
    {resume_content}
    """

    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "당신은 면접관입니다, 당신은 전문적인 개발자입니다"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # Parse the response and ensure it's in the correct format
    try:
        response_content = completion.choices[0].message.content
        questions = json.loads(response_content)
        return questions
    except json.JSONDecodeError:
        # If parsing fails, return a default structure
        return {
            "Q1": "첫 번째 질문을 생성하는 데 문제가 발생했습니다.",
            "Q2": "두 번째 질문을 생성하는 데 문제가 발생했습니다.",
            "Q3": "세 번째 질문을 생성하는 데 문제가 발생했습니다.",
            "Q4": "네 번째 질문을 생성하는 데 문제가 발생했습니다."
        }