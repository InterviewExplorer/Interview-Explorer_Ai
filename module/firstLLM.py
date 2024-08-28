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

    resume_content = ""
    if pdf_file:
        loader = PyPDFLoader(pdf_file)
        document = loader.load()
        resume_content = "\n".join([page.page_content for page in document])

    prompt = f"""
    # Role
    You are the interviewer

    # Task
    - You will be provided with user experience level: {years}, user role: {job}, and optionally a resume.
    - Create 2 technical questions based on the provided information.
    - If a resume is provided, prioritize creating questions based on the resume content.
    - If no resume is provided, create questions based on the user's role and experience level.

    # Policy
    - Create exactly 2 technical questions.
    - Do not create any other content beyond the two technical questions.
    - Just ask questions that can be explained in words.
    - Questions should be relevant, clear, and focused on assessing technical knowledge.

    # Output Format
    {{
        "first_question": "Your first question here",
        "second_question": "Your second question here"
    }}

    Resume content (if provided):
    {resume_content}
    """

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "당신은 면접관입니다, 당신은 전문적인 개발자입니다"},
            {"role": "user", "content": prompt}
        ]
    )

    # Parse the response and ensure it's in the correct format
    try:
        response_content = completion.choices[0].message.content
        questions = json.loads(response_content)
        return questions
    except json.JSONDecodeError:
        # If parsing fails, return a default structure
        return {
            "first_question": "첫 번째 질문을 생성하는 데 문제가 발생했습니다.",
            "second_question": "두 번째 질문을 생성하는 데 문제가 발생했습니다."
        }