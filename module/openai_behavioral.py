from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
import json

def generateQ_behavioral(job, years, pdf_file=None, max_retries=3):
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

    # 이력서가 없는 경우의 프롬프트
    prompt_without_resume = f"""
    # Role
    You are a professional interviewer specializing in assessing behavioral aspects.

    # Task
    1. Please create 50 expected questions for the ({job})'s ({years})th year personality interview.
    2. Randomly choose 2 questions from the 50 you created.

    # Policy
    - Focus primarily on assessing the applicant's personality, values, and teamwork skills.
    - Do not duplicate questions that have the same meaning.
    - Generate questions only in Korean.
    - Responses must be in JSON format.
    - Do not include any additional explanations beyond the specified output format.

    # Output Format
    {{
        "Q1": "",
        "Q2": ""
    }}
    """

    # 이력서가 있는 경우의 프롬프트
    prompt_with_resume = f"""
    # Role
    You are a professional interviewer specializing in assessing behavioral aspects.

    # Task
    - There are two main tasks.

    ## Task 1
    1. Please create 50 expected questions for the ({job})'s ({years})th year personality interview.
    2. Randomly choose 2 questions from the 50 you created.

    ## Task 2
    1. Please create 50 expected questions for the ({job})'s ({years})th year personality interview based on the resume content.
    2. Randomly choose 2 questions from the 50 you created.

    # Policy
    - Generate a total of 4 questions:
        - 2 questions from Task 1.
        - 2 additional questions from Task 2.
    - Focus primarily on assessing the applicant's personality, values, and teamwork skills.
    - Do not duplicate questions that have the same meaning.
    - Generate questions only in Korean.
    - Responses must be in JSON format.
    - Do not include any additional explanations beyond the specified output format.

    # Output Format
    {{
        "Q1": "",
        "Q2": "",
        "Q3": "",
        "Q4": ""
    }}
    
    Resume content:
    {resume_content}
    """

    def get_behavioralQ(prompt):
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=os.getenv("gpt"),
                    messages=[
                        {"role": "system", "content": "You are a professional interviewer specializing in assessing behavioral aspects."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5
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

    questions = get_behavioralQ(selected_prompt)

    if questions is None:
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