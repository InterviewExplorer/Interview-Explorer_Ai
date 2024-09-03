from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from elasticsearch import Elasticsearch
import json

def generateQ(job, years, pdf_file=None):
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise ValueError("API_KEY가 없습니다.")

    client = OpenAI(api_key=api_key)

    # # Elasticsearch 연결
    # es = Elasticsearch([os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")])
    # INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX", "qa_index")

    # # Elasticsearch에서 관련 질문 검색
    # es_query = {
    #     "query": {
    #         "multi_match": {
    #             "query": f"{job} {years}년차",
    #             "fields": ["question"]
    #         }
    #     },
    #     "size": 10  # 상위 10개 결과만 가져옴
    # }
    # es_result = es.search(index=INDEX_NAME, body=es_query)
    
    # # Elasticsearch 결과에서 질문 추출
    # es_questions = [hit["_source"]["question"] for hit in es_result["hits"]["hits"]]
    # es_questions_str = "\n".join(es_questions)

    resume_content = ""
    if pdf_file:
        loader = PyPDFLoader(pdf_file)
        document = loader.load()
        resume_content = "\n".join([page.page_content for page in document])

    prompt = f"""
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

        ## Task 2: Resume-Based Technical Questions (Only if resume is provided)
        1. If a resume is provided, create 2 additional technical questions based on the resume content.
        2. Focus these questions on specific projects, technologies, or experiences mentioned in the resume.

        # Instructions
        - If a resume is provided, you must generate a total of 4 questions:
            - 2 questions based on the user's role and experience level (Task 1).
            - 2 additional questions based on the resume content (Task 2).
        - If no resume is provided, you should generate only 2 questions based on the user's role and experience level (Task 1).
        - Ensure that questions from Task 1 are not influenced by the resume content.
        - Questions should be relevant, clear, and focused on assessing technical knowledge.
        - Questions should be answerable through verbal explanation.
        - Write your questions in Korean only.
        - Construct questions at a level appropriate for the years of experience provided.
        - Do not ask for code examples.

        # Output Format
        If resume is provided:
        {{
            "Q1": "Your first question based on role and experience",
            "Q2": "Your second question based on role and experience",
            "Q3": "Your third question based on resume content",
            "Q4": "Your fourth question based on resume content"
        }}

        If no resume is provided:
        {{
            "Q1": "Your first question based on role and experience",
            "Q2": "Your second question based on role and experience"
        }}

        Resume content (if provided):
        {resume_content}
        """


    completion = client.chat.completions.create(
        model=os.getenv("gpt"),
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
        default_questions = {
            "Q1": "첫 번째 질문을 생성하는 데 문제가 발생했습니다.",
            "Q2": "두 번째 질문을 생성하는 데 문제가 발생했습니다."
        }
        if resume_content:
            default_questions.update({
                "Q3": "세 번째 질문을 생성하는 데 문제가 발생했습니다.",
                "Q4": "네 번째 질문을 생성하는 데 문제가 발생했습니다."
            })
        return default_questions