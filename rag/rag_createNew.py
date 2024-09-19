import os
import openai
from elasticsearch import Elasticsearch
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY가 없습니다.")

gpt_model = os.getenv("gpt")
if gpt_model is None:
    raise ValueError("GPT_Model이 없습니다.")

client = OpenAI(api_key=api_key)

# Elasticsearch 클라이언트 설정
ELASTICSEARCH_HOST = os.getenv("elastic")
INDEX_NAME = 'new_technology'
es = Elasticsearch([ELASTICSEARCH_HOST])

# Elasticsearch에서 관련 문서 검색
def search_documents(query, index_name=INDEX_NAME):
    response = es.search(
        index=index_name,
        body={
            "query": {
                "match": {
                "question": {
                    "query": query,
                    "fuzziness": "AUTO"
                }
                }
            },
            "size": 10  # 관련 문서 5개를 가져옴
        }
    )
    hits = response['hits']['hits']
    return [hit['_source']['question'] for hit in hits]

def generate_questions(years, job, combined_context, num_questions):
    prompt = f"""
    # Role
    You are the interviewer.

    # Task
    Create {num_questions} technical questions based on the following criteria:
    - User experience level: {years} years
    - User role: {job}
    - Context: {combined_context}

    # Instructions
    - Generate {num_questions} questions to assess the user's interest in new technologies related to their role.
    - Specify the name of a newly released technology in each question.
    - The questions should be light in terms of level, focusing on concepts or the degree of interest.
    - Questions should be answerable through verbal explanation.

    # Policy
    - Write your questions in Korean only.
    - Do not ask for code examples.
    - If you don't know the answer, just say that you don't know.
    - You must strictly adhere to the following JSON format.
    - Only include the values corresponding to the questions in the output format.
    - Do not include any other text, numbers, or explanations.
    - Refer to users as '면접자'.
    
    # Output Format
    {{
        "Questions": [
            ""
            ...
        ]
    }}
    """
    
    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are a professional interviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    response_content = completion.choices[0].message.content
    
    try:
        result = json.loads(response_content)

        if isinstance(result, dict) and "Questions" in result:
            if isinstance(result["Questions"], set):
                result["Questions"] = list(result["Questions"])
        
        return result
    
    except json.JSONDecodeError as e:
        return {"error": f"JSON 파싱 오류: {e}"}

def create_newQ(job: str, years: str) -> dict:
    related_docs = search_documents(job)

    print("job, years : ", job, years)
    if related_docs:
        combined_context = " ".join(related_docs)
        num_questions = 10
        questions = generate_questions(job, years, combined_context, num_questions)

        return questions
    else:
        return {"Questions": ["문서를 찾지 못했습니다."]}

