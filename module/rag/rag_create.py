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
INDEX_NAME = 'dj-strawberry'
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

# OpenAI를 사용하여 질문 생성
def generate_question(context):
    prompt = f"""
    # Role
    You are the interviewer.

    # Task
    Create technical questions based on the following criteria:
    - User experience level: 0 years
    - User role: AI Developer
    - Context: {context}

    # Instructions
    - Generate one question to assess the user's interest in new technologies related to their role.
    - Specify the name of a newly released technology.
    - The question should be light in terms of level, focusing on concepts or the degree of interest.
    - Questions should be answerable through verbal explanation.
    - Write your questions in Korean only.
    - Construct questions at a level appropriate for the years of experience provided.
    - Do not ask for code examples.
    
    # Output Format
    You must strictly adhere to the following JSON format:
    {{
        "Question": ""
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
    result = json.loads(response_content)
    
    return result

# 사용자가 입력한 질문으로 검색하고 질문 생성
def main():
    user_input = input("검색할 질문을 입력하세요: ")
    
    # 1. Elasticsearch에서 관련 문서 검색
    related_docs = search_documents(user_input)
    
    if related_docs:
        print(f"관련 문서 {len(related_docs)}개를 찾았습니다.")
        for i, doc in enumerate(related_docs, 1):
            print(f"\n--- 문서 {i} ---")
            print(doc)
        
        # 2. 검색된 문서를 바탕으로 질문 생성
        combined_context = " ".join(related_docs)  # 검색된 문서를 하나의 컨텍스트로 결합
        generated_question = generate_question(combined_context)
        
        print("\n생성된 질문:")
        print(generated_question)
    else:
        print("관련 문서를 찾을 수 없습니다.")

if __name__ == '__main__':
    main()
