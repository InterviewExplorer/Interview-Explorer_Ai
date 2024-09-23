import requests
import os
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import torch
from elasticsearch import Elasticsearch
import random
from dotenv import load_dotenv
import json
from openai import OpenAI

# Load environment variables
load_dotenv()

# 설정
ELASTICSEARCH_HOST = os.getenv("elastic")
API_KEY = os.getenv("API_KEY")
GPT_MODEL = os.getenv("gpt")

if API_KEY is None:
    raise ValueError("API_KEY가 없습니다.")
if GPT_MODEL is None:
    raise ValueError("GPT_Model이 없습니다.")

client = OpenAI(api_key=API_KEY)

# Elasticsearch 클라이언트 설정
es = Elasticsearch([ELASTICSEARCH_HOST])

# BERT 모델 및 토크나이저 불러오기
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_vector(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0].numpy()

def searchDocs_generate(query, index_name):
    query_vector = get_vector(query).tolist()
    
    # 하이브리드 검색을 위한 쿼리
    response = es.search(
        index=index_name,
        body={
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "question": {
                                    "query": query,
                                    "fuzziness": "AUTO"
                                }
                            }
                        },
                        {
                            "script_score": {
                                "query": {
                                    "match_all": {}
                                },
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                                    "params": {
                                        "query_vector": query_vector
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            "size": 10  # 관련 문서 10개를 가져옴
        }
    )
    
    hits = response['hits']['hits']
    return [hit['_source']['question'] for hit in hits]

def generate_questions(job, type, combined_context, num_questions):
    if type == "technical":
        prompt = f"""
        # Role
        You are the interviewer.

        # Task
        Create {num_questions} technical questions based on the following criteria:
        - User role: {job}
        - Context: {combined_context}

        # Instructions
        - Generate {num_questions} unique questions to assess the user's interest in new technologies related to {job}.
        - Generate light-level questions about technology related to the {job}.
        - The questions should focus on concepts or the degree of interest.
        - Specify the name of a newly released technology in each question.
        - Only ask questions related to developers or the IT field. Do not ask questions about other fields such as art creation, life sciences, etc.

        # Example
        - How do you think the free availability of MLOps platforms positively impacts the developer community?
        - Have you heard of OpenAI's 'Strawberry' project? How do you think this project could contribute to the advancement of AI?
        - Have you heard of the recently announced 'Mistral NeMo'? What benefits could this technology offer to developers?
        - What do you think about the impact of AI model price reductions on developers?

        # Policy
        - Questions should be answerable through verbal explanation.
        - Write your questions in Korean only.
        - Do not ask for code examples.
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
    elif type == "behavioral":
        prompt = f"""
        # Role
        You are the interviewer.

        # Task
        Create {num_questions} behavioral questions based on the following criteria:
        - User role: {job}
        - Context: {combined_context}

        # Instructions

        # Policy

        # Output Format
        {{
            "Questions": [
                ""
                ...
            ]
        }}
        """
    else:
        raise ValueError("Invalid type provided. Must be 'technical' or 'behavioral'.")

    completion = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a professional interviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    response_content = completion.choices[0].message.content
    print("response_content", response_content)

    try:
        result = json.loads(response_content)

        if isinstance(result, dict) and "Questions" in result:
            questions = result["Questions"]

            # questions 필드가 문자열인 경우
            if isinstance(questions, str):
                questions_list = json.loads(questions)
            # questions 필드가 리스트인 경우
            elif isinstance(questions, list):
                questions_list = questions
            else:
                return {"error": "Questions 필드의 형식이 올바르지 않습니다."}

            # 리스트에서 랜덤으로 하나 선택
            if questions_list:
                selected_question = random.choice(questions_list)
                return {"Questions": selected_question}
            else:
                return {"Questions": "질문이 없습니다."}

        return {"error": "Questions 필드가 없거나 예상된 형식이 아닙니다."}

    except json.JSONDecodeError as e:
        return {"error": f"JSON 파싱 오류: {e}"}

# 새로운 질문을 생성하는 함수
def create_newQ(job: str, type: str) -> dict:
    # type에 따라 INDEX_NAME 변경
    if type == 'technical':
        index_name = 'new_technology'
    elif type == 'behavioral':
        index_name = 'new_personality'
    else:
        return {"error": "잘못된 type 값입니다. 'technical' 또는 'behavioral' 중 하나여야 합니다."}

    related_docs = searchDocs_generate(job, index_name)
    print("related_docs", related_docs)

    if related_docs:
        combined_context = " ".join(related_docs)
        num_questions = 10
        questions = generate_questions(job, type, combined_context, num_questions)

        return questions
    else:
        return {"Questions": ["문서를 찾지 못했습니다."]}
