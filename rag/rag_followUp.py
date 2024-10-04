from openai import OpenAI
import os
from dotenv import load_dotenv
import time
from elasticsearch import Elasticsearch
from transformers import BertTokenizer, BertModel
import torch
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import json

# .env 파일에서 환경 변수 로드
load_dotenv()

ELASTICSEARCH_HOST = os.getenv("elastic")
es = Elasticsearch([ELASTICSEARCH_HOST])

# API 키 가져오기
api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY가 없습니다.")

# GPT 모델 가져오기
gpt_model = os.getenv("gpt")
if gpt_model is None:
    raise ValueError("GPT_Model이 없습니다.")

# OpenAI 클라이언트 초기화 및 api키 등록
client = OpenAI(api_key=api_key)

# BERT 모델과 토크나이저 초기화
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings.flatten().tolist()

def get_date_range(days: int):
    today = datetime.now()
    start_date = today - timedelta(days=days)
    return today.strftime("%Y-%m-%d"), start_date.strftime("%Y-%m-%d")

def text_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def remove_duplicates(questions, similarity_threshold=0.8):
    unique_questions = []
    for question in questions:
        if not any(text_similarity(question, uq) > similarity_threshold for uq in unique_questions):
            unique_questions.append(question)
    return unique_questions

def ragFollwUp(job: str, type: str, questionsRag: str, answerRag: str, explain=True, profile=True):
    # type에 따른 인덱스 선택
    if type == 'technical':
        index_name = 'new_technology'
    elif type == 'behavioral':
        index_name = 'test_rag_behavioral'
    else:
        return {"error": "잘못된 type 값입니다. 'technical' 또는 'behavioral' 중 하나여야 합니다."}

    today_str, thirty_days_ago_str = get_date_range(30)

    combined_query = f"{questionsRag}"
    query_vector = get_bert_embedding(combined_query)
    must_queries = []

    if type == "behavioral":
        must_queries.append({
            "range": {
                "date_field": {
                    "gte": thirty_days_ago_str,
                    "lte": today_str
                }
            }
        })

    must_queries.append({
        "bool": {
            "should": [
                {
                    "match": {
                        "question": {
                            "query": combined_query,
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
    })

    try:
        response = es.search(
            index=index_name,
            body={
                "query": {
                    "bool": {
                        "must": must_queries
                    }
                },
                "size": 1,  # 가장 높은 점수의 결과 하나만 가져옵니다.
                "explain": explain,
                "profile": profile
            }
        )
    except Exception as e:
        print(f"Elasticsearch 검색 중 오류 발생: {str(e)}")
        return {"error": "검색 중 오류가 발생했습니다."}

    hits = response['hits']['hits']
    if not hits:
        return {"error": "No search results found."}

    top_hit = hits[0]
    original_content = top_hit['_source'].get('original', '')

    print(f"\nTop Search Result for {type} interview:")
    print(f"ID: {top_hit['_id']}, Score: {top_hit['_score']}")
    print(f"Original Content: {original_content[:100]}...")

    if type == "technical":
        prompt = f"""
        # Role
        You are a technical interviewer.

        # Context
        Original question: {questionsRag}
        Candidate's answer: {answerRag}
        Related technical information: {original_content}

        # Task
        Generate 3 technical follow-up questions based on the given context.

        # Instructions
        - Identify areas in the candidate's answer that could be explored further or where additional technical knowledge could be demonstrated.
        - Utilize the related technical information to ask about additional knowledge or opinions.
        - Each question should start by referencing the candidate's previous answer.
        - Questions should be open-ended and not answerable with a simple 'yes' or 'no'.
        - Ensure questions are progressively more challenging or explore different aspects of the topic.
        - Questions are only asked in Korean.
        - Just ask one question.

        # Output Format
        {{
            "Question": [
                "First follow-up question"
            ]
        }}
        """
    elif type == "behavioral":
        prompt = f"""
        # Role
        You are a behavioral interviewer.

        # Context
        Original question: {questionsRag}
        Candidate's answer: {answerRag}
        Related information: {original_content}

        # Task
        Generate 3 behavioral follow-up questions based on the given context.

        # Instructions
        - Identify experiences or opinions in the candidate's answer that could be explored more deeply.
        - Use the related information to present additional situations or scenarios and ask for the candidate's perspective.
        - Each question should start by referencing the candidate's previous answer.
        - Questions should assess the candidate's values, problem-solving abilities, interpersonal skills, etc.
        - Ensure questions are progressively more challenging or explore different aspects of the candidate's behavior.
        - Questions are only asked in Korean.
        - Just ask one question.

        # Output Format
        {{
            "Question": ""
        }}
        """
    else:
        return {"error": "Invalid type value. It should be either 'technical' or 'behavioral'."}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": "You are an expert in interviewing and question generation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            response_content = completion.choices[0].message.content
            result = json.loads(response_content)
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed, retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(2)  # Short wait before retrying

    # Return default structure if all retries fail
    return {"error": "JSONDecodeError"}