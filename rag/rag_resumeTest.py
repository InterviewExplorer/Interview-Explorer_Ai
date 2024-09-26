from datetime import datetime, timedelta

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

# 질문을 벡터로 변환하는 함수
def get_vector(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0].numpy()

# Elasticsearch에서 벡터 기반 검색을 수행하는 함수
def searchDocs_generate(query, index_name, explain=True, profile=True):

    query_vector = get_vector(query).tolist()   # 쿼리를 벡터로 변환    
    must_queries = []                           # 기본 쿼리 구성

    # 질문과 벡터 쿼리 추가
    must_queries.append({
        "bool": {
            "should": [
                {
                    "match": {
                        "resume": {
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
    })

    response = es.search(
        index=index_name,
        body={
            "query": {
                "bool": {
                    "must": must_queries  # 구성된 쿼리 추가
                }
            },
            "size": 10,
            "explain": explain,
            "profile": profile
        }
    )

    # hits = response['hits']['hits']
    # return [hit['_source']['resume'] for hit in hits]

    # 검색된 문서 출력
    hits = response['hits']['hits']
    for i, hit in enumerate(hits):
        print(f"\nDocument {i+1}:")
        print(f"Question: {hit['_source']['resume']}")
        
        # explain 옵션이 True일 경우 value와 description만 출력
        if explain and '_explanation' in hit:
            explanation = hit['_explanation']
            value = explanation.get('value', 'N/A')  # value 값 추출
            description = explanation.get('description', 'N/A')  # description 값 추출
            
            print(f"Explanation Value: {value}")
            print(f"Explanation Description: {description}")

    # 질문을 리스트로 반환
    return [hit['_source']['resume'] for hit in hits]



# 새로운 질문을 생성하는 함수
def resume_test(query: str) -> dict:
    # index_name = 'kookoo'
    index_name = 'test'

    related_docs = searchDocs_generate(query, index_name)
    print("related_docs", related_docs)

