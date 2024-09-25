import os
import numpy as np
import torch
from elasticsearch import Elasticsearch
from transformers import BertTokenizer, BertModel

ELASTICSEARCH_HOST = os.getenv("elastic")
INDEX_NAME = 'kookoo'
es = Elasticsearch([ELASTICSEARCH_HOST])

# BERT 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_vector(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0].numpy()

def search_resume_info(query):
    # 쿼리를 벡터로 변환
    query_vector = get_vector(query).tolist()

    # 벡터 검색 쿼리
    knn_query = {
        "knn": {
            "field": "vector",  # 벡터 필드명
            "query_vector": query_vector,
            "k": 10  # 결과의 개수를 제한
        }
    }

    # 키워드 검색 쿼리
    keyword_query = {
        "match": {
            "resume": query  # 텍스트 필드에서 키워드 매칭
        }
    }

    # 두 쿼리를 함께 사용
    response = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "bool": {
                    "should": [
                        knn_query,
                        keyword_query
                    ],
                    "minimum_should_match": 1  # 최소 하나의 조건이 맞으면 결과 반환
                }
            },
            "size": 10  # 검색 결과 수 제한
        }
    )

    if response['hits']['hits']:
        vector_results = []
        keyword_results = []

        for hit in response['hits']['hits']:
            source = hit['_source'].get('source', 'Unknown source')
            resume = hit['_source'].get('resume', '')
            score = hit['_score']

            # 스코어가 1 이하인 결과는 무시
            if score <= 1:
                continue

            # 벡터 검색 결과와 키워드 검색 결과 구분
            if "vector" in hit.get('_source', {}):
                vector_results.append((source, resume, score))
            else:
                keyword_results.append((source, resume, score))

        # 벡터 검색 결과 출력
        if vector_results:
            print("### 벡터 검색 결과 ###")
            for source, resume, score in vector_results:
                print(f"Source: {source} (Score: {score})")
                print(resume)
                print("-" * 40)
        else:
            print("No vector search results found.")

        # 키워드 검색 결과 출력
        if keyword_results:
            print("### 키워드 검색 결과 ###")
            for source, resume, score in keyword_results:
                print(f"Source: {source} (Score: {score})")
                print(resume)
                print("-" * 40)
        else:
            print("No keyword search results found.")
    else:
        print("No documents found.")

if __name__ == "__main__":
    # 검색 실행
    search_terms = [
        "name", "date_of_birth", "project", "personal history"
    ]

    for term in search_terms:
        print(f"Searching for {term}:")
        search_resume_info(term)
        print("\n" + "="*50 + "\n")
