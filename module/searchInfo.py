import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict

# Elasticsearch 설정
ELASTICSEARCH_HOST = os.getenv("elastic")
INDEX_NAME = 'pdf_array'
es = Elasticsearch([ELASTICSEARCH_HOST])

# Sentence Transformer 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')  # 경량화된 모델 사용

def get_vector(query):
    """
    Sentence Transformer를 이용해 쿼리를 벡터화하는 함수
    """
    return model.encode(query)

async def search_resume_info(source_value):
    print(f"Searching for source value: {source_value}")

    keywords = ["name", "date_of_birth", "personality_keywords", "programming_languages", "frontend", "backend", "database", "tools"]
    best_results = {}

    for keyword in keywords:
        query_vector = get_vector(keyword)

        knn_query = {
            "bool": {
                "must": [
                    {
                        "match": {
                            "source": source_value
                        }
                    }
                ],
                "should": [
                    {
                        "knn": {
                            "field": "vector",
                            "query_vector": query_vector.tolist(),
                            "k": 10,  # KNN에서 가장 높은 점수 10개
                            "num_candidates": 100
                        }
                    }
                ]
            }
        }

        response = es.search(
            index=INDEX_NAME,
            body={"query": knn_query, "_source": ["resume", "source"], "size": 10}
        )

        if not response['hits']['hits']:
            print(f"No hits found for keyword: {keyword}")
            continue

        # 가장 점수가 높은 검색 결과
        best_hit = max(response['hits']['hits'], key=lambda hit: hit['_score'])
        print(f"Best hit for keyword '{keyword}': {best_hit}")

        # resume과 source 추출
        resume = best_hit['_source'].get('resume', 'Invalid resume format')
        source = best_hit['_source'].get('source', '')

        # resume에서 각 키워드에 맞는 값을 정규식으로 추출
        extracted_info = extract_info_from_resume(keyword, resume)

        # best_results에 저장
        if extracted_info:
            best_results[keyword] = {
                "source": source,
                **extracted_info
            }

    # 최종 결과 통합
    return integrate_results(best_results)

def extract_info_from_resume(keyword, resume):
    """
    주어진 키워드에 따라 resume 필드에서 적절한 정보를 추출하는 함수
    여러 가능한 키워드 패턴을 검사합니다.
    """
    patterns = {
        "name": [r'"name"\s*:\s*"([^"]+)"'],
        "date_of_birth": [r'"date_of_birth"\s*:\s*"([^"]+)"', r'"birth_date"\s*:\s*"([^"]+)"'],
        "personality_keywords": [r'"personality_keywords"\s*:\s*"([^"]+)"', r'"personality"\s*:\s*"([^"]+)"'],
        "programming_languages": [r'"languages"\s*:\s*"([^"]+)"', r'"programming_languages"\s*:\s*"([^"]+)"', r'"coding_languages"\s*:\s*"([^"]+)"'],
        "frontend": [r'"front_end"\s*:\s*"([^"]+)"', r'"frontend"\s*:\s*"([^"]+)"'],
        "backend": [r'"backend"\s*:\s*"([^"]+)"', r'"back_end"\s*:\s*"([^"]+)"'],
        "database": [r'"dbms"\s*:\s*"([^"]+)"', r'"database"\s*:\s*"([^"]+)"', r'"databases"\s*:\s*"([^"]+)"'],
        "tools": [r'"tools"\s*:\s*"([^"]+)"', r'"development_tools"\s*:\s*"([^"]+)"']
    }

    if keyword in patterns:
        for pattern in patterns[keyword]:
            match = re.search(pattern, resume)
            if match:
                return {keyword: match.group(1)}

    # 해당 키워드에 맞는 정보가 없을 경우
    return None

def integrate_results(results):
    """
    검색 결과를 통합하여 각 소스(PDF)당 하나의 완성된 결과만 반환하는 함수
    """
    integrated = defaultdict(lambda: {
        "source": "",
        "name": "No name provided",
        "date_of_birth": "No date of birth provided",
        "personality_keywords": "No personality keywords provided",
        "programming_languages": "No programming languages provided",
        "frontend": "No frontend skills provided",
        "backend": "No backend skills provided",
        "database": "No database skills provided",
        "tools": "No tools provided"
    })

    for keyword, result in results.items():
        source = result.get("source", "")
        integrated[source]["source"] = source

        # 각 필드가 result에 존재하는지 확인하면서 통합
        for field in integrated[source].keys():
            if field in result and result[field] != f"No {field} provided":
                integrated[source][field] = result[field]
    
    return list(integrated.values())

