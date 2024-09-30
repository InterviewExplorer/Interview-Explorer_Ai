from elasticsearch import Elasticsearch
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from transformers import AutoModel, AutoTokenizer
# Elasticsearch 클라이언트 설정
ELASTICSEARCH_HOST="http://192.168.0.49:9200"
es = Elasticsearch([ELASTICSEARCH_HOST])
model = AutoModel.from_pretrained("klue/bert-base")
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
INDEX_NAME="pdf-test"


def get_vector(text):
    # 입력 텍스트를 토큰화
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    # BERT 모델을 사용해 텍스트 임베딩을 계산
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 마지막 레이어의 출력 값(임베딩)을 추출
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embeddings







def list_resumes(query):
        # 검색어 벡터 생성
    query_text = query
    query_vector = get_vector(query_text).tolist()  # 리스트로 변환

    # Elasticsearch 쿼리 생성
    query_body = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'vector')",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        }
    }

    # Elasticsearch에서 쿼리 실행
    response = es.search(index=INDEX_NAME, body=query_body)
    result=[]
    for hit in response['hits']['hits']:
        content = hit['_source'].get('content', 'No content field')
        source = hit['_source'].get('source', 'No source field')
        score = hit['_score']
        result.append({
        source : score,
    
    })
    score_dict={}
    for index in result:
        for key, value in index.items():
            if key in score_dict:
                score_dict[key] += value
                
            else:
                score_dict[key] = value
                

    sorted_dict = dict(sorted(score_dict.items(), key=lambda item: item[1],reverse=True))     
    # print(sorted_dict)   
    result = [{'source': key, 'score': value} for key, value in sorted_dict.items()]
    return result