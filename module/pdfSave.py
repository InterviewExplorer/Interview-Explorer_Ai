import json
import os
from typing import Union, List, Dict
from transformers import BertTokenizer, BertModel
import torch
from elasticsearch import Elasticsearch
from langchain_text_splitters import CharacterTextSplitter

ELASTICSEARCH_HOST = os.getenv("elastic")
INDEX_NAME = 'kookoo'
es = Elasticsearch([ELASTICSEARCH_HOST])

# 새로운 전처리 함수
def preprocess_data(data: Union[str, List[str]]) -> List[str]:
    if isinstance(data, list):
        data = '\n'.join(data)
    
    try:
        json_data = json.loads(data)
    except json.JSONDecodeError:
        json_data = data
    
    result = []
    
    def process_item(key: str, value: Union[str, Dict, List]):
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                process_item(f"{key}.{sub_key}", sub_value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                process_item(f"{key}[{i}]", item)
        else:
            result.append(f"{key}: {value}")
    
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            process_item(key, value)
    else:
        return split_text(data)
    
    return result

# 기존의 split_text 함수 (변경 없음)
def split_text(text):
    if isinstance(text, list):
        text = '\n'.join(text)

    lines = text.split('\n')
    cleaned_lines = [''.join(line.split()) for line in lines]

    text_splitter = CharacterTextSplitter(
        separator='\n',
        length_function=len,
    )

    split_contents = []
    for cleaned_line in cleaned_lines:
        split_contents.extend(text_splitter.split_text(cleaned_line))

    return split_contents

# 텍스트를 벡터로 변환 (변경 없음)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_vector(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0].numpy()

# Elasticsearch에 인덱스 생성 (변경 없음)
def create_index():
    es.indices.create(
        index=INDEX_NAME,
        body={
            "mappings": {
                "properties": {
                    "resume": {"type": "text"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": 768
                    }
                }
            }
        },
        ignore=400
    )

# Elasticsearch에 문서 추가 (변경 없음)
def index_documents(index_name, resumes):
    for i, resume in enumerate(resumes):
        vector = get_vector(resume).tolist()
        doc = {
            'resume': resume,
            'vector': vector
        }
        es.index(index=index_name, id=i, body=doc)

# 인덱스에서 문서 출력 (변경 없음)
def print_text_from_index():
    response = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "match_all": {}
            },
            "size": 1000
        }
    )
    
    if response['hits']['hits']:
        for hit in response['hits']['hits']:
            print(f"ID: {hit['_id']}")
            print(f"resume: {hit['_source'].get('resume', 'No question field')}")
            print("-" * 40)
    else:
        print("No documents found.")

def main(results):
    # split_text 대신 preprocess_data 사용
    preprocessed_contents = preprocess_data(results)
    print(preprocessed_contents)

    # 인덱스 생성
    create_index()

    # 문서 인덱싱
    index_documents(INDEX_NAME, preprocessed_contents)

    print(f"총 {len(preprocessed_contents)}개의 청크로 분할되었습니다.")
    for i, chunk in enumerate(preprocessed_contents, 1):
        print(f"\n--- 청크 {i} ---")
        print(chunk)

    # 문서 확인
    # print_text_from_index()

if __name__ == '__main__':
    main()