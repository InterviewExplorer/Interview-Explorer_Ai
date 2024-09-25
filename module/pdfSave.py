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

# 기존 문서의 수를 파악하여 새로운 ID를 생성
def get_next_id(index_name):
    response = es.count(index=index_name)
    return response['count']

# Elasticsearch에 문서 추가 (변경 없음)
def index_documents(index_name, resumes, sources):
    next_id = get_next_id(index_name)
    for i, (resume, source) in enumerate(zip(resumes, sources), start=next_id):
        vector = get_vector(resume).tolist()
        doc = {
            'resume': resume,
            'vector': vector,
            'source': source
        }
        es.index(index=index_name, id=i, body=doc)

def main(results, source):
    # split_text 대신 preprocess_data 사용
    preprocessed_contents = preprocess_data(results)

    print(preprocessed_contents)

    # 인덱스 생성
    create_index()

    # 문서 인덱싱
    index_documents(INDEX_NAME, preprocessed_contents, [source] * len(preprocessed_contents))

if __name__ == '__main__':
    main()