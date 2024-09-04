import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import torch
from elasticsearch import Elasticsearch
from langchain_text_splitters import CharacterTextSplitter

# 설정
ELASTICSEARCH_HOST = 'http://192.168.0.211:9200'
INDEX_NAME = 'newtechnologyquestions'
URL = 'https://corin-e.tistory.com/entry/%EC%8B%A0%EC%9E%85-IT-%EA%B0%9C%EB%B0%9C%EC%9E%90-%EB%A9%B4%EC%A0%91-%EC%A7%88%EB%AC%B8-%EC%B4%9D-%EC%A0%95%EB%A6%AC-%EC%9D%B8%EC%84%B1%ED%9A%8C%EC%82%AC%EC%A7%81%EB%AC%B4%EA%B2%BD%ED%97%98%EA%B8%B0%EC%88%A0'

# Elasticsearch 클라이언트 
es = Elasticsearch([ELASTICSEARCH_HOST])

# 기존 인덱스 삭제
# def delete_index(index_name):
#     if es.indices.exists(index=index_name):
#         es.indices.delete(index=index_name)
#         print(f"인덱스 '{index_name}'가 삭제되었습니다.")
#     else:
#         print(f"인덱스 '{index_name}'가 존재하지 않습니다.")

# 웹사이트에서 텍스트 추출
def fetch_questions(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content_tags = soup.find_all('b')
    return ' '.join([tag.get_text().strip() for tag in content_tags])

# 텍스트 분할
def split_text(text):
    text_splitter = CharacterTextSplitter(
        separator = '',
        chunk_size = 50,
        chunk_overlap = 5,
        length_function = len,
    )
    return text_splitter.split_text(text)

# 텍스트를 벡터로 변환
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_vector(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0].numpy()

# Elasticsearch에 인덱스 생성
def create_index():
    es.indices.create(
        index=INDEX_NAME,
        body={
            "mappings": {
                "properties": {
                    "question": {"type": "text"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": 768  # BERT 모델을 사용할 경우
                    }
                }
            }
        },
        ignore=400  # 이미 존재하는 인덱스일 경우 오류 무시
    )

# 기존 문서의 수를 파악하여 새로운 ID를 생성
def get_next_id(index_name):
    response = es.count(index=index_name)
    return response['count']

# Elasticsearch에 문서 추가
def index_documents(index_name, questions):
    next_id = get_next_id(index_name)
    for i, question in enumerate(questions, start=next_id):
        vector = get_vector(question).tolist()
        doc = {
            'question': question,
            'vector': vector
        }
        es.index(index=index_name, id=i, body=doc)

# 인덱스에서 문서 출력
def print_text_from_index():
    response = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "match_all": {}
            },
            "size": 1000  # 필요한 만큼 문서 수 조절
        }
    )
    
    # 검색 결과에서 텍스트 출력
    if response['hits']['hits']:
        for hit in response['hits']['hits']:
            print(f"ID: {hit['_id']}")
            print(f"Question: {hit['_source'].get('question', 'No question field')}")
            print("-" * 40)
    else:
        print("No documents found.")

# 전체 작업 실행
def main():

    # 인덱스 삭제
    # delete_index('newtechnologyquestions')

    # 웹에서 질문 데이터 추출 및 분할
    content = fetch_questions(URL)
    split_contents = split_text(content)

    # # 인덱스 생성
    create_index()

    # 문서 인덱싱
    index_documents(INDEX_NAME, split_contents)

    print(f"총 {len(split_contents)}개의 청크로 분할되었습니다.")
    for i, chunk in enumerate(split_contents, 1):
        print(f"\n--- 청크 {i} ---")
        print(chunk)

    # 문서 확인
    print_text_from_index()

if __name__ == '__main__':
    main()
