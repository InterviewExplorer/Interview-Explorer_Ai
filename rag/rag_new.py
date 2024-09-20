import requests
import os
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import torch
from elasticsearch import Elasticsearch
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# 설정
ELASTICSEARCH_HOST = os.getenv("elastic")
INDEX_NAME = 'test_rag_behavioral'
# URL = 'https://www.aitimes.com/news/articleView.html?idxno=163446'
URL = 'https://n.news.naver.com/mnews/article/005/0001726018'

es = Elasticsearch([ELASTICSEARCH_HOST])

# 웹사이트에서 텍스트 추출
def fetch_questions(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    heading_tags = soup.select('#title_area span')
    body_paragraphs = soup.select('#newsct_article *')

    header_text = ' '.join([tag.get_text().strip() for tag in heading_tags])
    body_text = ' '.join([tag.get_text().strip() for tag in body_paragraphs])
    
    return header_text + ' ' + body_text
    # return header_text

# 텍스트 분할
def split_text(text):
    text_splitter = CharacterTextSplitter(
        separator = '',
        chunk_size = 180,
        chunk_overlap = 10,
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
                        "dims": 768
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

def main():
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
