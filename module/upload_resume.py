import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
# .env 파일 로드
load_dotenv()

# .env 파일에서 Elasticsearch 호스트 정보 가져오기
ELASTICSEARCH_HOST = os.getenv('elastic')

# Elasticsearch 클라이언트 초기화
es = Elasticsearch([ELASTICSEARCH_HOST])
INDEX_NAME="pdf-test"
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("klue/bert-base")
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")


# 모든 인덱스 이름 출력
def print_indice():
    indices = es.cat.indices(format='json')
    for index in indices:
        print(index['index'])

def split_text_into_sentences(text):
    # 문장 단위로 텍스트를 분리합니다
    sentences = sent_tokenize(text)
    return sentences


def create_index(index_name):
    es.indices.create(
        index=index_name,
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

def delete_index(index_name):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"인덱스 '{index_name}'가 삭제되었습니다.")
    else:
        print(f"인덱스 '{index_name}'가 존재하지 않습니다.")    

def read_pdf(resume):
    resume_content = ""
    loader = PyPDFLoader(resume)
    document = loader.load()
    resume_content = "\n".join([page.page_content for page in document])
    return resume_content


def count_docs(index_name):
    response = es.count(index=index_name)
    print(f"문서 수: {response['count']}")

def read_pdf(resume):
    resume_content = ""
    loader = PyPDFLoader(resume)
    document = loader.load()
    resume_content = "\n".join([page.page_content for page in document])
    return resume_content



def search_by_topic(topic):
    query = {
        "query": {
            "match": {
                "content": topic
            }
        }
    }
    response = es.search(index=INDEX_NAME, body=query)
    return response['hits']['hits']



def split_text(text):
    text_splitter = CharacterTextSplitter(
        separator = '',
        chunk_size = 50,
        chunk_overlap = 5,
        length_function = len,
    )
    return text_splitter.split_text(text)

def get_vector(text):
    # 입력 텍스트를 토큰화
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    # BERT 모델을 사용해 텍스트 임베딩을 계산
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 마지막 레이어의 출력 값(임베딩)을 추출
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    return embeddings
def get_next_id(index_name):
    response = es.count(index=index_name)
    return response['count']

# Elasticsearch에 문서 추가
def index_documents(index_name, text, resume_name):
    next_id = get_next_id(index_name)
    
    for i, content in enumerate(text, start=next_id):
        vector = get_vector(content).tolist()
        doc = {
            'content': content,
            'vector': vector,
            'source' : resume_name
        }
        es.index(index=index_name, id=i, body=doc)
def get_document_by_id(index_name, doc_id):
    try:
        response = es.get(index=index_name, id=doc_id)
        return response['_source']  # 문서의 내용을 반환
    except Exception as e:
        print(f"문서 가져오기 실패: {e}")
        return None
def upload_resumes(resume_name,resume_content):
    sentences=split_text_into_sentences(resume_content)
    index_documents(INDEX_NAME, sentences, resume_name)

def reset_resumes():
    es.delete_by_query(index=INDEX_NAME, body={"query": {"match_all": {}}}) 


def main():
    # pdf_name = "초록색 깔끔한 경력 이직 이력서 자기소개서.pdf"
    # text=read_pdf(pdf_name)
    
    sentences=split_text_into_sentences("""저는 대학 시절부터 다양한 경험을 통해 여러 방면에서 스스로를 발전시켜 왔습니다. 학문적인 성과뿐만 아니라, 동아리 활동과 대외활동에서도 꾸준히 도전하며 새로운 기회를 만들어 왔습니다.

저는 책임감을 바탕으로 주어진 일에 최선을 다하고, 팀과의 협력을 중요시합니다. 대학교에서 팀 프로젝트를 진행할 때, 맡은 바를 철저히 이행하며 팀원들과 원활한 소통을 통해 좋은 결과를 이끌어냈습니다. 이러한 경험들은 문제 해결과정에서도 창의적인 접근을 시도할 수 있게 해주었습니다.

이처럼 다양한 상황에서 성과를 만들어 왔으며, 앞으로도 이러한 역량을 바탕으로 더 큰 도전을 이어나가고자 합니다. 제 목표는 단순한 업무 수행을 넘어서, 조직과 함께 성장하는 것입니다.""")
    
    index_documents(INDEX_NAME, sentences,"test3")
    # # print(get_document_by_id(INDEX_NAME,2))


    es.delete_by_query(index=INDEX_NAME, body={"query": {"match_all": {}}}) 
    
    
    count_docs(INDEX_NAME)
    # print_indice()
    

    

