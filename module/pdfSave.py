import os
from typing import Union, List
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter

# Elasticsearch 설정
ELASTICSEARCH_HOST = os.getenv("elastic")
INDEX_NAME = 'pdf_array'
es = Elasticsearch([ELASTICSEARCH_HOST])

# Sentence Transformer 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')  # 경량화된 모델 사용

# 새로운 전처리 함수
def preprocess_data(data: Union[str, List[str]]) -> List[str]:
    if isinstance(data, list):
        data = '\n'.join(data)
    
    result = []
    
    # 문자열을 줄 단위로 분리
    lines = data.strip().split('\n')
    
    for line in lines:
        # 따옴표 제거 및 공백 정리
        line = line.strip().replace('"', '')
        
        # 키와 값 분리
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # 키-값 쌍을 개별적으로 결과 리스트에 추가
            result.append(f"{key}: {value}")
    
    return result

# 기존의 split_text 함수
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

# 텍스트를 벡터로 변환
def get_vector(text):
    """
    Sentence Transformer를 이용해 텍스트를 벡터화하는 함수
    """
    return model.encode(text)

# Elasticsearch에 인덱스 생성
def create_index():
    es.indices.create(
        index=INDEX_NAME,
        body={
            "mappings": {
                "properties": {
                    "key": {"type": "keyword"},
                    "value": {"type": "text"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": 384  # MiniLM 모델의 벡터 차원
                    },
                    "source": {"type": "keyword"}
                }
            }
        },
        ignore=400
    )

# Elasticsearch에 문서 추가
def get_next_id(index_name):
    try:
        result = es.search(index=index_name, body={"aggs": {"max_id": {"max": {"field": "id"}}}})
        return int(result['aggregations']['max_id']['value']) + 1
    except:
        return 1

# Elasticsearch에 문서 추가
def index_documents(index_name, resumes, source):
    actions = []
    next_id = get_next_id(index_name)
    
    for resume in resumes:
        key, value = resume.split(':', 1)
        key = key.strip()
        value = value.strip()
        vector = get_vector(value).tolist()  # 값만 벡터화
        
        doc = {
            '_index': index_name,
            '_id': next_id,
            '_source': {
                'id': next_id,
                'key': key,
                'value': value,
                'vector': vector,
                'source': source
            }
        }
        actions.append(doc)
        print(f"준비된 문서 {next_id}: {key} - {value}")
        next_id += 1
    
    # 벌크 인덱싱 수행
    success, failed = helpers.bulk(es, actions, stats_only=True)
    print(f"인덱싱 완료: {success}개 성공, {failed}개 실패")

def main(results, source):
    preprocessed_contents = preprocess_data(results)
    
    print("전처리된 결과:")
    for item in preprocessed_contents:
        print(item)
    print("전처리된 항목 수:", len(preprocessed_contents))

    # 인덱스 생성
    create_index()

    # 문서 인덱싱
    index_documents(INDEX_NAME, preprocessed_contents, source)

if __name__ == '__main__':
    # 예시 데이터와 출처 (실제 데이터를 넣어주셔야 합니다)
    main()
