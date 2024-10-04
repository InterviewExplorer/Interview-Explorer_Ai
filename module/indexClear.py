from elasticsearch import Elasticsearch, NotFoundError
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# .env 파일에서 Elasticsearch 호스트 정보 가져오기
ELASTICSEARCH_HOST = os.getenv("elastic")
es = Elasticsearch([ELASTICSEARCH_HOST])
INDEX_NAME = "pdf_array"

def delete_docs():
    try:
        # 인덱스 존재 여부 확인
        if not es.indices.exists(index=INDEX_NAME):
            print(f"인덱스 '{INDEX_NAME}'가 존재하지 않습니다.")
            return

        # 인덱스가 존재하면 문서 삭제 진행
        result = es.delete_by_query(index=INDEX_NAME, body={"query": {"match_all": {}}})
        print(f"{result['deleted']} 개의 문서가 삭제되었습니다.")
    except NotFoundError:
        print(f"인덱스 '{INDEX_NAME}'를 찾을 수 없습니다.")
    except Exception as e:
        print(f"문서 삭제 중 오류 발생: {e}")
