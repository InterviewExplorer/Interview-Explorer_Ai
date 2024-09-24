import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# 환경 변수 불러오기
load_dotenv()

# 엘라스틱 경로 가져옴
ELASTICSEARCH_HOST = os.getenv("elastic")

# 인덱스 이름 설정
INDEX_NAME = "test_rag_behavioral"

# 크롤링 할 주소 설정
URl = "https://n.news.naver.com/mnews/article/657/0000030135"

# 엘라스틱 클라이언트와 URL을 연결해서 클라이언트 초기화 하기
# hosts는 생략 가능, 여러 호스트로도 연결 가능하기 때문에 배열로 보냄, ex) ([A, B])
es = Elasticsearch([ELASTICSEARCH_HOST])
# es = Elasticsearch(hosts=[ELASTICSEARCH_HOST])

# 웹사이트 텍스트 추출 함수
def fetch_content(url):
    # 본문에서 상태 코드와 콘텐츠를 가져오기
    res = requests.get(url)

    # 첫 번재 인자: 요청으로 가져온 콘텐츠 전달
    # 두 번째 인자: 파싱 타입 선택(html 요소로 접근 가능)
    soup = BeautifulSoup(res.content, "html.parser")

    # CSS 접근 및 리스트 타입 반환
    heading_tags = soup.select("#title_area")
    body_paragraphs = soup.select("span")
    print(body_paragraphs)


    body_text = ' '.join([tag.get_text().strip() for tag in body_paragraphs])

fetch_content(URl)