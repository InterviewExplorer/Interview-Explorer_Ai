import time
from datetime import datetime
import requests
import os
from bs4 import BeautifulSoup
from langchain_community.callbacks.uptrain_callback import formatter
from sympy import content
from torch.onnx.symbolic_opset11 import chunk
from transformers import BertTokenizer, BertModel
import torch
from elasticsearch import Elasticsearch
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.ie.service import Service

load_dotenv()

# 설정
ELASTICSEARCH_HOST = os.getenv("elastic")
INDEX_NAME = 'test_rag_behavioral'
URL = 'https://news.naver.com/section/102'

es = Elasticsearch([ELASTICSEARCH_HOST])

# Selenum을 사용하여 웹사이트에서 텍스트 추출
def fetch_article_data(driver, url):
    try:
        driver.get(url)
        # 페이지가 로드될 때까지 대기
        driver.implicitly_wait(10)

        # 웹사이트에서 텍스트 추출
        heading_elements = driver.find_elements(By.CSS_SELECTOR, '#title_area span')
        body_elements = driver.find_elements(By.CSS_SELECTOR, '#newsct_article *')
        timestamp_element = driver.find_element(By.CSS_SELECTOR, '.media_end_head_info_datestamp_time')

        header_text = ' '.join([elem.text.strip() for elem in heading_elements])
        body_text = ' '.join([elem.text.strip() for elem in body_elements])
        timestamp_text = timestamp_element.text.strip() if timestamp_element else "시간 정보 없음"

        # 날짜만 추출하기
        if timestamp_text != "시간 정보 없음":
            date_str = timestamp_text.split('.')[0:3]
            formatted_date = '-'.join(date_str)
        else:
            formatted_date = None

        print(f"URL: {url}")
        print("header_text:", header_text)
        print("body_text:", body_text)
        print("timestamp_text:", timestamp_text)
        print("formatted_date:", formatted_date)

        return header_text + ' ' + (formatted_date if formatted_date else "날짜 없음") + ' ' + body_text, formatted_date

    except NoSuchElementException as e:
        print(f"Error fetching article data from {url}: {e}")
        return "", None
    except TimeoutException as e:
        print(f"Timeout while fetching {url}: {e}")
        return "", None

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
    for i, (question, date_field) in enumerate(questions, start=next_id):
        vector = get_vector(question).tolist()
        doc = {
            'question': question,
            'vector': vector,
            'date_field': date_field
        }
        es.index(index=index_name, id=i, body=doc)

# 메인 페이지에서 모든 뉴스 링크 수집
def collect_all_article_urls(driver):
    driver.get(URL)
    driver.implicitly_wait(10)
    # 페이지 완전 로딩을 위한 추가 대기 시간
    time.sleep(2)

    # 'sa_item_flex _LAZY_LOADING_WRAP' 클래스를 가진 모든 요소 찾기
    # Selenium에서는 여러 클래스를 포함하는 요소를 찾을 때 CSS Selector를 사용
    news_items = driver.find_elements(By.CSS_SELECTOR, ".sa_item_flex._LAZY_LOADING_WRAP")
    article_urls = []

    for item in news_items:
        try:
            link_element = item.find_element(By.TAG_NAME, "a")
            href = link_element.get_attribute("href")
            if href and href.startswith("https://n.news.naver.com"):
                article_urls.append(href)
        except NoSuchElementException as e:
            continue

    # 중복 제거
    unique_urls = list(set(article_urls))
    print(f"수집된 고유한 기사 링크 수 {len(unique_urls)}")
    return unique_urls

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
    # Selenium 옵션 설정
    chrome_options = Options()
    # 브라우저 창을 띄우지 않음
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")

    # WebDriver 설정 (webDriver_manager 사용)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        # 모든 기사 링크 수집
        article_urls = collect_all_article_urls(driver)

        if not article_urls:
            print("수집된 기사 링크가 없습니다.")
            return

        # 인덱스 생성
        create_index()

        # 각 기사 링크를 순회하며 데이터 추출 및 인덱싱
        for idx, url in enumerate(article_urls, 1):
            print(f"\n--- 기사 {idx}/{len(article_urls)} ---")
            content, formatted_date = fetch_article_data(driver, url)
            if not content:
                print(f"데이터를 가져올 수 없습니다. URL: {url}")

            split_contents = split_text(content)
            index_documents(INDEX_NAME, [(chunk, formatted_date) for chunk in split_contents])

            print(f"총 {len(split_contents)}개의 청크로 분할되어 인덱싱 되었습니다.")

            # 서버 부하 방지(잠시 대기)
            time.sleep(1)

        # 디버깅용 문서 확인
        print_text_from_index()

    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        driver.quit()

if __name__ == '__main__':
    main()
