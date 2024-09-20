import os
from transformers import BertTokenizer, BertModel
import torch
from elasticsearch import Elasticsearch
from langchain_text_splitters import CharacterTextSplitter
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time
import hashlib

# 설정
ELASTICSEARCH_HOST = os.getenv("elastic")
INDEX_NAME = 'test_koo'

# Elasticsearch 클라이언트
es = Elasticsearch([ELASTICSEARCH_HOST])

# Selenium을 사용하여 데이터 크롤링
def setup_driver():
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    return webdriver.Chrome(service=service, options=options)

def crawl_aitimes():
    driver = setup_driver()
    driver.get("https://www.aitimes.kr/")
    results = []

    try:
        for i in range(1, 11):
            try:
                xpath = f"//div[@class='item']//em[contains(@class, 'number for-middle size-20 line-2x1') and text()='{i}']"
                item = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", item)
                time.sleep(1)
                driver.execute_script("arguments[0].click();", item)

                title = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.XPATH, "//h3[@class='heading']"))
                ).text

                content_elements = WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.XPATH, "//p"))
                )
                content = "\n".join([elem.text for elem in content_elements])

                results.append({
                    'number': i,
                    'title': title,
                    'content': content
                })

                driver.back()
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.XPATH, "//div[@class='item']"))
                )

            except (TimeoutException, NoSuchElementException) as e:
                print(f"Error processing item {i}: {e}")
                continue

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        driver.quit()

    return results

# 텍스트 분할
def split_text(text):
    text_splitter = CharacterTextSplitter(
        separator='',
        chunk_size=250,
        chunk_overlap=5,
        length_function=len,
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
                    },
                    "content_hash": {"type": "keyword"}
                }
            }
        },
        ignore=400  # 이미 존재하는 인덱스일 경우 오류 무시
    )

# 콘텐츠의 해시 값을 생성
def get_content_hash(content):
    return hashlib.md5(content.encode()).hexdigest()

# 중복 확인
def is_duplicate(content_hash):
    result = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "term": {
                    "content_hash": content_hash
                }
            }
        }
    )
    return len(result['hits']['hits']) > 0

# Elasticsearch에 문서 추가 (중복 체크 포함)
def index_documents(index_name, questions):
    for question in questions:
        content_hash = get_content_hash(question)
        if not is_duplicate(content_hash):
            vector = get_vector(question).tolist()
            doc = {
                'question': question,
                'vector': vector,
                'content_hash': content_hash
            }
            es.index(index=index_name, body=doc)
            print(f"Added new content: {question[:50]}...")
        else:
            print(f"Skipped duplicate content: {question[:50]}...")

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

    if response['hits']['hits']:
        for hit in response['hits']['hits']:
            print(f"ID: {hit['_id']}")
            print(f"Question: {hit['_source'].get('question', 'No question field')}")
            print("-" * 40)
    else:
        print("No documents found.")

# 전체 작업 실행
def main():

    # AI Times에서 데이터를 크롤링
    crawled_data = crawl_aitimes()

    # 모든 기사 본문(content)을 하나의 문자열로 결합
    combined_content = "\n".join([item['content'] for item in crawled_data])

    # 텍스트를 분할
    split_contents = split_text(combined_content)
    
    # 인덱스 생성
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
