from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
import json
import re
import os
from langchain_community.document_loaders import PyPDFLoader
from elasticsearch import Elasticsearch
from datetime import datetime

ELASTICSEARCH_HOST="http://192.168.0.49:9200"
es = Elasticsearch([ELASTICSEARCH_HOST])
INDEX_NAME="my_korean_index"
api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY가 없습니다.")




def search_career(career_options):
    career_list=[]
    doc={
    "query": {
        "match_all": {}
  }
}
    response = es.search(index=INDEX_NAME, body=doc)
    for hit in response['hits']['hits']:
        content = hit['_source'].get('content', 'No content field')
        source = hit['_source'].get('source', 'No source field')
        answer_json=openai_search_career(source,content)
        career_list.append(answer_json)
    matched_ids = match_numbers(career_options, career_list)
    return matched_ids    
    
def is_match(value, number):
    if value == "0":
        return number == 0
    elif value == "3년이하":
        return number <= 3
    elif value == "4년이상":
        return number >= 4 and number <= 6
    elif value == "7년이상":
        return number >=7
    return False

# 딕셔너리와 체크박스 값이 일치하는지 확인하는 함수
def match_numbers(selected_values, numbers):
    matched_sources = []
    
    for entry in numbers:
        career = entry["career"]
        # 선택된 값들 중 하나라도 해당 career와 일치하는지 확인
        for value in selected_values:
            if is_match(value, career):
                matched_sources.append(entry["source"])
                break  # 일치하면 더 이상 체크할 필요 없으므로 중단
    
    return matched_sources
    




    


def openai_search_career(key,value):
   


        client = OpenAI(
            api_key = api_key
        )

        array_content =  f"""다음 이력서 내용에서 총 경력을 계산해 주세요:
        이력서 내용: {value}
        총 경력(년수)을 계산해 주시고, 숫자만 반환해 주세요.
        현재 시간 : {datetime.now()} 
        # Output Format
            
            important : Do not include any Markdown code blocks like ```json```
            You must **strictly** adhere to the following JSON format. Do not include any Markdown code blocks like ```json```:
            {{
                "source": {key},
                "career": 경력
                
            }}


        """


        # API 요청 보내기
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature= 0,
            top_p=0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. "},
                {"role": "user", "content": array_content}
            ],
            
            
        )
        
        answer=response.choices[0].message.content
        print("answer",answer)
        answer_json=json.loads(answer)
        return answer_json
        
        
    

# print(search_career(["4년이상"]))
       
       
    

