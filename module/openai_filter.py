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
INDEX_NAME="pdf_array"
api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY가 없습니다.")

work_list=[]



    
def is_match(value, number):
    if value == "신입":
        return number == 0
    elif value == "1~3년":
        return number <= 36
    elif value == "3~5년":
        return number >= 36 and number <= 60
    elif value == "5~7년":
        return number >= 60 and number <= 84
    elif value == "7~10년":
        return number >= 84 and number <= 120
    elif value == "10년이상":
        return number >= 120
    return False

# 딕셔너리와 체크박스 값이 일치하는지 확인하는 함수
def match_numbers(selected_values, work_list):
    matched_sources = []
    
    for entry in work_list:
        career = entry["career"]
        career_parsed=parse_time(career)
        # 선택된 값들 중 하나라도 해당 career와 일치하는지 확인
        for value in selected_values:
            if is_match(value, career_parsed):
                matched_sources.append({
                    "source" : entry["source"],
                    "career" : career_parsed
                })
                
                
                break  # 일치하면 더 이상 체크할 필요 없으므로 중단
    
    return matched_sources
    


def get_work_experience(career_options):


    query = {
    "query": {
        "match": {
        "key": "work_experience"
        }
    }
    }


    # 검색 실행
    response = es.search(index=INDEX_NAME, body=query)

    # 결과 출력
    for hit in response['hits']['hits']:
        source = hit['_source'].get('source', 'No source field')
        key = hit['_source'].get('key', 'No source field')
        value = hit['_source'].get('value', 'No source field')
        # print(f"source: {source}")
        # print(f"key: {key}")
        # print(f"value: {value}")
        # print('---')
        work_list.append({
            "source" : source,
            "career" : value
        })
        
    matched_ids = match_numbers(career_options, work_list)
    return matched_ids

def parse_time(time_str):
        total_months = 0
        parts = time_str.split()
        for part in parts:
            if '년' in part:
                years = int(part.replace('년', ''))
                total_months += years * 12
            elif '개월' in part:
                months = int(part.replace('개월', ''))
                total_months += months
        return total_months
        
    

       
       
    

