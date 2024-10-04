from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
import json
import re
import os
from langchain_community.document_loaders import PyPDFLoader
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Elasticsearch 호스트 정보 가져오기
ELASTICSEARCH_HOST = os.getenv("elastic")
if ELASTICSEARCH_HOST is None:
    raise ValueError("elastic 환경 변수가 설정되지 않았습니다.")

es = Elasticsearch([ELASTICSEARCH_HOST])
INDEX_NAME="my_korean_index"
api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY가 없습니다.")

def search_all(keyword):
    filtered_list=[]
    doc={
    "query": {
        "match_all": {}
  }
}
    response = es.search(index=INDEX_NAME, body=doc)
    for hit in response['hits']['hits']:
        content = hit['_source'].get('content', 'No content field')
        source = hit['_source'].get('source', 'No source field')
        answer_json=openai_search(keyword,source,content)
        if(answer_json["score"]>50):
                filtered_list.append(answer_json)
    
    sorted_data = sorted(filtered_list, key=lambda x: x['score'], reverse=True)
    # print(sorted_data)
    return sorted_data

def openai_search(keyword,key,value):
        client = OpenAI(
            api_key = api_key
        )

        array_content =  f"""Find a part of your resume that is similar to "{keyword}"
    Score from 0 to 100 based on "{keyword}".  
    Following policies must be strongly reflected:
    Policies
        이 Poicies를 반드시 빠짐없이 지키세요. 하나라도 적용이 되지 않는 policy가 있어서는 안됩니다.
        Do not create content that is not on your resume.
        {keyword}가 "," 로 구분된 2개 이상의 단어라면 각각의 단어에 대해서 아래의 policy를 적용하세요.
        Check your resume thoroughly. Make sure you don't miss any words.
        If the resume contains the word {keyword} or a word that is similar to {keyword} is the case:
            -If there's a word that is exactly same with {keyword}, give the resume a high score.
            -If a word that has a similar meaning to {keyword} appears, give it a high score.
            -이력서 내에서 {keyword} 와 일치하는 단어가 있다면 높은 점수를 부여하세요.
            -{keyword} 와 유사한 의미를 가지고 있는 단어가 있다면 70점 이상을 부여하세요.
            -If the content of resume contains the opposite meaning of {keyword}, score it low.
            -If two or more words separated by "," are given as keywords, give a high score even if only one of several words is included.
            -만약 "," 로 구별된 2개 이상의 단어가 키워드로 주어진다면 앞쪽에 있는 단어일수록 중요 키워드로 판단해서 우선적으로 그 키워드와 관련된 문장을 찾아줘.
            예를 들어 키워드가 "적응력,협력" 이라면 우선적으로 적응력과 가장 유사한 부분을 찾아줘.
            - If a word similar to {keyword} has appeared, give it a high score even if it doesn't contains a specific example.
            - {keyword} 와 유사한 의미의 단어가 이력서에서 등장했다면 키워드에 대한 구체적인 예시가 없어도 높은 점수를 줘.
            - If two or more words separated by "," are given as keywords, give the resume a higher score if multiple keywords are inclued.
            -Consider the context. Consider if the word means something similar to {keyword} in this context.
            -언어가 다르더라도 {keyword}와 같은 대상을 가리키고 있는 단어가 이력서에 있다면 높은 점수를 주시오. 예를 들어 "Java" 와 "자바" 는 같은 단어입니다.
            
          
        이력서 내용: {value}
        # Output Format
            
            important : Never include any Markdown code blocks like ```json```
            You must **strictly** adhere to the following JSON format and Do not include any Markdown code blocks like ```json```:
            {{
                "source": {key},
                "score": 점수,
                "context": Sentence in the part of the resume that represents {keyword}. 
                "reason": context 결과의 이유.
            }}

        """


        # API 요청 보내기
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature= 0,
            top_p=0,
            messages=[
                {"role": "system", "content": "You are an assistant that helps search for keywords in resumes and retrieves related information clearly and concisely. Provide relevant details from the resume that are closely associated with the given keywords."},
                {"role": "user", "content": array_content}
            ],
            
            
        )
        
        answer=response.choices[0].message.content.replace("json","").replace("`","")
        print("answer",answer)
        answer_json=json.loads(answer)
        return answer_json
        
# search_all("java, python")
       
       
    

