from datetime import datetime, timedelta

import requests
import os
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import torch
from elasticsearch import Elasticsearch
import random
from dotenv import load_dotenv
import json
from openai import OpenAI

# Load environment variables
load_dotenv()

# 설정
ELASTICSEARCH_HOST = os.getenv("elastic")
API_KEY = os.getenv("API_KEY")
GPT_MODEL = os.getenv("gpt")

if API_KEY is None:
    raise ValueError("API_KEY가 없습니다.")
if GPT_MODEL is None:
    raise ValueError("GPT_Model이 없습니다.")

client = OpenAI(api_key=API_KEY)

# Elasticsearch 클라이언트 설정
es = Elasticsearch([ELASTICSEARCH_HOST])

# BERT 모델 및 토크나이저 불러오기
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 현재 날짜로 부터 30일 전 까지의 날짜 함수
def get_date_range(days: int):
    today = datetime.now()
    start_date = today - timedelta(days=days)
    return today.strftime("%Y-%m-%d"), start_date.strftime("%Y-%m-%d")

# 질문을 벡터로 변환하는 함수
def get_vector(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0].numpy()

# Elasticsearch에서 벡터 기반 검색을 수행하는 함수
def searchDocs_generate(job: str, answers: str, index_name: str, type: str, explain=True, profile=True):
    today_str, thirty_days_ago_str = get_date_range(30)

    combined_query = f"{job} {answers}"
    query_vector = get_vector(combined_query).tolist()
    must_queries = []

    if type == "behavioral":
        must_queries.append({
            "range": {
                "date_field": {
                    "gte": thirty_days_ago_str,
                    "lte": today_str
                }
            }
        })

    must_queries.append({
        "bool": {
            "should": [
                {
                    "match": {
                        "question": {
                            "query": combined_query,
                            "fuzziness": "AUTO"
                        }
                    }
                },
                {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {
                                "query_vector": query_vector
                            }
                        }
                    }
                }
            ]
        }
    })

    response = es.search(
        index=index_name,
        body={
            "query": {
                "bool": {
                    "must": must_queries
                }
            },
            "size": 50,
            "explain": explain,
            "profile": profile
        }
    )

    hits = response['hits']['hits']
    
    # print("\n유사성 판단 근거:")
    # for i, hit in enumerate(hits):
    #     print(f"\n문서 {i+1}:")
    #     print(f"질문: {hit['_source']['question']}")
    #     print(f"유사도 점수: {hit['_score']:.2f}")
    #
    #     if '_explanation' in hit:
    #         explanation = hit['_explanation']
    #         print("유사성 판단 이유:")
    #         print_human_readable_explanation(explanation)

    return [hit['_source']['question'] for hit in hits]

def print_human_readable_explanation(explanation):
    if 'description' in explanation:
        desc = explanation['description'].lower()
        if 'weight' in desc:
            print(f"- 텍스트 매칭 점수: {explanation['value']:.2f}")
        elif 'script score' in desc:
            print(f"- 벡터 유사도 점수: {explanation['value']:.2f}")
        elif 'sum of' in desc:
            print(f"- 총 유사도 점수: {explanation['value']:.2f}")
        elif 'product of' in desc:
            print(f"- 최종 유사도 점수: {explanation['value']:.2f}")
    
    if 'details' in explanation:
        for detail in explanation['details']:
            print_human_readable_explanation(detail)

def generate_questions(job, type, combined_context, num_questions):
    if type == "technical":
        prompt = f"""
        # Role
        You are the interviewer.

        # Task
        Create {num_questions} technical questions based on the following criteria:
        - User role: {job}
        - Context: {combined_context}

        # Instructions
        - Generate questions to assess the level of interest in new technologies related to {job}.
        - The questions should focus on concepts or the degree of interest.
        - Specify the name of a newly released technology in each question.
        - Please ask questions that focus solely on the concept of the technology, and if the interviewee has any information about it, request them to explain.
        - Provide a brief explanation of the presented technology, then ask a derived question.
        - Only ask questions about fields related to {job}.

        # Example
        - Have you come across any technologies or papers recently that you found interesting or enjoyable?
        - How do you think the free availability of MLOps platforms positively impacts the developer community?
        - Have you heard of OpenAI's 'Strawberry' project?
        - Have you heard of the recently announced 'Mistral NeMo'? If you know anything about 'Mistral NeMo,' please explain it.
        - What do you think about the impact of AI model price reductions on developers?

        # Policy
        - Generate {num_questions} unique questions
        - Questions should be answerable through verbal explanation.
        - Write your questions in Korean only.
        - Do not ask for code examples.
        - You must strictly adhere to the following JSON format.
        - Only include the values corresponding to the questions in the output format.
        - Do not include any other text, numbers, or explanations.
        - Refer to users as '면접자'.

        # Output Format
        {{
            "Questions": [
                ""
                ...
            ]
        }}
        """
    elif type == "behavioral":
        prompt = f"""
        # Role
        You are the interviewer.

        # Task
        Create {num_questions} behavioral questions based on the following criteria:
        - Context: {combined_context}

        # Instructions
        - To assess the interviewer's personality and opinions, you must write {num_questions} unique, non-overlapping questions.        
        - Each question should be clearly structured and include detailed background information on recent social issues.
        - Questions should refer to specific news events and clearly state the news source or background.
        - The interviewee may not be familiar with current social issues, so before asking questions, you should Provide a concise but clear explanation of the social issue, including key terms if necessary and include additional explanations of relevant keywords.
        - Questions should focus on assessing how the interviewee perceives the social issue.
        - Questions should encourage the interviewee to express their thoughts through verbal explanations.
        - The difficulty level of the questions should be such that the interviewee can answer even if they do not know much about the news.
            - Consistent with the key themes of the news.
        - When creating questions, you should not mention the interviewee's occupation.
        - Questions should be accessible enough for someone with little knowledge of the topic to provide an informed opinion
        - Each question must be designed to assess one of the following elements, and the relationship to the element must be clearly stated:
            - Honesty
            - Interpersonal skills
            - Self-motivation (passion)
            - Adaptability
            - Self-awareness
        - You should not ask for your own experience.
        - The background explanation must include why the event occurred and any relevant contributing factors, such as systemic issues, policy, or other causes.
        - The entire output (Background Information and Question) must be formatted as a single JSON object, and each field must be written in a single line.
        - The question should be about what they think rather than about improvements or solutions.
        - The last phrase in your question doesn't specify which elements are included.
        - Ensure that each question has a similar level of background detail, including what happened, why it happened, and any broader social implications.

        # Policy
        - The entire JSON object must be formatted in a single line.
        - Write your questions in Korean only.
        - You must strictly adhere to the following JSON format.
        - Only include the values corresponding to the questions in the output format.
        - Refer to users as '면접자'.
         
        # Example
        - Recently, there was an incident in Cheonan where an 8-year-old girl swallowed detergent. The reason nearby hospitals refused treatment in this case was because the hospital did not have a pediatric emergency specialist. As a result, it had to be transported to Daejeon, 80km away. What efforts do you think are needed to solve these problems?

        # Output Format
        {{
            "Questions": [
                ""
                ...
            ]
        }}
        """
    else:
        raise ValueError("Invalid type provided. Must be 'technical' or 'behavioral'.")

    completion = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a professional interviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    response_content = completion.choices[0].message.content
    print("response_content", response_content)

    try:
        result = json.loads(response_content)

        if isinstance(result, dict) and "Questions" in result:
            questions = result["Questions"]

            # questions 필드가 문자열인 경우
            if isinstance(questions, str):
                questions_list = json.loads(questions)
            # questions 필드가 리스트인 경우
            elif isinstance(questions, list):
                questions_list = questions
            else:
                return {"error": "Questions 필드의 형식이 올바르지 않습니다."}

            # 리스트에서 랜덤으로 하나 선택
            if questions_list:
                selected_question = random.choice(questions_list)
                return {"Questions": selected_question}
            else:
                return {"Questions": "질문이 없습니다."}

        return {"error": "Questions 필드가 없거나 예상된 형식이 아닙니다."}

    except json.JSONDecodeError as e:
        return {"error": f"JSON 파싱 오류: {e}"}

# 크롤링 데이터 랜덤으로 가져오기
def get_random_samples(data, sample_size=10):
    samples = random.sample(data, min(sample_size, len(data)))
    print(f"검색 문서 확인: {samples}")
    return samples

# 새로운 질문을 생성하는 함수
def create_newQ(job: str, type: str, answers: str) -> dict:
    # type에 따라 INDEX_NAME 변경
    if type == 'technical':
        index_name = 'new_technology'
    elif type == 'behavioral':
        index_name = 'rag_behavioral'
    else:
        return {"error": "잘못된 type 값입니다. 'technical' 또는 'behavioral' 중 하나여야 합니다."}

    related_docs = searchDocs_generate(job, answers, index_name, type)

    if related_docs:
        random_samples = get_random_samples(related_docs, sample_size=10)
        combined_context = " ".join(random_samples)
        num_questions = 10 if type == "technical" else 5
        # num_questions = 10
        questions = generate_questions(job, type, combined_context, num_questions)

        return questions
    else:
        return {"Questions": ["문서를 찾지 못했습니다."]}
