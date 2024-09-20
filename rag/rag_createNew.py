import os
from elasticsearch import Elasticsearch
import json
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY가 없습니다.")

gpt_model = os.getenv("gpt")
if gpt_model is None:
    raise ValueError("GPT_Model이 없습니다.")

client = OpenAI(api_key=api_key)

# Elasticsearch 클라이언트 설정
ELASTICSEARCH_HOST = os.getenv("elastic")
es = Elasticsearch([ELASTICSEARCH_HOST])

# Elasticsearch에서 관련 문서 검색
def searchDocs_generate(query, index_name):
    print(f"query: {query}, index_name: {index_name}")
    response = es.search(
        index=index_name,
        body={
            "query": {
                "match": {
                    "question": {
                        "query": query,
                        "fuzziness": "AUTO"
                    }
                }
            },
            "size": 10  # 관련 문서 10개를 가져옴
        }
    )
    hits = response['hits']['hits']
    return [hit['_source']['question'] for hit in hits]

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
        - Generate {num_questions} questions to assess the user's interest in new technologies related to their role.
        - Specify the name of a newly released technology in each question.
        - Mention the field to which the newly released technology belongs.
        - Assume that the interviewee might not be familiar with the new technology and ask questions accordingly.
        - If the question is not about the concept or awareness of the new technology, briefly explain the concept of the new technology before asking the question.
        - The questions should be light in terms of level, focusing on concepts or the degree of interest.
        - Questions should be answerable through verbal explanation.

        # Policy
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
        Create {num_questions} technical questions based on the following criteria:
        - User role: {job}
        - Context: {combined_context}

        # Instructions
        - “We need to create {num_questions} questions to evaluate the user’s personality.”
        - Each question should specify a recent social issue.
        - The interviewer may not be familiar with recent social issues, so adjust your questions accordingly.
        - If a question is not about the social issue, briefly explain the issue before asking the question.
        - Questions should focus on evaluating the interviewer's personality.
        - Questions should be answerable through verbal explanation.
        - When creating questions, you must create personality questions from the perspective of the {job}.

        # Policy
        - Write your questions in Korean only.
        - You must strictly adhere to the following JSON format.
        - Only include the values corresponding to the questions in the output format.
        - Refer to users as '면접자'.
        
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
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are a professional interviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    response_content = completion.choices[0].message.content

    try:
        print("@@@@response_content", response_content)

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

def create_newQ(job: str, type: str) -> dict:
    # type에 따라 INDEX_NAME 변경
    if type == 'technical':
        index_name = 'new_technology'
    elif type == 'behavioral':
        index_name = 'test_rag_behavioral'
    else:
        return {"error": "잘못된 type 값입니다. 'technical' 또는 'behavioral' 중 하나여야 합니다."}

    related_docs = searchDocs_generate(job, index_name)

    if related_docs:
        combined_context = " ".join(related_docs)
        num_questions = 10
        questions = generate_questions(job, type, combined_context, num_questions)

        return questions
    else:
        return {"Questions": ["문서를 찾지 못했습니다."]}
