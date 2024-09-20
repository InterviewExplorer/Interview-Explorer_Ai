import os
from elasticsearch import Elasticsearch
import json
from openai import OpenAI
from dotenv import load_dotenv
import time

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
def searchDocs_evaluate(query, index_name):
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

def evaluate_questions(question, answer, years, job, type, combined_context, num_questions):
    if type == "technical":
        prompt = f"""
        # Role
        You are a technical interviewer with expertise in conducting interviews.

        # Task
        - Evaluate only the interest in new technologies and understanding of current trends.
        - Regarding ({answer}), specific examples and details are excluded from evaluation factors.
        - If the ({answer}) contains no content related to the technology or field mentioned in the question ({question}), assign a score of 0.
        - If the ({answer}) is related to the field of technology mentioned in the question, assign a score between 50 and 100.
        - If the ({answer}) is related to the specific technology mentioned in the question, assign a score between 70 and 100.
        - Clearly explain the reason for the assigned score.
        - Provide an ideal answer to this question, tailored to the candidate's job role and experience level, and written in Korean.
        - The closer the ({answer}) is to the ideal answer, the closer the score should be to 100.
        - Evaluate the answer based on the following five criteria: problem-solving ability, technical understanding, logical thinking, learning ability, and collaboration/communication. Assign a score between 1 and 100 for each criterion.
        - If a criterion is not present in the answer, assign a null value; assign a score only if the criterion is included.

        # Policy
        - Ensure the evaluation is detailed and justifiable.
        - Clearly explain the reasoning behind the assigned score. The explanation must be in Korean.
        - Provide a model answer that reflects the appropriate depth for the job role and experience level, in Korean.
        - Responses must be in JSON format.
        - Place the score in the `score` value of the JSON output.
        - Place the explanation in the `explanation` value of the JSON output.
        - Place the model answer in the `model` value of the JSON output.
        - Do not include any additional explanations beyond the specified output format.
        - Refer to users as '면접자'.

        # Output Format
        {{
            "score": "",
            "explanation": "",
            "model": "",
            "criteria_scores": {{
                "problem_solving": null,
                "technical_understanding": null,
                "logical_thinking": null,
                "learning_ability": null,
                "collaboration_communication": null
            }}
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

        # Policy

        # Output Format
        {{
            "score": "",
            "explanation": "",
            "intention": "",
            "criteria_scores": {{
                "honesty_reliability": null,
                "interpersonal_skills": null,
                "self_motivation_passion": null,
                "adaptability": null,
                "self_awareness": null
            }}
        }}
        """
    else:
        raise ValueError("Invalid type provided. Must be 'technical' or 'behavioral'.")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": "You are a professional interviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            response_content = completion.choices[0].message.content
            result = json.loads(response_content)
            
            # Ensure the evaluation score is a number
            if isinstance(result.get("score"), str) and result["score"].isdigit():
                result["score"] = int(result["score"])
            
            return result
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패, 재시도 중... (시도 {attempt + 1}/{max_retries})")
            time.sleep(2)  # 짧은 대기 후 재시도

    # 모든 재시도 실패 시 기본 구조 반환
    return {"error": "JSONDecodeError"}

def evaluate_newQ(question: str, answer: str, years: str, job: str, type: str) -> dict:
    # type에 따라 INDEX_NAME 변경
    if type == 'technical':
        index_name = 'new_technology'
    elif type == 'behavioral':
        index_name = 'new_personality'
    else:
        return {"error": "잘못된 type 값입니다. 'technical' 또는 'behavioral' 중 하나여야 합니다."}

    related_docs = searchDocs_evaluate(job, index_name)

    if related_docs:
        combined_context = " ".join(related_docs)
        num_questions = 10
        questions = evaluate_questions(question, answer, years, job, type, combined_context, num_questions)

        return questions
    else:
        return {"Questions": ["문서를 찾지 못했습니다."]}
