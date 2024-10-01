import os
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
import json
from openai import OpenAI
from dotenv import load_dotenv
import time
import torch
from transformers import BertTokenizer, BertModel

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

# 질문을 벡터로 변환하는 함수
def get_vector(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0].numpy()

# Elasticsearch에서 벡터 기반 검색을 수행하는 함수
def searchDocs_evaluate(answers: str, index_name: str, type: str, explain=True, profile=True):
    query_vector = get_vector(answers).tolist()
    must_queries = []

    # if type == "behavioral":
    #     must_queries.append({
    #         "range": {
    #             "date_field": {
    #                 "gte": thirty_days_ago_str,
    #                 "lte": today_str
    #             }
    #         }
    #     })

    must_queries.append({
        "bool": {
            "should": [
                {
                    "match": {
                        "question": {
                            "query": answers,
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
    
    print("\n유사성 판단 근거:")
    for i, hit in enumerate(hits):
        print(f"\n문서 {i+1}:")
        print(f"질문: {hit['_source']['question']}")
        print(f"유사도 점수: {hit['_score']:.2f}")
    
        if '_explanation' in hit:
            explanation = hit['_explanation']
            print("유사성 판단 이유:")
            print_human_readable_explanation(explanation)

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

def evaluate_questions(question, answer, years, job, type, combined_context, num_questions):
    if type == "technical":
        prompt = f"""
        # Role
        You are a technical interviewer with expertise in conducting interviews.

        # Task
        Evaluate the answer based on the following criteria:
        - Interviewer's job: {job}
        - Interviewer's experience level: {years} years
        - Interviewer's answer: {answer}
        - Question: {question}

        # Scoring Scale
        A: Correctly includes the concept of the technology mentioned in the question, as well as any additional correct information beyond that concept
        B: Correctly explains only the concept of the technology mentioned in the question
        C: Correctly explains any content about the technology mentioned in the question, even if not directly related to the question
        D: Correctly explains content about the field to which the technology mentioned in the question belongs
        E: Includes any correct technology-related content
        F: No answer, no technical content, or incorrect information

        # Instructions
        - Score strictly according to the 'Scoring Scale' above only.
        - For an 'A' score, the answer must correctly include the concept of the technology mentioned in the question, plus any additional correct information related to that technology. Always give an 'A' score if there's any correct information beyond the basic concept, regardless of its depth or amount.
        - Only assign scores based on correct information. If any part of the answer is incorrect, adjust the score accordingly.
        - Do not consider the depth, specificity, or amount of additional information. Any correct additional information beyond the basic concept is sufficient for an 'A' score.
        - Do not arbitrarily consider other elements when scoring, such as specific details, examples, or in-depth explanations.
        - Do not include any contents related to 'Scoring Scale' or score in the explanation.
        - Provide a model answer to the question, considering the interviewee's role and experience. This model answer should demonstrate the correct concept and include some additional correct information.
        - The model answer must consist only of content that can be verbally expressed. Do not include special characters such as hyphens or colons.
        - Evaluate the answer on a scale of 1 to 100 based on the following criteria: problem-solving, technical understanding, logical thinking, learning ability, and collaboration/communication.
        - If a criterion is not present in the answer, assign a null value, and only assign a score if the criterion is included.

        # Policy
        - Provide the explanation for the answer in Korean, focusing only on the technical content without mentioning the score or scoring criteria.
        - Generate a model answer in Korean, reflecting the content of your explanation.
        - The 'score' value must be expressed as an alphabetical letter.
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
        - You are a personality interviewer.
        
        # Context
        - The interviewee is in the {job} position and is undergoing a personality interview at a related company.
        
        # Task
        - Make sure the interviewee's {answer} includes at least one of the following in relation to the company: self-motivation, self-awareness, honesty, adaptability, and interpersonal skills.
        - Evaluate the given answer {answer} to the question {question} using the following grading scale:
            A: Excellent
            B: Good
            C: Satisfactory
            D: Poor
            E: Unsatisfactory or No relevant content
            F: No answer or no honesty/trustworthiness, interpersonal, self-awareness, or adaptability content.
        
        # Policy
        - The description must not mention scores.
        - The answer to the question reflects the reasons explained, taking into account the interviewee's job {job} and experience, and presents an ideal answer.
        - The ideal answer should consist only of things that can be expressed verbally.
        - Strictly evaluate the answer based on the following five criteria: honesty_reliability, interpersonal skills, self motivation passion, adaptability, and self awareness. Assign a score between 1 and 100 for each criterion.
        - If a criterion is not present in the answer, assign a null value, and only assign a score if the criterion is included.
        - Refer to users as '면접자'.
        - The evaluation must be detailed and include at least one of the following elements: honesty/trustworthiness, self-awareness, interpersonal relationships, self-motivation, and adaptability.
        - Only the personality aspect is evaluated in the answer. Technical aspects should not be considered.
        - The score must be evaluated strictly according to the 'Scoring Scale' and expressed as an alphabetical letter.
        - Responses must be in JSON format.
        - Place the score in the `score` value of the JSON output.
        - Place the explanation in the `explanation` value of the JSON output.
        - Place the intention of the question in the `intention` value of the JSON output.
        - Do not include any additional explanations beyond the specified output format.

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
                model=GPT_MODEL,
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
        index_name = 'rag_behavioral'
    else:
        return {"error": "잘못된 type 값입니다. 'technical' 또는 'behavioral' 중 하나여야 합니다."}

    related_docs = searchDocs_evaluate(question, index_name, type)
    print(related_docs)

    if related_docs:
        combined_context = " ".join(related_docs)
        num_questions = 10
        questions = evaluate_questions(question, answer, years, job, type, combined_context, num_questions)

        return questions
    else:
        return {"Questions": ["문서를 찾지 못했습니다."]}