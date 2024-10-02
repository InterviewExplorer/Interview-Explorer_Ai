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
        - Do not include any contents related to 'Scoring Scale' or score in the explanation.
        - Provide a model answer to the question, considering the interviewee's role and experience. This model answer should demonstrate the correct concept and include some additional correct information.
        - The model answer must consist only of content that can be verbally expressed. Do not include special characters such as hyphens or colons.
        - Evaluate the answer based on the following five criteria: problem-solving, technical understanding, logical thinking, learning ability, and collaboration/communication. Assign a score between 1 and 100 for each criterion.
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
        You are a character interviewer with expertise in conducting interviews.

        # Task
        Evaluate the answer based on the following criteria:
        - Interviewer's job: {job}
        - Interviewer's experience level: {years} years
        - Interviewer's answer: {answer}
        - Question: {question}
        
        ### A-grade Example:
        The incident involving the district chief being acquitted despite charges of inadequate response highlights the importance of accountability in public safety. Public institutions are entrusted with ensuring citizen safety, which involves proactive measures, transparent communication, and clearly defined responsibilities.
        This perspective aligns closely with my values as an AI engineer. Just as public institutions must act transparently and responsibly, I believe that my role as an engineer requires the same standards. When developing AI systems, especially those with safety implications, it’s crucial to ensure transparency in decision-making processes, rigorous testing, and accountability for outcomes. For instance, AI models that predict crowd dynamics during events can help enhance public safety by providing timely alerts.
        Working at this company, I aim to contribute by applying these principles—developing AI technology that not only meets technical standards but also serves a social good, ensuring trust and reliability in our solutions.
        
        ### B-grade Example:
        The responsibility of public institutions is vital for citizen safety, and recent incidents have shown that proactive responses and transparency are essential. Although there may not have been legal liability in this case, public institutions still need to demonstrate ethical accountability.
        Similarly, as an AI engineer, I believe my role requires me to approach development responsibly. By predicting potential risks and ensuring transparency in our AI models, I can contribute to public safety.
        
        ### C-grade Example:
        Public institutions must protect citizens and ensure their safety. Even if no legal wrongdoing is found, the responsibility for public safety should be taken seriously, and improvements should be made when needed.
        As an engineer at this company, I aim to use AI to predict risks and contribute to enhancing public safety.
        
        ### D-grade Example:
        The responsibility of public institutions is important, and their role is to protect citizens. I think better responses could have been made in the incident mentioned.
        At this company, I hope to help improve public safety through technology.
        
        ### E-grade Example:
        Public institutions need to be responsible. The issue seems not to have been handled well.
        I want to contribute by working diligently at the company.
        
        ### F-grade Example:
        I'm not sure about public institutions, but I would like to work at this company.

        # Scoring Scale
        A: The answers are specific, coherent, and logically well-organized, perfectly reflecting the key personality elements of the question. In addition to the main elements, other personality elements are also well reflected, showing the depth and insight of the answers.
        B: Your answers faithfully reflect key elements of your personality and provide logical and relevant explanations. However, it may lack specificity or sufficient reference to other personality factors.
        C: Understand some of the key personality factors in the question and provide relevant basic answers. However, the answers may be somewhat general, lack specificity, or reflect little beyond the main elements.
        D: There is a lack of understanding of some of the key personality elements of the question, or the answer is not logical. Although the main elements are mentioned very briefly, there is little depth or concrete examples, and there is no reflection of other personality elements.
        E: The answers are answered in a way that does not match the intent of the question, rarely reflect relevant aspects of key personality factors, or are very superficial in content.
        F: Included when there is no answer, when you do not know the question, or when the answer is unrelated to the question.

        # Instructions
        - Score strictly according to the 'Scoring Scale' above only.
        - You should base your score on accurate information, and if any part of your answer is incorrect, your score should be adjusted accordingly.
        - Do not arbitrarily consider other elements when scoring, such as specific details, examples, or in-depth explanations.
        - Don't include anything related to your score in your description.
        - Provide a model answer to the question, considering the interviewee's role and experience. This model answer should demonstrate the correct concept and include some additional correct information.
        - Model answers should consist only of what can be expressed orally.
        - Responses are evaluated on a scale of 1 to 100 based on criteria such as “honesty (trust),” “interpersonal relationships,” “adaptability,” “self-motivation (passion),” and “self-awareness.”
        - If the answer contains no criteria, we assign a null value to the "score" value and assign a score only if the answer contains the criteria.
        - The scores for each element in "criteria_scores" must be very strict.
        - If a criterion can be evaluated, assign a score; if it cannot be evaluated, assign a NULL value.
        - Determine the core elements or key personality factors that the question is intended to evaluate before scoring the response.
        - Focus on evaluating how well the response addresses the key personality elements intended by the question, rather than the news content itself. If the answer effectively connects the individual's role and company context, it should be rated positively, regardless of specific references to the news.
        
        # Policy
        - Responses must be provided in Korean only.
        - Technical aspects should never be considered in the interviewer's answers, only personality aspects should be considered.
        - The explanation should explain what kind of personality question the news is intended to ask, rather than just mentioning the news.
        - You must create an explanation for the correct answer, focusing only on personality content.
        - A model answer is created by reflecting the explanation.
        - The 'score' value must be expressed as an alphabetical letter.
        - Responses must be in JSON format.
        - Place the score in the `score` value of the JSON output.
        - Place the explanation in the `explanation` value of the JSON output.
        - Place the model answer in the `model` value of the JSON output.
        - It must not contain any additional description beyond the specified output format.
        - You must evaluate whether each element of "criteria_scores" has been answered correctly as the question was intended.
        - Evaluation should determine whether key elements of the news are identified and answered rather than the question itself.
        
        # Output Format
        {{
            "score": "",
            "explanation": "",
            "model": "",
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