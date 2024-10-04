import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("API_KEY")
if API_KEY is None:
    raise ValueError("API_KEY가 없습니다.")

GPT_MODEL = os.getenv("gpt")
if GPT_MODEL is None:
    raise ValueError("GPT_Model이 없습니다.")

client = OpenAI(api_key=API_KEY)

async def evaluate_answer(question: str, answer: str, years: str, job: str) -> dict:
    prompt = f"""
        # Role
        You are a character interviewer with expertise in conducting interviews.

        # Absolute Task
        - Evaluate the answer based on the following criteria:
            - Interviewer's job: {job}
            - Interviewer's experience level: {years} years
            - Interviewer's answer: {answer}
            - Question: {question}
        - You need to evaluate how the interviewee’s personal goals, values, and perspectives align with broader societal or professional objectives.
        
        # Absolute Policy
        - Responses must be in Korean.
        - In your answer, you should only consider the personality aspect and not the technical aspect.
        
        ## Absolute Score Policy
        - The score is placed in the "score" type in the JSON output.
        - The "score" must strictly be one of the grades defined in the "Grade" section below: A, B, C, D, E, F.
        - Do not deviate from the grades provided in the "Grade" section.
        
        # Absolute Grade Policy
        - Assign a score to the response based on the following criteria:
            - If the answer is very specific, logically well-structured, and perfectly reflects the key personality elements required by the question, assign an "A" grade.
            - If the answer faithfully reflects the key personality elements with logical explanations, assign a "B" grade.
            - If the answer addresses some key elements but is general and lacks specificity, assign a "C" grade.
            - If the answer shows a lack of understanding of the key elements or lacks logical coherence, assign a "D" grade.
            - If the answer does not match the intent of the question and is superficial, assign an "E" grade.
            - If the answer is missing or completely unrelated to the question, assign an "F" grade.
            
        ## Never Explanation Policy
        - Descriptions must not mention discussions.
                
        ## Absolute Explanation Policy
        - The description is placed in the "explanation" type of the JSON output.
        - Descriptions must not mention grades and scores.
        - The explanation should emphasize the personality elements required by the question rather than providing an explanation of the question itself.
        - The explanation must not include any references to the news content.
        - When writing your description, you should first clearly state what personality factors are being assessed in the question.
        - Your description should not mention the interviewee.

        ## Absolute Model Policy
        - Model answers are placed in the “model” type of the JSON output.
        - The response must be crafted in a way that ensures it receives a grade of A when the model answer is used again as a response.
        - The model answer should focus on the core personality elements required by the question, rather than including all possible related elements.
        - A model answer should demonstrate how the interviewee's personal goals or values align with broader societal or professional objectives.        - Use feedback from the "explanation" type value to improve the model answer, ensuring that any identified shortcomings or negative aspects are addressed effectively.
        - Refer to the "A" grade conditions in the # Absolute Grade Policy section when crafting the model answer to ensure it meets all the requirements for an A grade.
        - The model answer must not mention any company-specific goals or directly reference the company.
        
        ## Absolute Criteria Scores Policy
        - In the "score scale" of JSON output, the value of each element type must be an integer between 1 and 100.
        
        ### Absolute Honesty and Reliability Policy
        - You need to evaluate how honest and trustworthy the answers are.
        - You should evaluate whether the answers honestly represent the interviewee's experience without exaggeration or inaccuracy.
        
        ### Absolute Interpersonal Skills Policy
        - In your answer you should assess how good your interpersonal skills are.
        - Scores must be assigned more strictly, reflecting the quality and depth of the answer in line with the expected standard.
        - A response that lacks detail, depth, or fails to demonstrate any relevant personality traits should receive a significantly lower score, potentially below 30.
        - The criteria scores must align with the overall grade assigned, ensuring consistency between individual element ratings and the final score.
        
        ### Absolute Self-Motivation and Passion Policy
        - We need to evaluate how well the respondent motivates himself and communicates his passion.
        
        ### Absolute Adaptability Policy
        - Assess your response to change and your ability to adapt to new environments.
        
        ### Absolute Self-Awareness Policy
        - Assess whether you are clear about your strengths and weaknesses.
        
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

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Please evaluate the interviewee’s answer according to the above guidelines."}
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