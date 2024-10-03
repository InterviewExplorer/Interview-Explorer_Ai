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

async def evaluate_answer(question: str, answer: str, years: str, job: str, type: str) -> dict:
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