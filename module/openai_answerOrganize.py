from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import time

# Load environment variables from .env file
load_dotenv()

# Get API key
api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY is missing.")

# Get GPT model
gpt_model = os.getenv("gpt")
if gpt_model is None:
    raise ValueError("GPT_Model is missing.")

# Initialize OpenAI client and register API key
client = OpenAI(api_key=api_key)

def answerOraganize(answers: str, questions: str, job: str, type: str) -> dict:
    if type == "technical":
        prompt = f"""
        # Role
        You are a technical expert and interviewer in the field of {job}.

        # Task
        - Compare {questions} with their corresponding {answers}.
        - Identify one answer that is vague or insufficient.
        - Generate a follow-up question in Korean for the identified answer, focusing on technical depth.

        # Policy
        - Carefully analyze the answers to find the most ambiguous or incomplete response.
        - Create only one follow-up question for the selected answer.
        - The follow-up question must be in Korean.
        - The follow-up question should aim to explore related technologies, models, or methodologies.
        - Instead of asking for specific examples, create questions that verify understanding of related technical concepts.
        - When asking the follow-up question, reference the original question and answer. For example, start with a phrase like "Regarding [original question], you answered [brief summary of answer]. ..."
        - Do NOT generate a follow-up question if the answer expresses uncertainty (e.g., "I don't know", "I'm not sure") or is a negative response.
        - Strictly adhere to the following JSON format.
        - Only include the values corresponding to the value in the output format.
        - Questions are only asked in Korean.

        # Output Format
        {{
            "Question": ""
        }}
        """
    elif type == "behavioral":
        prompt = f"""
        # Role
        You are a behavioral interview expert in the field of {job}.

        # Task
        - Compare {questions} with their corresponding {answers}.
        - Identify one answer that is vague or insufficient.
        - Generate a follow-up question in Korean for the identified answer, focusing on exploring the candidate's experience and competencies.

        # Policy
        - Carefully analyze the answers to find the most ambiguous or incomplete response.
        - Create only one follow-up question for the selected answer.
        - The follow-up question must be in Korean.
        - The follow-up question should aim to explore the candidate's behavior, decision-making process, or problem-solving abilities more deeply.
        - Instead of asking for specific examples, create questions that inquire about the candidate's approach to situations or lessons learned.
        - When asking the follow-up question, reference the original question and answer. For example, start with a phrase like "Regarding [original question], you mentioned [brief summary of answer]. ..."
        - Do NOT generate a follow-up question if the answer expresses uncertainty (e.g., "I don't know", "I'm not sure") or is a negative response.
        - Strictly adhere to the following JSON format.
        - Only include the values corresponding to the value in the output format.
        - Questions are only asked in Korean.

        # Output Format
        {{
            "Question": ""
        }}
        """
    else:
        raise ValueError("Invalid type. Only 'technical' or 'behavioral' are allowed.")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": "You are an expert in interviewing and question generation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            response_content = completion.choices[0].message.content
            result = json.loads(response_content)
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed, retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(2)  # Short wait before retrying

    # Return default structure if all retries fail
    return {"error": "JSONDecodeError"}