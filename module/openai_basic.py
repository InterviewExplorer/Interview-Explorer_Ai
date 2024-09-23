from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import random

load_dotenv()


api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY가 없습니다.")

gpt_model = os.getenv("gpt")
if gpt_model is None:
    raise ValueError("GPT_Model이 없습니다.")

client = OpenAI(api_key=api_key)

def create_basic_question(job, year, interviewType):
    
    if interviewType == "technical":
        prompt = f"""
        # Role
        You are the interviewer who creates technical interview questions.

        # Task
        Create technical questions based on the following criteria:
        - User experience level: {year} years
        - User role: {job}

        # Instructions
        - Generate questions that relate specific technologies, considering the job and experience.
        - Ensure that questions are relevant, clear, and focused on assessing technical knowledge.
        - Construct questions at a level appropriate for the years of experience provided.
        - Generate technical questions based on the user's role and experience level.

        # Task1
        - Generate 10 questions to evaluate the candidate's problem-solving approach.
        - Ask specific questions about the problem-solving process, your experience solving a specific technical problem, how you analyzed and determined a solution to a given problem, and the difficulties you faced in the problem-solving process and how you solved it.
        - Place the generated questions under the 'problem_solving' section in the Output Format.

        # Task2
        - Generate 10 questions to evaluate the candidate's technical-understanding approach.
        - Ask specific questions about your previous technology stack description, why it was appropriate, what programming language or tools you have a deep understanding of, examples of how you used it, and specific skills or latest trends needed for your job.
        - Place the generated questions under the 'technical_understanding' section in the Output Format.

        # Task3
        - Generate 10 questions to evaluate the candidate's logical-thinking approach.
        - Assume a specific situation where a problem needs to be solved, and then ask the question.
        - Place the generated questions under the 'logical_thinking' section in the Output Format.

        # Task4
        - Generate 10 questions to evaluate the candidate's learning-ability approach.
        - Ask specifically about their experiences learning new technologies or tools, their learning methods, how they applied what they learned, their responses to technical changes or new challenges, specific examples, and how they self-studied to solve technical problems and the results of those efforts.
        - Place the generated questions under the 'learning_ability' section in the Output Format.

        # Task5
        - Generate 10 questions to evaluate the candidate's collaboration-communication approach.
        - Ask specifically about the problems faced during collaboration and the solutions, experiences in resolving technical issues, and how they handled disagreements or conflicts.
        - Ask the candidate to provide specific examples from their projects or experiences.
        - Place the generated questions under the 'collaboration_communication' section in the Output Format.

        # Policy
        - Questions should be answerable through verbal explanation.
        - Write your questions in Korean only.
        - Do not ask for code examples.
        - Responses must be in JSON format.

        # Output Format
        {{
            "problem_solving" : {{
                "Q1": "",
                "Q2": "",
                "Q3": "",
                "Q4": "",
                "Q5": "",
                "Q6": "",
                "Q7": "",
                "Q8": "",
                "Q9": "",
                "Q10": ""
            }},
            "technical_understanding" : {{
                "Q1": "",
                "Q2": "",
                "Q3": "",
                "Q4": "",
                "Q5": "",
                "Q6": "",
                "Q7": "",
                "Q8": "",
                "Q9": "",
                "Q10": ""
            }},
            "logical_thinking" : {{
                "Q1": "",
                "Q2": "",
                "Q3": "",
                "Q4": "",
                "Q5": "",
                "Q6": "",
                "Q7": "",
                "Q8": "",
                "Q9": "",
                "Q10": ""
            }},
            "learning_ability" : {{
                "Q1": "",
                "Q2": "",
                "Q3": "",
                "Q4": "",
                "Q5": "",
                "Q6": "",
                "Q7": "",
                "Q8": "",
                "Q9": "",
                "Q10": ""
            }},
            "collaboration_communication" : {{
                "Q1": "",
                "Q2": "",
                "Q3": "",
                "Q4": "",
                "Q5": "",
                "Q6": "",
                "Q7": "",
                "Q8": "",
                "Q9": "",
                "Q10": ""
            }}
        }}
        """
    elif interviewType == "behavioral":
        prompt = f"""
        # Role
        You are the interviewer who creates personality interview questions.

        # Instructions
        - Questions should be relevant, clear, and focused on assessing character.
        - Questions should be answerable through verbal explanation.
        - Write your questions in Korean only.
        - Generate 50 unique (non-duplicate) personality questions.
        - Randomly choose 10 out of 50 questions you created
        - The first problem is to create a question that can evaluate "honesty (trustworthiness)".
        - The second problem is to create questions that can evaluate "interpersonal relationships."
        - The third problem is to create a question that can evaluate "self-motivation (passion)."
        - The fourth problem is to create a question that can evaluate "adaptability."
        - The fifth problem is to create a question that can evaluate "self-awareness."

        # Policy
        - Responses must be in JSON format.
        - The first question generated places the values in order from "honesty" in the JSON output.
        - The first question generated places the values in order from "interpersonal_relationships" in the JSON output.
        - The first question generated places the values in order from "self_motivation" in the JSON output.
        - The first question generated places the values in order from "adaptability" in the JSON output.
        - The first question generated places the values in order from "self_awareness" in the JSON output.

        # Output Format
        {{
            "honesty" : {{
                "Q1": "",
                "Q2": "",
                "Q3": "",
                "Q4": "",
                "Q5": "",
                "Q6": "",
                "Q7": "",
                "Q8": "",
                "Q9": "",
                "Q10": ""
            }},
            "interpersonal_relationships" : {{
                "Q1": "",
                "Q2": "",
                "Q3": "",
                "Q4": "",
                "Q5": "",
                "Q6": "",
                "Q7": "",
                "Q8": "",
                "Q9": "",
                "Q10": ""
            }},
            "self_motivation " : {{
                "Q1": "",
                "Q2": "",
                "Q3": "",
                "Q4": "",
                "Q5": "",
                "Q6": "",
                "Q7": "",
                "Q8": "",
                "Q9": "",
                "Q10": ""
            }},
            "adaptability" : {{
                "Q1": "",
                "Q2": "",
                "Q3": "",
                "Q4": "",
                "Q5": "",
                "Q6": "",
                "Q7": "",
                "Q8": "",
                "Q9": "",
                "Q10": ""
            }},
            "self_awareness" : {{
                "Q1": "",
                "Q2": "",
                "Q3": "",
                "Q4": "",
                "Q5": "",
                "Q6": "",
                "Q7": "",
                "Q8": "",
                "Q9": "",
                "Q10": ""
            }}
        }}
        """
    else:
        raise ValueError("인터뷰 유형이 잘못되었습니다. 다시 선택해 주세요.")

    try:
        completion = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are a professional interviewer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        response_content = completion.choices[0].message.content
        result = json.loads(response_content)

        if interviewType == "technical":
            categories = [
                "problem_solving",
                "technical_understanding",
                "logical_thinking",
                "learning_ability",
                "collaboration_communication"
            ]
            questions = {
                "problem_solving": [],
                "technical_understanding": [],
                "logical_thinking": [],
                "learning_ability": [],
                "collaboration_communication": []
            }
        else:
            categories = [
                "honesty",
                "interpersonal_relationships",
                "self_motivation",
                "adaptability",
                "self_awareness"
            ]
            questions = {
                "honesty": [],
                "interpersonal_relationships": [],
                "self_motivation": [],
                "adaptability": [],
                "self_awareness": []
            }

        for category in categories:
            for i in range(1, 11):
                question = result[category].get(f"Q{i}")
                if question:
                    questions[category].append(question)

        # 모든 질문 생성 프린트문
        for category, question_list in questions.items():
            print(f"{category}:")
            for question in question_list:
                print(f" - {question}")

        selected_questions = {}
        for category, question_list in questions.items():
            if question_list:
                random.shuffle(question_list)
                selected_questions[category] = question_list[0]

        json_output = {}
        for index, (category, question) in enumerate(selected_questions.items(), start=1):
            json_output[f"Q{index}"] = question

        return json_output
    except Exception as e:
        raise ValueError(f"질문 생성 실패: {e}")
