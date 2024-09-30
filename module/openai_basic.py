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
        - Generate questions that relate specific technologies, considering the {job} and {year} of experience.
        - Ensure that questions are relevant, clear, and focused on assessing technical knowledge.
        - Construct questions at a level appropriate for the years of experience provided.
        - Generate technical questions based on the user's role and experience level.

        # Task1
        - Generate 10 questions to evaluate the interviewee's technical-understanding approach.
        - Ask specific questions about your previous technology stack description, why it was appropriate, what programming language or tools you have a deep understanding of, examples of how you used it, and specific skills or latest trends needed for your job.
        - Place the generated questions under the 'technical_understanding' section in the Output Format.

        # Task2
        - Generate 10 questions to evaluate the interviewee's problem-solving approach.
        - Ask specific questions about the problem-solving process, your experience solving a specific technical problem, how you analyzed and determined a solution to a given problem, and the difficulties you faced in the problem-solving process and how you solved it.
        - Place the generated questions under the 'problem_solving' section in the Output Format.

        # Task3
        - Generate 10 questions to evaluate the interviewee's logical-thinking approach.
        - Assume a specific situation where a problem needs to be solved, and then ask the question.
        - Place the generated questions under the 'logical_thinking' section in the Output Format.

        # Task4
        - Generate 10 questions to evaluate the interviewee's learning-ability approach.
        - Ask specifically about their experiences learning new technologies or tools, their learning methods, how they applied what they learned, their responses to technical changes or new challenges, specific examples, and how they self-studied to solve technical problems and the results of those efforts.
        - Place the generated questions under the 'learning_ability' section in the Output Format.

        # Task5
        - Generate 10 questions to evaluate the interviewee's collaboration-communication approach.
        - Ask specifically about the problems faced during collaboration and the solutions, experiences in resolving technical issues, and how they handled disagreements or conflicts.
        - Ask the interviewee to provide specific examples from their projects or experiences.
        - Place the generated questions under the 'collaboration_communication' section in the Output Format.

        # Policy
        - Ask for specific experiences or examples in the question.
        - Questions should be answerable through verbal explanation.
        - Write your questions in Korean only.
        - Do not ask for code examples.
        - Responses must be in JSON format.

        # Output Format
        {{
            "technical_understanding" : [
                ""
                ...
            ],
            "problem_solving" : [
                ""
                ...
            ],
            "logical_thinking" : [
                ""
                ...
            ],
            "learning_ability" : [
                ""
                ...
            ],
            "collaboration_communication" : [
                ""
                ...
            ],
        }}
        """
    elif interviewType == "behavioral":
        prompt = f"""
        # Role
        You are an interviewer specializing in conducting personality interviews.

        # Task
        Create personality interview questions based on the following criteria:
        - User experience level: {year} years
        - User role: {job}

        # Instructions
        - Generate personality assessment questions considering the {job} and {year} of experience.
        - Ensure that questions are relevant, clear, and focused on assessing personality traits.
        - Ask the interviewee for specific experiences or examples in the questions.

        # Task1
        - Generate 10 questions to assess the interviewee's self-motivation and passion.
        - Ask specific questions about how to motivate oneself, how to maintain motivation, what efforts are made to achieve goals, and the challenges and success stories encountered in that process.
        - Place the generated questions under the 'self_motivation' section in the Output Format.

        # Task2
        - Generate 10 questions to assess the interviewee's self-awareness.
        - Example : How do you perceive your strengths and weaknesses and what efforts are you making to improve them?
        - Example : In which part of the work you do you feel you need improvement and how have you tried to improve?
        - Example : How do you accept feedback when you receive it, and how do you adjust your behavior and attitude?
        - Place the generated questions under the 'self_awareness' section in the Output Format.

        # Task3
        - Generate 10 questions to assess the interviewee's interpersonal-relationships.
        - Place the generated questions under the 'interpersonal_relationships' section in the Output Format.

        # Task4
        - Generate 10 questions to assess the interviewee's honesty.
        - Ask specific questions about experiences and how to deal with dishonest situations, experiences and how to deal with mistakes during work, lessons learned along the way, and how to respond when a team member shows an unfaithful attitude.
        - Place the generated questions under the 'honesty' section in the Output Format.

        # Task5
        - Generate 10 questions to assess the interviewee's adaptability.
        - Place the generated questions under the 'adaptability' section in the Output Format.

        # Policy
        - Ask for specific experiences or examples in the question.
        - Questions should be answerable through verbal explanation.
        - Write your questions in Korean only.
        - Responses must be in JSON format.
        - Refer to users as '면접자'.

        # Output Format
        {{
            "self_motivation " : [
                ""
                ...
            ],
            "self_awareness" : [
                ""
                ...
            ],
            "interpersonal_relationships" : [
                ""
                ...
            ],
            "honesty" : [
                ""
                ...
            ],
            "adaptability" : [
                ""
                ...
            ],
        }}
        """
    else:
        raise ValueError("인터뷰 유형이 잘못되었습니다. 다시 선택해 주세요.")
    
    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are a professional interviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    response_content = completion.choices[0].message.content
    print("response_content", response_content)

    try:
        result = json.loads(response_content)

        categories = []

        if interviewType == "technical":
            categories = [
                "technical_understanding", 
                "problem_solving", 
                "logical_thinking", 
                "learning_ability", 
                "collaboration_communication"
            ]
        elif interviewType == "behavioral":
            categories = [
                "self_motivation", 
                "self_awareness",
                "interpersonal_relationships",
                "honesty",
                "adaptability", 
            ]
            # behavioral 인터뷰의 경우 시작 인덱스는 1 (기본값)

        selected_questions = {}
        for index, category in enumerate(categories, start=3):
            questions_list = result.get(category, [])
            if questions_list:
                selected_question = random.choice(questions_list)
                selected_questions[f"Q{index}"] = selected_question
            else:
                selected_questions[f"Q{index}"] = "질문이 없습니다."

        return selected_questions


    except json.JSONDecodeError as e:
        return {"error": f"JSON 파싱 오류: {e}"}
    
    except Exception as e:
        raise ValueError(f"질문 생성 실패: {e}")
