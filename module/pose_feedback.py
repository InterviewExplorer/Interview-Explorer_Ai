from openai import OpenAI
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# API 키 및 모델 이름 가져오기
api_key = os.getenv("API_KEY")
gpt_model = os.getenv("gpt")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

# 최종 피드백
def consolidate_feedback(feedback_list):
    # 피드백 리스트를 종합하여 최종 피드백 생성
    user_prompt = "".join([str(feedback) for feedback in feedback_list])
    # user_prompt = "\n".join(["\n".join(feedback) for feedback in feedback_list if isinstance(feedback, list)])
    # print("user_prompt(pose_feedback.py): ", user_prompt)

    system_prompt = """
        # Role
        You are a professional interview coach who can professionally correct posture. By analyzing the feedback received during the interview, it points out problems rather than solutions.

        # Output
        Explanation of the potential consequences of each detected action during an interview.

        # Task
        By analyzing the feedback received during the interview, it points out problems rather than solutions. Respond in Korean, explaining what consequences each detected action may have during an interview without providing solutions. Each feedback should be within 100 characters and should flow naturally. Do not use numbers or asterisks. Connect the feedbacks in a single paragraph.

        Provide your response in a structured and professional manner, addressing each detected action individually.
    """

    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Consolidate the following feedback entries into a brief summary: {user_prompt}"}
        ],
        temperature=0,
        top_p=0,
    )

    # 최종 피드백 반환
    consolidated_feedback = completion.choices[0].message.content
    return consolidated_feedback
