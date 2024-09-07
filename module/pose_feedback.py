from openai import OpenAI
import os
from dotenv import load_dotenv
from module.check_distance import analyze_landmarks

# 환경 변수 로드
load_dotenv()

# API 키 및 모델 이름 가져오기
api_key = os.getenv("API_KEY")
gpt_model = os.getenv("gpt")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

def analyze_pose_movement(pose_results):
    if not pose_results or not hasattr(pose_results, 'pose_landmarks'):
        raise ValueError("포즈 랜드마크를 가져올 수 없습니다.")
    
    feedback: str = ""

    # 결과에서 포즈 랜드마크를 가져오기
    pose_landmarks = pose_results.pose_landmarks

    if pose_landmarks:
        print("포즈 랜드마크가 사용 가능하고 유효합니다.")
    else:
        print("포즈 랜드마크를 사용할 수 없거나 유효하지 않습니다.")

    if pose_landmarks:
        feedback_list = analyze_landmarks(pose_landmarks)
        feedback = get_feedback_from_llm(feedback_list)
        consolidated_feedback = consolidate_feedback(feedback_list)
    else:
        feedback = "포즈와 손을 감지할 수 없습니다."

    print("행동 감지(pose_feedback.py): " + "".join(feedback_list))
    print("개별 피드백(pose_feedback.py): " + "".join(feedback))
    return feedback

# 각 영상 피드백
def get_feedback_from_llm(feedback_list):
    # 데이터를 텍스트로 변환
    user_prompt = "\n".join(feedback_list)

    system_prompt = """
        # Role
        You are an interviewer who is stubborn and strict, and whose way of speaking makes people feel hurt. Your job is to analyze the provided feedback about pose landmarks and provide concise, actionable feedback

        # Output
        Please provide your feedback in Korean, focusing on the following:

        # Task
        1. **Concise Feedback**: Provide clear and actionable suggestions for improvement without unnecessary details.

        Ensure your analysis is straightforward and helps the interviewee improve their body language effectively.
    """

    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze the following pose landmarks and provide feedback: {user_prompt}"}
        ],
        temperature=0,
        top_p=0,
    )

    # 피드백 적재
    return completion.choices[0].message.content

# 최종 피드백
def consolidate_feedback(feedback_list):
    # 피드백 리스트를 종합하여 최종 피드백 생성
    user_prompt = "".join(feedback_list)

    system_prompt = """
        # Role
        You are a distinguished expert tasked with consolidating individual feedback into a comprehensive analysis. Your task is to combine multiple feedback entries and provide a brief summary of the overall observations and recommendations.

        # Output
        Provide a structured summary in Korean, synthesizing the individual feedback entries into a coherent and actionable analysis in about 3 sentences.

        # Task
        1. **Comprehensive Analysis**: Combine and summarize the provided feedback entries.
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
