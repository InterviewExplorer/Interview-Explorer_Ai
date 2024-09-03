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

def analyze_pose_movement(results):
    if not results or not hasattr(results, 'pose_landmarks'):
        raise ValueError("포즈 랜드마크를 가져올 수 없습니다.")
    
    feedback: str = ""

    # 결과에서 포즈 랜드마크를 가져오기
    pose_landmarks = results.pose_landmarks

    if results.pose_landmarks:
        print("포즈 랜드마크가 사용 가능하고 유효합니다.")
    else:
        print("포즈 랜드마크를 사용할 수 없거나 유효하지 않습니다.")

    if pose_landmarks:
        # 랜드마크의 좌표를 텍스트로 변환
        landmarks_text = []

        for i, landmark in enumerate(pose_landmarks.landmark):
            x, y, z = landmark.x, landmark.y, landmark.z
            landmarks_text.append(f"Landmark {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}")

        # AI 모델에 피드백 요청
        feedback = get_feedback_from_llm(landmarks_text)
    else:
        feedback = "포즈를 감지할 수 없습니다."

    return feedback
def get_feedback_from_llm(landmarks_text):
    # 데이터를 텍스트로 변환
    user_prompt = "".join(landmarks_text)

    system_prompt = """"
        # Role
        You are an esteemed and formidable interviewer renowned for your keen observational skills and exacting standards. Your task is to meticulously analyze the provided video of an interview and identify areas where the interviewee exhibited excessive or unnecessary movements. Your analysis should focus on the following:

        # Output
        Please provide your feedback in Korean, using clear and actionable language. Your feedback should include:
        Your feedback should be delivered in a structured format, clearly indicating any issues or areas of improvement based on the pose landmarks provided.

        # Task
        1. **Identification of Excessive Movements**: Pinpoint specific instances or segments where the interviewee's movements were notably excessive or distracting.
        2. **Detailed Feedback**: Offer a thorough and insightful critique of these movements, explaining how they may impact the overall effectiveness and professionalism of the interview.

        Ensure your analysis reflects the high standards expected from an esteemed interviewer, providing clear and constructive feedback that can help improve the interviewee's performance in future scenarios.

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

# def consolidate_feedback(feedback_list):
#     # 피드백 리스트를 종합하여 최종 피드백 생성
#     user_prompt = "".join(feedback_list)

#     system_prompt = """
#         # Role
#         You are a distinguished expert tasked with consolidating individual feedback into a comprehensive analysis. Your task is to combine multiple feedback entries and provide a brief summary of the overall observations and recommendations.

#         # Output
#         Provide a structured summary in Korean, synthesizing the individual feedback entries into a coherent and actionable analysis in about 3 sentences.

#         # Task
#         1. **Comprehensive Analysis**: Combine and summarize the provided feedback entries.
#     """

#     completion = client.chat.completions.create(
#         model=gpt_model,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f"Consolidate the following feedback entries into a brief summary: {user_prompt}"}
#         ],
#         temperature=0,
#         top_p=0,
#     )

#     # 최종 피드백 반환
#     consolidated_feedback = completion.choices[0].message.content
#     return consolidated_feedback
