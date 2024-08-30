from openai import OpenAI
import os
from dotenv import load_dotenv
import json

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 키 가져오기
api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY가 없습니다.")

# GPT 모델 가져오기
gpt_model = os.getenv("gpt")
if gpt_model is None:
    raise ValueError("GPT_Model이 없습니다.")

# OpenAI 클라이언트 초기화 및 API 키 등록
client = OpenAI(api_key=api_key)

# def generate_question(user_info: UserInput):
def generate_question(user_info):

    # 경력이 신입이 아닐 경우 "차" 문자열 붙이기
    years_with_suffix: str

    if user_info['years'] != "신입":
        years_with_suffix = f"{user_info['years']}년차"
    else:
        years_with_suffix = user_info['years']

    print(f"직업: {user_info['job']}")
    print(f"경력: {years_with_suffix}")
    print(f"답변: {user_info['answer']}")


    # API 호출을 위한 프롬프트
    user_prompt = (
        f"""
        
        면접자는 {user_info["job"]} 직군이고, 경력은 {years_with_suffix} 입니다.
        이 정보에 기반하여 {user_info["answer"]}에 관한 적절한 난이도의 꼬리물기 질문을 생성하세요.
        경력에 따라 난이도는 달라야하며, 경력이 오래될 수록 수준높은 질문과, 낮을 수록 난이도가 낮아야합니다.
        문제는 한 문제만 만들어야 합니다.
        질문은 한글로만 만들어야 합니다.

        # Output Format
        "Your first question here",
        """
    )

    system_prompt = (
        "당신은 면접관 입니다.\n"
        "사용자 입력에 따라 적절한 꼬리물기 질문을 생성해야 하며,\n"
        "기술적인 질문만 만들어야 합니다.\n"
        "면접자가 답한 내용은 말하면 안 되며, 오로직 당신은 질문만 만들어야 합니다.\n"
        "질문이 똑같아서는 안 됩니다."
    )

    # API 호출
    completion = client.chat.completions.create(
        model=gpt_model,
        # model="gpt-4o",
        # model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        top_p=1.0,
        n=2
    )

    response_content = completion.choices[0].message.content

    return json.loads(response_content)
