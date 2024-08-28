from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 키 가져오기
api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY가 없습니다.")

# OpenAI 클라이언트 초기화 및 API 키 등록
client = OpenAI(api_key=api_key)

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})

# 데이터 모델 정의
class UserInput(BaseModel):
    experience_level: str # 신입/경력 값
    role: str # 직군 값
    answer: str # 앞에서의 질문 값

@app.post("/generate_question")
async def generate_question(user_info: UserInput):

    # 경력이 신입이 아닐 경우 "차" 문자열 붙이기
    experience_with_suffix: str

    if user_info.experience_level != "신입":
        experience_with_suffix = f"{user_info.experience_level}차"
    else:
        experience_with_suffix = user_info.experience_level
    
    # API 호출을 위한 프롬프트
    user_prompt = (
        f"면접자는 {user_info.role} 직군이고, 경력은 {experience_with_suffix} 이며, "
        f"이 정보에 기반하여 {user_info.answer}에 관한 적절한 난이도의 꼬리물기 질문을 생성하세요."
    )

    print("경력: ", experience_with_suffix)
    print("직군: ", user_info.role)
    print("답변: ", user_info.answer)

    system_prompt = (
        "당신은 면접관 입니다.\n"
        "사용자 입력에 따라 적절한 꼬리물기 질문을 생성해야 하며,\n"
        "기술적인 질문만 만들어야 합니다.\n"
        "면접자가 답한 내용은 말하면 안 되며, 오로직 당신은 질문만 만들어야 합니다.\n"
        "질문이 똑같아서는 안 됩니다."
    )

    try:
        # API 호출
        completion = client.chat.completions.create(
            # model="gpt-3.5-turbo",  # 또는 다른 지원되는 모델 이름
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1.0,
            top_p=1.0,
            n=2 # 생성할 응답의 수를 2로 설정
        )

        # 두 개의 질문 출력
        questions  = [choice.message.content for choice in completion.choices]

        return {"questions ": questions }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API 호출 에러: {str(e)}")
    