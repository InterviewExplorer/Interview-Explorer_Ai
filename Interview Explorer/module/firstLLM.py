from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader

# PDF 파일 로드
loader = PyPDFLoader("test.pdf")
document = loader.load()

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 키 가져오기
api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY가 없습니다.")

# OpenAI 클라이언트 초기화 및 api키 등록
client = OpenAI(api_key=api_key)

# 사용자 입력 예시
test = document

# API 호출을 위한 프롬프트 구성
prompt = (
    f"{test}를 토대로 이 사람이 써본 기술과 프로젝트 경험을 토대로 5가지의 질문을 만들고 이 사람의 인적사항과 진행한 프로젝트들의 이름도 알려줘"
)

# API 호출
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",  # 또는 다른 지원되는 모델 이름
    # model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "당신은 면접관입니다. 사용자 입력에 따라 적절한 꼬리물기 질문을 생성해야 하며, 기술적인 질문만 만들어야 합니다."},
        {"role": "user", "content": prompt}
    ]
)

# 결과 출력, 여러 개의 답변 중 첫 번째 답변 가져오기
print(completion.choices[0].message.content)
# 결과 출력
# print(completion.choices[0].message.content.strip())

