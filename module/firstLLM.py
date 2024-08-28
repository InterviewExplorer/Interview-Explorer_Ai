from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader

def generateQ(pdf_file) :

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
  user_experience_level = "2년"  # 경력 수준 예시
  user_role = "백엔드 개발자"  # 직군 예시
  user_skill = "자바"  # 사용자 기술 예시

  resume = document

  # API 호출을 위한 프롬프트 구성
  markdown_prompt = f"""
  # Role
  you are the interviewer

  Output Format
  {{
  첫번째 질문: ,
  두번째 질문: 
  }}

  # Task
  - You will be provided with `{user_experience_level}`, `{user_role}`, `{user_skill}`, and optionally `{resume}`.
  - If `{resume}` is provided, create 2 technical questions based on the technologies and project experience described in `{resume}`.
  - If `{resume}` is not provided, create 2 technical questions based on `{user_experience_level}`, `{user_role}`, and `{user_skill}`.

  # Policy
  - If `{resume}` is provided, prioritize creating questions based on `{resume}` and do not rely on `{user_experience_level}`, `{user_role}`, or `{user_skill}` unless necessary.
  - If `{resume}` is not provided, create questions solely based on `{user_experience_level}`, `{user_role}`, and `{user_skill}`.
  - Do not create any other content beyond the two technical questions.
  - Just ask questions that can be explained in words.
  - Questions should be relevant, clear, and focused on assessing technical knowledge.
  """

  # API 호출
  completion = client.chat.completions.create(
      # model="gpt-3.5-turbo",  # 또는 다른 지원되는 모델 이름
      model="gpt-4o-mini",
      messages=[
          {"role": "system", "content": "당신은 면접관입니다, 당신은 전문적인 개발자입니다"},
          {"role": "user", "content": markdown_prompt}
      ]
  )

  # 결과 출력, 여러 개의 답변 중 첫 번째 답변 가져오기
  return completion.choices[0].message.content


