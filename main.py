# main.py
from module import firstLLM
import shutil
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from module.audio_extraction import convert_webm_to_mp3
from module.whisper_medium import transcribe_audio
import io
import os
import uuid
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from module.llm_openai import generate_question
from module.openai_evaluate import evaluate_answer
from module.openai_summarize import summarize_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 여기에 프론트엔드의 도메인 또는 '*'을 추가합니다
    allow_credentials=True,
    allow_methods=["*"],  # 필요한 HTTP 메서드를 설정합니다
    allow_headers=["*"],  # 필요한 헤더를 설정합니다
)

# 오디오 파일을 저장할 폴더를 확인하고, 없으면 생성합니다.
if not os.path.exists('audio'):
    os.makedirs('audio')

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        # 업로드된 파일을 메모리에서 직접 처리
        webm_file = io.BytesIO(await file.read())
        
        # 고유한 파일명을 생성
        unique_filename = f"{uuid.uuid4().hex}.mp3"
        audio_output_path = os.path.join("audio", unique_filename)

        # webm 파일을 mp3로 변환
        convert_webm_to_mp3(webm_file, audio_output_path)
        
        # MP3 파일을 텍스트로 변환
        with open(audio_output_path, "rb") as mp3_file:
            transcript = transcribe_audio(mp3_file)

        # MP3 파일 삭제 (옵션: 디스크 공간 절약)
        os.remove(audio_output_path)

        return JSONResponse(content={
            "status": "success",
            "message": "MP3 파일의 텍스트가 추출되었습니다.",
            "transcript": transcript
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generateQ/")
async def create_upload_file(
    job: str = Form(...),
    years: str = Form(...),
    file: UploadFile = File(None)
):
    if not job or not years:
        raise HTTPException(status_code=400, detail="직업군과 연차는 필수 입력 항목입니다.")

    pdf_content = None
    if file:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            pdf_content = temp_file.name

    print(f"직업군: {job}, 연차: {years}")
    if pdf_content:
        print(f"PDF 파일 저장 경로: {pdf_content}")

    result = firstLLM.generateQ(job, years, pdf_content)
    return JSONResponse(content=result)

# 데이터 모델 정의 (스웨거 테스트용, 곧 삭제 예정)
class UserInfo(BaseModel):
    job: str
    years: str

@app.post("/generate_question")
async def create_question(user_info: UserInfo):
    try:
        user_info_dic = user_info.dict()
        questions = generate_question(user_info_dic)
        return JSONResponse(content={"questions": questions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class EvaluateRequest(BaseModel):
    question: str
    answer: str
    years: str
    job: str

@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    # 요청 본문에서 데이터 추출
    question = request.question
    answer = request.answer
    years = request.years
    job = request.job

    if not question or not answer or not years or not job:
        raise HTTPException(status_code=400, detail="직업, 경력, 질문, 답변은 필수 항목입니다.")
    
    try:
        # 답변 평가
        evaluation = evaluate_answer(question, answer, years, job)
        return JSONResponse(content={"evaluation": evaluation})
    
    except Exception as e:
        # 에러 메시지 반환
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 데이터 모델 정의
class EvaluationData(BaseModel):
    evaluations: dict

@app.post("/summarize")
async def summarize(data: EvaluationData):
    try:
        # 평가 내용을 기반으로 요약 생성
        summary = summarize_text(data.evaluations)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))