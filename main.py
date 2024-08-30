# main.py
from module import firstLLM
import shutil
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form,Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from module.audio_extraction import convert_webm_to_mp3
from module.whisper_medium import transcribe_audio
from module.ai_presenter import fetch_result_url
import io
import os
import uuid
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from module.llm_openai import generate_question
from typing import Dict
import asyncio

app = FastAPI()
@app.get("/")
async def hello_world():
    return {"message": "hello"}
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

@app.post("/ai-presenter")
async def ai_presenter(request: Request):
    form_data = await request.form()
    questions = {key: value for key, value in form_data.items()}  # FormData를 딕셔너리로 변환
    print(questions)
    # 모든 질문에 대해 결과 URL을 비동기적으로 가져오기
    tasks = [fetch_result_url(question) for question in questions.values()]
    results_list = await asyncio.gather(*tasks)
    print(results_list, "results_list")
    # 결과를 하나의 딕셔너리로 병합하기
    results = {}
    for result in results_list:
        results.update(result)
    print(results)
    # 결과를 JSON 형태로 반환
    return {"results": results}

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