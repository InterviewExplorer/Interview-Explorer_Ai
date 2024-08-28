# main.py
from module import firstLLM
import shutil
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyPDFLoader
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from module.audio_extraction import convert_webm_to_mp3
from module.whisper_medium import transcribe_audio
import io
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 허용할 도메인
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 오디오 파일을 저장할 폴더를 확인하고, 없으면 생성합니다.
if not os.path.exists('audio'):
    os.makedirs('audio')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/extract_audio")
async def extract_audio(file: UploadFile = File(...)):
    try:
        # 업로드된 파일을 메모리에서 직접 처리
        webm_file = io.BytesIO(await file.read())
        
        # 고유한 파일명을 생성
        unique_filename = f"{uuid.uuid4().hex}.mp3"
        audio_output_path = os.path.join("audio", unique_filename)

        # webm 파일을 mp3로 변환
        convert_webm_to_mp3(webm_file, audio_output_path)

        return JSONResponse(content={
            "status": "success",
            "message": f"오디오가 '{audio_output_path}'로 저장되었습니다."
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_text")
async def extract_text_from_mp3(file: UploadFile = File(...)):
    try:
        # 업로드된 MP3 파일을 메모리에서 직접 처리
        mp3_file = io.BytesIO(await file.read())
        
        # MP3 파일을 텍스트로 변환 (파일 경로 대신 파일 스트림을 전달)
        transcript = transcribe_audio(mp3_file)

        return JSONResponse(content={
            "status": "success",
            "message": "MP3 파일의 텍스트가 추출되었습니다.",
            "transcript": transcript
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generateQ/")
async def create_upload_file(file: UploadFile = File(...), job: str = "", years: str = ""):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

        # 여기서 직업군과 연차 데이터를 사용할 수 있습니다.
        print(f"직업군: {job}, 연차: {years}")
        print(f"PDF 파일 저장 경로: {temp_file_path}")
        
        # PDF 파일과 추가 데이터를 기반으로 질문 생성
        result = firstLLM.generateQ(temp_file_path)  # firstLLM 부분은 구현에 따라 수정

        return {"result": result}
