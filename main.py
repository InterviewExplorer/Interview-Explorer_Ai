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
