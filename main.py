# main.py
from module import firstLLM
import shutil
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyPDFLoader
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from module.audio_extraction import convert_webm_to_mp3
from module.whisper_medium import transcribe_audio
import io
import os
import uuid

app = FastAPI()

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
async def create_upload_file(file: UploadFile):

     with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        # 업로드된 파일 내용을 읽어 임시 파일에 씁니다.
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
        
        return { firstLLM.generateQ(temp_file_path)}
