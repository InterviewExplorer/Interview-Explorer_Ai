# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from module.audio_extraction import convert_webm_to_mp3
from module.whisper_medium import transcribe_audio
import io
import os
import uuid

app = FastAPI()

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
        
        # 고유한 파일명을 생성
        unique_filename = f"{uuid.uuid4().hex}.mp3"
        audio_output_path = os.path.join("audio", unique_filename)

        # MP3 파일을 로컬 파일로 저장
        with open(audio_output_path, "wb") as f:
            f.write(mp3_file.getbuffer())

        # MP3 파일을 텍스트로 변환
        transcript = transcribe_audio(audio_output_path)

        return JSONResponse(content={
            "status": "success",
            "message": f"'{audio_output_path}' 파일의 텍스트가 추출되었습니다.",
            "transcript": transcript
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
