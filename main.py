# main.py
from module import firstLLM
import shutil
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyPDFLoader
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from module.audio_extraction import convert_webm_to_mp3
from module.whisper_medium import transcribe_audio
from module.llm_openai import generate_question
import io
import os
import uuid
from pydantic import BaseModel

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
async def create_upload_file(file: UploadFile):

     with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        # 업로드된 파일 내용을 읽어 임시 파일에 씁니다.
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    
    
    
        return { firstLLM.generateQ(temp_file_path)}
     
# 데이터 모델 정의 (스웨거 테스트용, 곧 삭제 예정)
class UserInfo(BaseModel):
    role: str
    experience_level: str
    answer: str

@app.post("/generate_question")
async def create_question(user_info: UserInfo):
    try:
        user_info_dic = user_info.dict()
        questions = generate_question(user_info_dic)
        return JSONResponse(content={"questions": questions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
