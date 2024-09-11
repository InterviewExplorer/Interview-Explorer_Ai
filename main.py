# main.py
from module import firstLLM
import shutil
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from module.audio_extraction import convert_webm_to_mp3
from module.whisper_medium import transcribe_audio
# from module.ibm import transcribe_audio
from module.ai_presenter import fetch_result_url
import io
import os
import uuid
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from module.llm_openai import follow_Q
from typing import Dict
import asyncio
from module.openai_evaluate import evaluate_answer
from module.openai_summarize import summarize_text
from module.pose_feedback import consolidate_feedback
from module.openai_speaking import evaluate_speaking
from module import openai_behavioral
# from module.pose_feedback import consolidate_feedback

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

# 상태 관리 클래스 정의
class FeedbackManager:
    def __init__(self):
        self.feedback = []  # 피드백을 저장할 리스트 초기화

    def add_feedback(self, feedback):
        self.feedback.extend(feedback)  # 피드백을 추가

    def reset_feedback(self):
        self.feedback = []  # 피드백 리스트 초기화

    def get_feedback(self):
        return self.feedback  # 현재 피드백 리스트 반환

feedback_manager = FeedbackManager() # 피드백 인스턴스 생성

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        # 업로드된 파일을 메모리에서 직접 처리
        webm_file = io.BytesIO(await file.read())
        
        # 고유한 파일명을 생성
        unique_filename = f"{uuid.uuid4().hex}.mp3"
        audio_output_path = os.path.join("audio", unique_filename)

        # webm 파일을 mp3로 변환
        feedback = convert_webm_to_mp3(webm_file, audio_output_path)
        # print("feedback(main.py): ", "".join(feedback))
        
        # MP3 파일을 텍스트로 변환
        with open(audio_output_path, "rb") as mp3_file:
            transcript = transcribe_audio(mp3_file)

        # MP3 파일 삭제 (옵션: 디스크 공간 절약)
        os.remove(audio_output_path)

        # 피드백 적재
        feedback_manager.add_feedback(feedback)

        return JSONResponse(content={
            "status": "success",
            "message": "MP3 파일의 텍스트가 추출되었습니다.",
            "transcript": transcript,
        })

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class FeedbackRequest(BaseModel):
    feedback: bool

# 프론트에서 "면접 종료" 누르면 boolean 값을 받아와서 피드백을 쏴주고 리셋
@app.post("/get_consolidate_feedback")
def get_consolidate_feedback(req: FeedbackRequest):
    try:
        if req.feedback:
            feedback_list = feedback_manager.get_feedback()
            # print("피드백 리스트(main.py): ", "".join(feedback_list))
            consolidated_feedback = consolidate_feedback(feedback_list)
            feedback_manager.reset_feedback()
            # print("통합 피드백(main.py): ", consolidated_feedback)
            return JSONResponse(content={
                "status": "success",
                "consolidated_feedback": consolidated_feedback
            })
        else:
            return JSONResponse(content={
                "status": "error",
                "message": "면접이 종료되지 않았습니다."
            })
    except Exception as e:
        print(f"에러 발생: {str(e)}")  # 에러 로깅 추가
        raise HTTPException(status_code=422, detail=str(e))

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

    # PDF 파일 삭제
    if pdf_content:
        try:
            os.remove(pdf_content)
            print(f"PDF 파일 삭제 완료: {pdf_content}")
        except Exception as e:
            print(f"PDF 파일 삭제 실패: {e}")
            
    return JSONResponse(content=result)

@app.post("/generateQ_behavioral/")
async def create_upload_file_behavioral(
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

    result = openai_behavioral.generateQ_behavioral(job, years, pdf_content)

    # PDF 파일 삭제
    if pdf_content:
        try:
            os.remove(pdf_content)
            print(f"PDF 파일 삭제 완료: {pdf_content}")
        except Exception as e:
            print(f"PDF 파일 삭제 실패: {e}")
            
    return JSONResponse(content=result)

@app.post("/ai-presenter")
async def ai_presenter(request: Request):
    form_data = await request.form()
    questions = {key: value for key, value in form_data.items()}  # FormData를 딕셔너리로 변환
    # print(questions)
    # 모든 질문에 대해 결과 URL을 비동기적으로 가져오기
    tasks = [fetch_result_url(key, question) for key, question in questions.items()]
    results_list = await asyncio.gather(*tasks)
    # print(results_list, "results_list")
    # 결과를 하나의 딕셔너리로 병합하기
    results = {}
    for result in results_list:
        results.update(result)
    # print(results)
    # 결과를 JSON 형태로 반환
    return  results

class UserInfo(BaseModel):
    job: str
    years: str
    answer: str
    questions: dict
    type: str

@app.post("/follow_question")
async def follow_question(userinfo: UserInfo):
    job = userinfo.job
    years = userinfo.years
    answer = userinfo.answer
    questions = userinfo.questions
    type = userinfo.type

    if not answer or not years or not job:
        raise HTTPException(status_code=400, detail="직업, 경력, 답변은 필수 항목입니다.")

    try:
        followQuestion = follow_Q(answer, years, job, questions, type)
        return JSONResponse(content=followQuestion)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

class EvaluateRequest(BaseModel):
    question: str
    answer: str
    years: str
    job: str
    type: str

@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    # 요청 본문에서 데이터 추출
    question = request.question
    answer = request.answer
    years = request.years
    job = request.job
    type = request.type

    if not question or not answer or not years or not job:
        raise HTTPException(status_code=400, detail="직업, 경력, 질문, 답변은 필수 항목입니다.")
    
    try:
        # 답변 평가
        evaluation = evaluate_answer(question, answer, years, job, type)
        return JSONResponse(content={"evaluation": evaluation})
    
    except Exception as e:
        # 에러 메시지 반환
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 데이터 모델 정의
class EvaluationData(BaseModel):
    evaluations: dict
    type: str

@app.post("/summarize")
async def summarize(data: EvaluationData):
    try:
        summary = summarize_text(data.evaluations, data.type)
        return JSONResponse(content=summary)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AnswersInput(BaseModel):
    answers: Dict[str, str]
    
@app.post("/speaking")
async def speaking(input: AnswersInput):
    try:
        evaluation = evaluate_speaking(input.answers)
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))