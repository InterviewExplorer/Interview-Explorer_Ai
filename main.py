# main.py
from module import firstLLM
import shutil
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from module.audio_extraction import convert_webm_to_mp3
# from module.whisper_medium import transcribe_audio
from module.whisper_api import transcribe_audio
from module.ai_presenter import fetch_result_url
import io
import os
import uuid
import cv2
import base64
import numpy as np
from module.guide import process_frame
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from module.llm_openai import follow_Q
from typing import Dict, List
import asyncio
from module.openai_evaluate import evaluate_answer
from module.openai_summarize import summarize_text
from module.pose_feedback import consolidate_feedback
from module.openai_speaking import evaluate_speaking
from module import openai_behavioral
from module.openai_resumeTech import technical_resume
from module.openai_resumBehav import behavioral_resume
# from module.pose_feedback import consolidate_feedback
from module.openai_basic import create_basic_question
from module.openai_each import assessment_each
from module.openai_average import calculate_average
import json
from module.indexClear import delete_docs
from rag.rag_createNew import create_newQ
from rag.rag_evaluateNew import evaluate_newQ
from module.openai_contentSummary import summaryOfContent
from module.pdfSave import main
from module.openai_pdf import pdf
from module.pdfSearch import search

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
        # feedback, face_touch_total, hand_move_total, not_front_total = convert_webm_to_mp3(webm_file, audio_output_path)
        feedback = convert_webm_to_mp3(webm_file, audio_output_path)
        print("feedback(main.py): ", feedback)
        
        # MP3 파일을 텍스트로 변환
        with open(audio_output_path, "rb") as mp3_file:
            transcript = transcribe_audio(mp3_file)

        # MP3 파일 삭제 (옵션: 디스크 공간 절약)
        os.remove(audio_output_path)

        print("@@@@@추출된 답변", transcript)

        return JSONResponse(content={
            "status": "success",
            "message": "MP3 파일의 텍스트가 추출되었습니다.",
            "transcript": transcript,
            "feedback": feedback,
            # "face_touch_total": face_touch_total,
            # "hand_move_total": hand_move_total,
            # "not_front_total": not_front_total
        })

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_consolidate_feedback")
# async def get_consolidate_feedback(req: List[str] = Form(...)):
async def get_consolidate_feedback(req: Request):
    try:
        req_json = await req.json()
        # print("req_json(main.py): ", req_json)
        feedback_list = req_json.get("feedback", {}).get("feedbackList")

        # print("Received feedback_list(main.py): ", feedback_list)
        # print("Type of feedback_list: ", type(feedback_list))

        if feedback_list:
            consolidated_feedback = consolidate_feedback(feedback_list)
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

# 우현 면접 질문 생성
@app.post("/basic_question")
async def basic_question(job: str = Form(...), years: str = Form(...), interviewType: str = Form(...)):
    if not job or not years:
        raise HTTPException(status_code=400, detail="직업군과 연차는 필수 입력 항목입니다.")

    result = create_basic_question(job, years, interviewType)

    if isinstance(result, str):
        # print("반환 값이 STR 입니다.")
        result = json.loads(result)

    # 요소당 한 개의 질문, 총 다섯 개의 질문 생성 프린트문
    # print("질문 생성 목록(BE): ", json.dumps(result, indent=4, ensure_ascii=False))

    return JSONResponse(content=result)
        
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

@app.post("/each")
async def each(
    question: str = Form(...), 
    answer: str = Form(...), 
    years: str = Form(...), 
    job: str = Form(...), 
    type: str = Form(...)
):

    if not question or not answer or not years or not job:
        raise HTTPException(status_code=400, detail="직업, 경력, 질문, 답변은 필수 항목입니다.")
    
    try:
        # 답변 평가
        evaluation = assessment_each(question, answer, years, job, type)
        return JSONResponse(content={"evaluation": evaluation})
    
    except Exception as e:
        # 에러 메시지 반환
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/average")
async def average(job: str = Form(...), years: str = Form(...), type: str = Form(...)):
    if not job or not years:
        raise HTTPException(status_code=400, detail="직업군과 연차는 필수 입력 항목입니다.")

    result = calculate_average(job, years, type)

    if isinstance(result, str):
        result = json.loads(result)

    return JSONResponse(content=result)

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
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()

            try:
                img_data = base64.b64decode(data.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                processed_frame, success_flag = process_frame(frame)
                
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                processed_image = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_json({
                    "image": f"data:image/jpeg;base64,{processed_image}",
                    "success": success_flag
                })
            except Exception as e:
                (f"Error processing frame: {str(e)}")
    except Exception as e:
        (f"WebSocket error: {str(e)}")
    finally:
        ("WebSocket connection closed")

@app.post("/technical_resume")
async def create_upload_file(
    job: str = Form(...),
    years: str = Form(...),
    interviewType: str = Form(...),
    file: UploadFile = File(None),
    basicQuestion_Q3: str = Form(None),
    basicQuestion_Q4: str = Form(None),
    basicQuestion_Q5: str = Form(None),
    basicQuestion_Q6: str = Form(None),
    basicQuestion_Q7: str = Form(None)
):
    if not job or not years:
        raise HTTPException(status_code=400, detail="직업군과 연차는 필수 입력 항목입니다.")

    pdf_content = None
    if file:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            pdf_content = temp_file.name

    print(f"직업군: {job}, 연차: {years}, 면접 유형: {interviewType}")
    if pdf_content:
        print(f"PDF 파일 저장 경로: {pdf_content}")

    # 기본 질문 처리
    basic_questions = {
        "Q3": basicQuestion_Q3,
        "Q4": basicQuestion_Q4,
        "Q5": basicQuestion_Q5,
        "Q6": basicQuestion_Q6,
        "Q7": basicQuestion_Q7
    }
    print("기본 질문:", basic_questions)

    # 여기서 technical_resume 함수를 호출하고 필요한 인자들을 전달합니다.
    result = technical_resume(job, years, pdf_content, basic_questions)

    # PDF 파일 삭제
    if pdf_content:
        try:
            os.remove(pdf_content)
            print(f"PDF 파일 삭제 완료: {pdf_content}")
        except Exception as e:
            print(f"PDF 파일 삭제 실패: {e}")
            
    return JSONResponse(content=result)

@app.post("/behavioral_resume")
async def create_upload_file_behavioral(
    job: str = Form(...),
    years: str = Form(...),
    interviewType: str = Form(...),
    file: UploadFile = File(None),
    basicQuestion_Q3: str = Form(None),
    basicQuestion_Q4: str = Form(None),
    basicQuestion_Q5: str = Form(None),
    basicQuestion_Q6: str = Form(None),
    basicQuestion_Q7: str = Form(None)
):
    if not job or not years:
        raise HTTPException(status_code=400, detail="직업군과 연차는 필수 입력 항목입니다.")

    pdf_content = None
    if file:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            pdf_content = temp_file.name

    print(f"직업군: {job}, 연차: {years}, 면접 유형: {interviewType}")
    if pdf_content:
        print(f"PDF 파일 저장 경로: {pdf_content}")

    # 기본 질문 처리
    basic_questions = {
        "Q3": basicQuestion_Q3,
        "Q4": basicQuestion_Q4,
        "Q5": basicQuestion_Q5,
        "Q6": basicQuestion_Q6,
        "Q7": basicQuestion_Q7
    }
    print("기본 질문:", basic_questions)

    # 여기서 behavioral_resume 함수를 호출하고 필요한 인자들을 전달합니다.
    result = behavioral_resume(job, years, pdf_content, basic_questions)

    # PDF 파일 삭제
    if pdf_content:
        try:
            os.remove(pdf_content)
            print(f"PDF 파일 삭제 완료: {pdf_content}")
        except Exception as e:
            print(f"PDF 파일 삭제 실패: {e}")
            
    return JSONResponse(content=result)

@app.post("/newQ_create")
async def newQuestion_create(job: str = Form(...), type: str = Form(...), answers: str = Form(...)):

    print("job: ", job)
    print("type: ", type)
    print("answers: ", answers) 

    if not job or not type or not answers:
        raise HTTPException(status_code=400, detail="직업, 타입, 답변은 필수 입력 항목입니다.")
    
    # 전달받은 답변 내용 요약해서 value 값만 전달
    resultOfSummary = summaryOfContent(answers)
    print("요약 답변: ", resultOfSummary["Summary"])
    summaryOfAnswers = resultOfSummary.get('Summary', '')

    result = create_newQ(job, type, summaryOfAnswers)

    return JSONResponse(content=result)

@app.post("/newQ_evaluate")
async def newQuestion_evaluete(
    question: str = Form(...), 
    answer: str = Form(...), 
    years: str = Form(...), 
    job: str = Form(...), 
    type: str = Form(...)
):
    if not question or not answer or not years or not job or not type:
        raise HTTPException(status_code=400, detail="필수 입력 항목을 확인해주세요.")
    
    result = evaluate_newQ(question, answer, years, job, type)

    return JSONResponse(content={"evaluation": result})

@app.post("/reset_index")
async def delete_resumes_nori():
    delete_docs()

    return ("삭제완료")

@app.post("/pdf")
async def create_upload_files(files: list[UploadFile] = File(...), sources: List[str] = Form(...)):
    pdf_contents = []
    results = []

    # 업로드된 PDF 파일 처리
    for file in files:
        if file.content_type == "application/pdf":
            # 임시 파일로 PDF 저장
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                pdf_contents.append(temp_file.name)

    # 각 PDF에 대해 텍스트 추출 및 JSON 변환
    for pdf_content, source in zip(pdf_contents, sources):
        # pdf 함수가 비동기 함수라면 await로 호출
        result = await pdf(pdf_content)
        main(result, source)

        # PDF 파일 삭제
        try:
            os.remove(pdf_content)
            print(f"PDF 파일 삭제 완료: {pdf_content}")
        except Exception as e:
            print(f"PDF 파일 삭제 실패: {e}")

    await asyncio.sleep(3)

    for source in sources:
        info = await search(source)
        results.append(info)
    
    # 결과 반환
    return JSONResponse(results)
