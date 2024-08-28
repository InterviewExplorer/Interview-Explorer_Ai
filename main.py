from fastapi import FastAPI, File, UploadFile
from module import firstLLM
import shutil
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyPDFLoader
app = FastAPI()

@app.post("/generateQ/")
async def create_upload_file(file: UploadFile):

     with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        # 업로드된 파일 내용을 읽어 임시 파일에 씁니다.
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    
    
    
        return { firstLLM.generateQ(temp_file_path)}