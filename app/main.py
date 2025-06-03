# app/main.py
from fastapi import FastAPI, UploadFile, File
from .qa_chain import answer
from .ingestion import load_documents, build_index
from pathlib import Path
import shutil, uuid

app = FastAPI(title="TutorBot API")

@app.post("/ask")
async def ask_question(q:str):
    return answer(q)

@app.post("/upload")
async def upload_doc(f: UploadFile = File(...)):
    fname = f"{uuid.uuid4()}_{f.filename}"
    dest  = Path("data/docs_raw") / fname
    with dest.open("wb") as buffer:
        shutil.copyfileobj(f.file, buffer)
    # REBUILD index asynchronously in prod
    build_index()
    return {"msg":"Document ingested", "file": fname}

@app.get("/")
def health():
    return {"status": "ok"}

