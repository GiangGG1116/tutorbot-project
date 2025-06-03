# app/ingestion.py
from pathlib import Path
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import os
from dotenv import load_dotenv
# Tải biến môi trường từ file .env
load_dotenv()

def load_documents(folder: Path):
    docs = []
    for file in folder.glob("*"):
        if file.suffix.lower() == ".pdf":
            docs += PyPDFLoader(str(file)).load()
        elif file.suffix.lower() in [".docx", ".doc"]:
            docs += Docx2txtLoader(str(file)).load()
        elif file.suffix.lower() in [".pptx", ".ppt"]:
            docs += UnstructuredPowerPointLoader(str(file)).load()
    return docs

def build_index():
    docs_folder = Path("/root/tutorbot-project/data/docs_raw")
    index_folder = Path("/root/tutorbot-project/data/faiss_index")
    index_folder.mkdir(parents=True, exist_ok=True)
    docs = load_documents(docs_folder)
    if not docs:
        print("No documents loaded. Exiting.")
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vecs = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )  # hoặc SentenceTransformerEmbeddings()
    store = FAISS.from_documents(chunks, vecs)
    store.save_local(str(index_folder))
    print(f"Indexed {len(chunks)} chunks.")

if __name__ == "__main__":
    build_index()
