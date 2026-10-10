import os
import shutil
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = FastAPI(title="Bank Mandiri RAG API", version="1.0.0")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_dir = "./chroma_db"
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

@app.post("/upload")
async def upload_dokumen(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Hanya menerima file PDF")
    
    file_path = f"./{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_dir)
        
        return {
            "status": "Sukses",
            "pesan": f"File {file.filename} berhasil diproses dan masuk database!",
            "total_potongan_teks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask")
async def tanya_ai(query: str):
    try:
        vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        docs = vectorstore.similarity_search(query, k=3)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt_template = """Anda adalah asisten AI profesional untuk Bank Mandiri. 
        Gunakan dokumen konteks berikut untuk menjawab pertanyaan pengguna secara akurat. 
        Jika jawaban tidak ada di dalam dokumen, katakan bahwa Anda tidak memiliki informasi tersebut.

        Konteks:
        {context}

        Pertanyaan: {question}
        
        Jawaban Profesional:"""
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = prompt | llm
        
        response = chain.invoke({"context": context, "question": query})
        
        if isinstance(response.content, list):
            teks_jawaban = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content])
        else:
            teks_jawaban = str(response.content)
            
        return {
            "pertanyaan": query,
            "jawaban": teks_jawaban.strip()
        }
    except Exception as data_error:
        raise HTTPException(status_code=500, detail=str(data_error))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)