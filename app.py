from dotenv import load_dotenv
from fastapi import FastAPI,Security,HTTPException,Depends
import time
import tempfile
import requests
from pydantic import BaseModel,Field
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
API_BEARER_TOKEN="44c4e1bfaa1815c327c40af5037b7dc1abe33a8af2271394da8bbb13690fd99c"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "gpt-4o"

app=FastAPI(title="Sahyog : Your policy query solver",description="Intelligent query retrieval bot",version="1.0.0")

llm=None
embedding_model=None
@app.on_event("startup")
async def event_startup():
    global llm,embedding_model
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0
        )
        embedding_model=HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True}
        )
        print("LLM loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load LLM during startup. Error: {e}")



def fetch_pdf_to_tempfile(url: str) -> str:
    """Downloads a PDF from a URL and saves it to a temporary file."""
    try:
        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            print(f"Downloading PDF from {url} to {temp_file.name}")
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
            return temp_file.name
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")

def process_document_and_create_retriever(pdf_path: str):
    print(f"Processing document: {pdf_path}")
    
    # Load and split the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from the document.")


    
    
    # Create an in-memory FAISS index
    print("Creating in-memory FAISS index...")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})



auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials

class HackRxRunRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxRunResponse(BaseModel):
    answers: List[str]


@app.post(
    "/hackrx/run",
    response_model=HackRxRunResponse,
    dependencies=[Depends(verify_token)]
)
async def hackrx_run(request: HackRxRunRequest):
    if not llm:
        raise HTTPException(status_code=503, detail="Service Unavailable: LLM not initialized.")

    start_full_process_time = time.time()
    temp_pdf_path = None
    
    try:
        temp_pdf_path = fetch_pdf_to_tempfile(request.documents)

        retriever = process_document_and_create_retriever(temp_pdf_path)

        qa_prompt_template =  """
You are an intelligent assistant trained to answer questions based strictly on the provided document context.

Use ONLY the context below to answer the question. Do not use prior knowledge. If the answer is not found, reply with "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
        qa_prompt = PromptTemplate.from_template(qa_prompt_template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        print("Invoking RAG chain in batch mode...")
        answers = await rag_chain.abatch(request.questions)
        
        end_full_process_time = time.time()
        print(f"Full on-the-fly process completed in {end_full_process_time - start_full_process_time:.2f} seconds.")
        
        return HackRxRunResponse(answers=answers)

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")
    
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            print(f"Cleaned up temporary file: {temp_pdf_path}")
