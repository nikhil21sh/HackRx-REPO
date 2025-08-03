import os
import time
import requests
import tempfile
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings



load_dotenv()

API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "44c4e1bfaa1815c327c40af5037b7dc1abe33a8af2271394da8bbb13690fd99c")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

app = FastAPI(
    title="Intelligent Query-Retrieval System (On-the-Fly)",
    description="Processes documents and questions in real-time.",
    version="2.0.0"
)

llm = None

@app.on_event("startup")
async def startup_event():
    """
    Loads the reusable Language Model once when the application starts.
    The embedding model is loaded on-demand to keep startup fast.
    """
    global llm
    print("Application startup... Loading LLM client.")
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.0
        )
        print("LLM client loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load LLM during startup. Error: {e}")


def fetch_pdf_to_tempfile(url: str) -> str:
    """Downloads a PDF from a URL and saves it to a temporary file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            print(f"Downloading PDF from {url} to {temp_file.name}")
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
            return temp_file.name
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document from URL: {e}")

def process_document_and_create_retriever(pdf_path: str):
    """Loads, chunks, and creates an in-memory vector store and retriever."""
    print(f"Processing document: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from the document.")

    print(f"Loading embedding model: {EMBEDDING_MODEL}... (This is the slow step)")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
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

        qa_prompt_template = """You are an expert AI for summarizing policy documents. Your goal is to provide a very brief, factual summary that directly answers the user's question, based on the provided context.

        **CONTEXT:**
        ---
        {context}
        ---

        **QUESTION:** {question}

        **INSTRUCTIONS:**
        1.  Read the context to understand the relevant policy rules.
        2.  Synthesize the key information into a single, concise sentence or two.
        3.  The answer should be a summary, not a direct quote or a long explanation.
        4.  Focus only on the most critical details needed to answer the question (e.g., the time period, the percentage, the core condition).
        5.  If the answer cannot be found in the context, respond with only this exact phrase: "The answer to this question could not be found in the provided document sections."

        **SUMMARY ANSWER:**
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

@app.get("/", include_in_schema=False)
async def root():
    return {"status": "ok", "message": "Service is live and ready to process requests."}
