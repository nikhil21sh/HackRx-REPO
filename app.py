import os
import time
import requests
import tempfile
import asyncio
import logging
import faiss
import hashlib

os.environ['LLAMA_INDEX_CACHE_DIR'] = '/tmp/llama_index_cache'
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/tmp/sentence_transformers_cache'
os.environ['NLTK_DATA'] = '/tmp/nltk_data' 


from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer
from pydantic import BaseModel, HttpUrl
from typing import List, Dict
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank



load_dotenv()
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
LLM_MODEL = "gpt-4o"

app = FastAPI(
    title="Sahyog: Policy Doc Retriever (with Pre-loading)",
    description="Pre-loads and caches large documents on startup to ensure fast responses.",
    version="5.7.1"
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

llm: OpenAI = None
embed_model: HuggingFaceEmbedding = None
query_engine_cache: Dict[str, BaseQueryEngine] = {}
APP_IS_LOADING: bool = True


async def build_and_cache_engine(doc_url: str):
    """
    This is the core logic that downloads, processes, and caches a single document.
    It's now designed to be called either on startup or on-the-fly.
    """
    global query_engine_cache
    temp_pdf_path = None
    try:
        logging.info(f"Processing document from URL: {doc_url}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            response = requests.get(doc_url, stream=True, timeout=120) 
            response.raise_for_status()
            doc_content = response.content
            temp_file.write(doc_content)
            temp_pdf_path = temp_file.name

        content_hash = hashlib.sha256(doc_content).hexdigest()
        
        if content_hash in query_engine_cache:
            logging.info(f"Document hash {content_hash} is already in cache. Skipping.")
            return

        reader = SimpleDirectoryReader(input_files=[temp_pdf_path])
        docs = reader.load_data()
        if not docs:
            raise ValueError("Could not extract text from the document.")

        embedding_dim = len(embed_model.get_text_embedding("test"))
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

        logging.info(f"Creating VectorStoreIndex for hash {content_hash}...")
        index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context,
            transformations=[node_parser],
            embed_model=embed_model
        )

        qa_prompt_template = PromptTemplate(
            "You are a precise financial policy assistant. Your task is to answer questions by extracting information ONLY from the provided CONTEXT.\n"
            "Your answer must be short, direct, and to the point. Do not provide descriptive or lengthy explanations.\n"
            "Focus on extracting key figures, numbers, percentages, and specific clauses directly from the text.\n"
            "If the context does not contain the answer, you MUST state 'The provided document does not contain sufficient information to answer this question.'\n"
            "Do not use any outside knowledge.\n"
            "---------------------\n"
            "CONTEXT: {context_str}\n"
            "---------------------\n"
            "QUESTION: {query_str}\n"
            "---------------------\n"
            "ANSWER:"
        )
        
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            llm=llm,
            text_qa_template=qa_prompt_template
        )
        
        reranker = CohereRerank(api_key=COHERE_API_KEY, top_n=3)

        query_engine = index.as_query_engine(
            response_synthesizer=response_synthesizer,
            similarity_top_k=10, 
            node_postprocessors=[reranker]
        )

        query_engine_cache[content_hash] = query_engine
        logging.info(f"New engine cached for hash {content_hash}")

    except Exception as e:
        logging.error(f"Failed to process document {doc_url}: {e}", exc_info=True)
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


@app.on_event("startup")
async def startup_event():
    global llm, embed_model, APP_IS_LOADING
    logging.info("Application startup: Loading base models...")
    if not all([API_BEARER_TOKEN, OPENAI_API_KEY, COHERE_API_KEY]):
        logging.warning("CRITICAL: One or more API keys are NOT SET in the environment!")
    try:
        llm = OpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL, temperature=0.1)
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
        logging.info("Base models loaded successfully.")
    except Exception as e:
        logging.critical(f"FATAL: A Python exception occurred during model loading: {e}", exc_info=True)
        raise

    preload_file = "preload_docs.txt"
    if os.path.exists(preload_file):
        logging.info(f"Found {preload_file}. Starting to pre-warm the cache...")
        with open(preload_file, "r") as f:
            urls_to_preload = [line.strip() for line in f if line.strip()]
        
        tasks = [build_and_cache_engine(url) for url in urls_to_preload]
        await asyncio.gather(*tasks)
        logging.info("Cache pre-warming completed.")
    else:
        logging.info(f"{preload_file} not found. Skipping cache pre-warming.")

    APP_IS_LOADING = False



auth_scheme = HTTPBearer()

def verify_token(credentials = Security(auth_scheme)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials

class RunRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]



@app.post("/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_end_to_end_query(request: RunRequest):
    if APP_IS_LOADING:
        raise HTTPException(status_code=503, detail="Service Unavailable: Application is still pre-loading documents.")
    if not llm or not embed_model:
        raise HTTPException(status_code=503, detail="Service Unavailable: Models are not ready.")

    start_request_time = time.time()
    
    # Check cache first
    doc_url_str = str(request.documents)
    response = requests.get(doc_url_str, timeout=60)
    response.raise_for_status()
    doc_content = response.content
    content_hash = hashlib.sha256(doc_content).hexdigest()

    if content_hash not in query_engine_cache:
        # If not in cache, process it on-the-fly (for documents not in preload_docs.txt)
        await build_and_cache_engine(doc_url_str)
    
    if content_hash not in query_engine_cache:
         raise HTTPException(status_code=500, detail="Failed to build engine for the provided document.")

    query_engine = query_engine_cache[content_hash]

    async def run_query(question: str):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, query_engine.query, question)
        return str(response)

    logging.info(f"Running {len(request.questions)} queries concurrently...")
    answers = await asyncio.gather(*(run_query(q) for q in request.questions))
    
    end_request_time = time.time()
    logging.info(f"Request completed in {end_request_time - start_request_time:.2f} seconds.")
    
    return RunResponse(answers=answers)


@app.get("/", include_in_schema=False)
async def root():
    if APP_IS_LOADING:
        return {"status": "loading", "message": "Service is pre-loading documents. Please wait."}
    if llm and embed_model:
        return {"status": "ok", "message": "Service is live and models are loaded."}
    return {"status": "error", "message": "Models did not load correctly."}
