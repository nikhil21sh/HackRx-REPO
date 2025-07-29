from dotenv import load_dotenv
from fastapi import FastAPI,Security,HTTPException,Depends
import time
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
load_dotenv()
API_BEARER_TOKEN="44c4e1bfaa1815c327c40af5037b7dc1abe33a8af2271394da8bbb13690fd99c"
FAISS_INDEX_PATH="faiss_database"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o"

app=FastAPI(title="Sahyog : Your policy query solver",description="Intelligent query retrieval bot",version="1.0.0")


try:
    embed_model=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db=FAISS.load_local(FAISS_INDEX_PATH,embed_model,allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,
    )
    prompt_template ="""You are an expert AI for summarizing policy documents. Your goal is to provide a very brief, factual summary that directly answers the user's question, based on the provided context.

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

    prompt=PromptTemplate(template=prompt_template,
                          input_variables=["context","question"])
    def format_docs(docs):
        """Concatenates document contents for the context."""
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

except Exception as e:
    print(f'FATAL:Could not load models or RAG pipelines.Error:{e}')
    rag_chain=None


auth_scheme=HTTPBearer()
def verify_token(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    """Dependency to verify the bearer token."""
    if credentials.scheme != "Bearer" or credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

class HackRxRunRequest(BaseModel):
    documents: str=Field(... , description="URL of pdf")
    questions:List[str]=Field(...,description="List of questions to be asked about the document ")

class HackRxRunResponse(BaseModel):
    answers: List[str]=Field(...,description="List of Answers corresponding to the asked questions")


@app.post(
    "/hackrx/run",
    response_model=HackRxRunResponse,
    summary="Run Test Cases",
    description="Processes a list of questions against a document and returns answers.",
    dependencies=[Depends(verify_token)]
)
async def hackrx_run(request:HackRxRunRequest):
    """
    This endpoint takes a list of questions, processes them using the RAG chain,
    and returns a list of corresponding answers.
    """

    if not rag_chain:
        raise HTTPException(status_code=503, detail="Service Unavailable: RAG chain not initialized.")
    
    questions=request.questions
    if not questions:
        return HackRxRunResponse(answers=[])
    
    try:
        start_time=time.time()
        answers=await rag_chain.abatch(questions)
        end_time=time.time()
        print(f"Batch processing completed in {end_time - start_time:.2f} seconds.")
        return HackRxRunResponse(answers=answers)
    
    except Exception as e:
        print(f"An error occurred during RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the questions.")
    

    @app.get("/",include_in_schema=False)
    async def root():
        return {"message": "Intelligent Qu  ery-Retrieval System is running."}
