import requests
import time
def fetch_pdf_from_url(url : str,path:str):
    try:
        with requests.get(url=url,timeout=30) as response:
            response.raise_for_status()
            # print(response.text)
            with open(path,'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        
        return path
    except Exception as e:
        print(f'Error {e} while loading the file ')

# demo_url="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
# local_file_path="temp.pdf"

# print(fetch_pdf_from_url(url="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",path="temp.pdf"))
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf_files(path:str):
    loader=PyPDFLoader(file_path=path)
    documents=loader.load()
    return documents

# print(load_pdf_files("temp.pdf"))


def create_pdf_chunks(documents):
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=splitter.split_documents(documents=documents)
    return chunks

# print(create_pdf_chunks(load_pdf_files("temp.pdf")))

from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def create_embeddings_and_vectorstore(url):
    path=fetch_pdf_from_url(url=url,path="temp.pdf")
    if not path:
        print("Failed to Download Pdf , process aborted.")
    documents=load_pdf_files(path=path)
    chunks=create_pdf_chunks(documents=documents)
    if not chunks:
        print("PDF couldnt be loaded !!")
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store=FAISS.from_documents(documents=chunks,embedding=embedding_model)
    return vector_store

if __name__=='__main__':
    start=time.time()
    url="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    vstore=create_embeddings_and_vectorstore(url=url)

    # if not vstore:
    #     print("Error is loading in database")
    # else:
    #     print("Vector store loaded successsfully")
    end=time.time()
    print(f'time : {end-start}')