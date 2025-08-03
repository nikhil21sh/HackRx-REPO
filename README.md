title: Sahyog Policy Doc Retriever
emoji: 📄🔍
colorFrom: blue
colorTo: green
sdk: docker
app_file: main.py
app_port: 7860
Sahyog: Policy Doc Retriever (with Caching)
This is a Retrieval-Augmented Generation (RAG) API that processes policy documents and answers questions with high accuracy.

Version: 4.2.0

How it Works
On-the-fly Processing: The API receives a document URL and a list of questions.
Caching: It calculates a SHA-256 hash of the document's content. If this hash has been seen before, it retrieves a pre-built query engine from an in-memory cache.
Indexing: If it's a new document, it's processed using BAAI/bge-large-en-v1.5 for embeddings and indexed into a FAISS vector store. The resulting query engine is then cached.
Querying: It uses the query engine and a gpt-4o model with a strict prompt to answer the user's questions based only on the provided document.
API Endpoint
POST /hackrx/run
Request Body:
{
  "documents": "[https://your-document-url.pdf](https://your-document-url.pdf)",
  "questions": [
    "What is question one?",
    "What is question two?"
  ]
}