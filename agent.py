from __future__ import annotations
import os
import json
import time
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Literal
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_database"


load_dotenv()
api_key = os.getenv("GITHUB_API_KEY")
base_url = os.getenv("AZURE_BASE_URL")
# start_time = time.time()
embed_model=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vstore = FAISS.load_local(FAISS_INDEX_PATH,embeddings=embed_model,allow_dangerous_deserialization=True)
# end_time = time.time()
# print(f"Vector store created in {end_time - start_time:.2f} seconds")


class ClauseMapping(BaseModel):
    Clause_Title: str = Field(description="Title of clause or section")
    Clause_Details: List[str] = Field(description="Relevant excerpts from the clause")

class Justification(BaseModel):
    Reasoning: str = Field(description="Explanation of the logic behind the decision based on the query and clauses")
    Mapped_Clauses: List[ClauseMapping] = Field(description="List of clauses that support the decision")

class DecisionOutput(BaseModel):
    Decision: Literal["Approved", "Rejected", "Partially Approved"] = Field(description="Final decision")
    Amount: str = Field(description="Amount or conditional approval description")
    JustificationDetails: Justification = Field(description="Explanation and clause mapping for the decision")

parser = JsonOutputParser(pydantic_object=DecisionOutput)

#TO DO : token limiting - justification and amount should be limited to 1000 characters each
# reduce repetition in clause details

# ```json
# {{
#   "Decision": "Approved | Rejected | Partially Approved",
#   "Amount": "<amount or condition>",
#   "Justification": {{
#     "Reasoning": "<Explain the logic behind the decision based on the query and clauses.>",
#     "Mapped_Clauses": [
#       {{
#         "Clause_Title": "<Title of clause or section>",
#         "Clause_Details": [
#           "<Relevant sentence or excerpt 1>",
#           "<Relevant sentence or excerpt 2>"
#         ]
#       }}
#     ]
#   }}
# }}
# ```
prompt = PromptTemplate(
    template="""You are an intelligent assistant trained to evaluate user queries against policy or legal documents and provide structured, explainable outputs.

Your task is to:
1. Analyze the user's natural language query.
2. Review the provided relevant clauses from the document.
3. Based on the policy logic, make a clear decision (e.g., Approved or Rejected).
4. Determine the applicable amount, if relevant (can be exact, proportional, or "as per policy").
5. Justify the decision using specific reasoning and reference the exact clauses that support your decision.
6. Justifications should be precise, clear, and directly tied to the query and clauses.
7. Reasoning should be precise ,concise and to the point and not elaborative.

### Output Format:
{format_instruction}\n

user query : \n{query}

\n\n
Relevant Clauses: {context}
""",
    input_variables=["query","context"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

retriever = vstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
retrieve_query = """You are an intelligent assistant that analyzes large unstructured documents such as insurance policies, legal contracts, or company rules.

Your task is to retrieve all relevant information based on a natural language user query.

Instructions:
1. Carefully read the user query below.
2. Identify key entities such as:
   - People-related info (age, gender, role, etc.)
   - Events or actions (e.g., accident, surgery, resignation)
   - Location and time-based info (e.g., city, duration, effective date)
   - Any other details implied in the query
3. Read the document provided and retrieve all policy clauses, rules, exclusions, terms, or conditions that:
   - Directly match the query
   - Indirectly affect the query outcome (e.g., limitations, eligibility, exceptions)
4. Include relevant information even if the match is partial or needs interpretation.
5. Return results with:
   - The clause or summarized text
   - Why it is relevant to the query
   - Page number or section (if available)
user query : \n{query}   
"""

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=api_key,
    base_url=base_url,
)

retriever_chain = retriever

def format_docs(docs):
    context_text = "\n\n".join(doc.page_content for doc in docs)
    return context_text


parallel_chain = RunnableParallel({
    "context" : retriever_chain | format_docs,
    "query": RunnablePassthrough()  
})

main_chain = parallel_chain | prompt | llm | parser

# res = main_chain.invoke("Does the policy cover preventive diagnostics and immunizations during maternity hospitalization only?")
# print(json.dumps(res, indent=2))




if __name__ == '__main__':
    start_time = time.time()
    try:
        user_query = [" What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"]
        #user_query = input("Enter your query: ")
        if not user_query.strip():
            print("Query cannot be empty. Please enter a valid query.")
        else:
            response = main_chain.invoke(user_query)
            print("Response:", json.dumps(response, indent=2))
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")