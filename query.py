from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings  # Or another embedding method
from typing import List
import logging
from vectorstore_setup import get_vectorstore
# Initialize embeddings and vectorstore
embedding_fn = OllamaEmbeddings(model="llama3.2")  # Ensure this matches your Ollama setup
vectorstore = Chroma(persist_directory="./chroma", embedding_function=embedding_fn)

def retrieve_questions(job_role: str = None, question_type: str = None, query: str = "interview question"):
    vectorstore = get_vectorstore()

    # Build metadata filter
    filter_criteria = {}
    if job_role:
        filter_criteria["job_role"] = job_role
    if question_type:
        filter_criteria["question_type"] = question_type

    # Perform similarity search
    results = vectorstore.similarity_search(query=query, k=10, filter=filter_criteria)

    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")
        print(f"   Metadata: {doc.metadata}")
    
    return results
