from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
import os

def get_vectorstore(persist_directory="./rag_store"):
    embedding_model = OllamaEmbeddings(model="llama3.2")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    return vectorstore
