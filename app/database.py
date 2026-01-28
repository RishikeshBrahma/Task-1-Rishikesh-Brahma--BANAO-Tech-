import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.normpath(os.path.join(BASE_DIR, "../vectorstore/"))

def get_retriever():
    """Initializes and returns the vector store retriever."""
    # Lightweight embeddings for local performance
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings
    )
    # Fetches top 3 relevant context pieces
    return vectorstore.as_retriever(search_kwargs={"k": 3})