import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Setup absolute paths for reliability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "../data/"))
DB_PATH = os.path.normpath(os.path.join(BASE_DIR, "../vectorstore/"))

def run_ingestion():
    print(f"--- STARTING ROBUST INGESTION ---", flush=True)
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data folder not found at {DATA_PATH}", flush=True)
        return

    all_docs = []

    # 1. Load PDFs
    print("Loading PDFs...", flush=True)
    try:
        pdf_loader = DirectoryLoader(DATA_PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
        all_docs.extend(pdf_loader.load())
    except Exception as e:
        print(f"Warning: Issue loading PDFs: {e}", flush=True)

    # 2. Load TXTs with Encoding Fallback
    print("Loading Text files with encoding fallback...", flush=True)
    txt_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        file_path = os.path.join(DATA_PATH, txt_file)
        try:
            # Try UTF-8 first
            loader = TextLoader(file_path, encoding='utf-8')
            all_docs.extend(loader.load())
        except UnicodeDecodeError:
            try:
                # Fallback to Latin-1 if UTF-8 fails
                print(f"  - Fallback encoding used for: {txt_file}", flush=True)
                loader = TextLoader(file_path, encoding='ISO-8859-1')
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"  - Skipping {txt_file} due to error: {e}", flush=True)

    print(f"--- TOTAL DOCUMENTS LOADED: {len(all_docs)} ---", flush=True)

    if not all_docs:
        print("ERROR: No documents were successfully loaded.", flush=True)
        return

    # 3. Chunking Strategy: 500 characters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_docs)
    print(f"--- CREATED {len(chunks)} CHUNKS ---", flush=True)
    
    # 4. Generate Embeddings & Save locally
    print("--- GENERATING EMBEDDINGS & SAVING TO CHROMADB ---", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print("--- SUCCESS: INGESTION COMPLETE ---", flush=True)

if __name__ == "__main__":
    run_ingestion()