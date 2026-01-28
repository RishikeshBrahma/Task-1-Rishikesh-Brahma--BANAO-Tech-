# Task-1-Rishikesh-Brahma--BANAO-Tech-
# 📚 Study Buddy: Agentic RAG System (Task 1)

An intelligent Retrieval-Augmented Generation (RAG) agent designed to help students query their study materials. This system uses **LangGraph** for orchestration and **TinyLlama** for local inference, ensuring privacy and offline capability.

## 🚀 Features
- **Local Inference**: Powered by TinyLlama-1.1B (GGUF) - no API keys required.
- **Agentic Workflow**: Uses LangGraph to manage the retrieval and generation states.
- **Multi-Format Support**: Ingests both **PDF** and **TXT** files with robust encoding handling.
- **REST API**: Built with **FastAPI** for easy integration with frontend applications.

## 🛠️ Project Structure
```text
├── app/
│   ├── api.py          # FastAPI Endpoint
│   ├── main.py         # LangGraph Logic & LLM Setup
│   ├── database.py     # ChromaDB Connection
│   └── ingest.py       # Data Processing Script
├── data/               # Place PDF/TXT files here
├── models/             # Place TinyLlama GGUF model here
└── requirements.txt    # Dependency list


⚙️ Setup & Installation
Clone the Repository
git clone [https://github.com/RishikeshBrahma/Task-1-Rishikesh-Brahma--BANAO-Tech-.git](https://github.com/RishikeshBrahma/Task-1-Rishikesh-Brahma--BANAO-Tech-.git)
cd Task-1-Rishikesh-Brahma--BANAO-Tech-
Install Dependencies

pip install -r requirements.txt
Download the Model Download tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf from Hugging Face and place it in the models/ folder.

Ingest Documents Place your study materials in the data/ folder and run:


python app/ingest.py
Start the API


python app/api.py
Access the interactive API docs at http://127.0.0.1:8000/docs.

📊 Performance Metrics
Average Latency: 7.18s - 9.48s (CPU-based generation).

Chunking: 500 characters with 50-character overlap.

Vector Store: ChromaDB with all-MiniLM-L6-v2 embeddings.

🛡️ Technical Challenges Overcome
Encoding Issues: Handled UnicodeDecodeError in text files by implementing a fallback encoding mechanism (utf-8 -> latin-1).

Large File Management: Utilized .gitignore to manage model binaries while maintaining repository efficiency.
>>>>>>> 21044ffc16aafef8f64b0c0bbd10c8a535b79d9d
http://127.0.0.1:8000/docs 
