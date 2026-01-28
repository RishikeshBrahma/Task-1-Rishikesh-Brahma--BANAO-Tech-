from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# The variable name MUST be 'api'
api = FastAPI(title="Study Buddy RAG API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@api.post("/ask")
async def ask_question(request: QueryRequest):
    # Import inside the function to avoid startup crashes
    from main import app as rag_agent
    result = rag_agent.invoke({"question": request.question})
    return {
        "answer": result['answer'], 
        "latency": f"{result['latency']:.2f}s"
    }

if __name__ == "__main__":
    uvicorn.run(api, host="127.0.0.1", port=8000)