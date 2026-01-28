import time
import os
from typing import TypedDict
from langchain_community.llms import LlamaCpp
from langgraph.graph import StateGraph, END
from database import get_retriever

# Path to the TinyLlama model
MODEL_PATH = os.path.join("..", "models", "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf")

# Initialize the LLM with local parameters
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.1,
    max_tokens=256,
    n_ctx=2048,
    verbose=False
)

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    latency: float

def retrieve(state: AgentState):
    """Retrieves relevant chunks from the vector store."""
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    context = "\n".join([d.page_content for d in docs])
    return {"context": context}

def generate(state: AgentState):
    """Generates a response using the local model and tracks latency."""
    start_time = time.time()
    prompt = f"<|system|>\nAnswer based ONLY on context: {state['context']}<|user|>\n{state['question']}<|assistant|>\n"
    response = llm.invoke(prompt)
    latency = time.time() - start_time
    return {"answer": response, "latency": latency}

# Build LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# This is the 'app' object imported by api.py
app = workflow.compile()

# This loop ONLY runs if you run 'python main.py'
if __name__ == "__main__":
    print("\n--- STUDY BUDDY CHATBOT MODE ---")
    while True:
        query = input("\nYou: ")
        if query.lower() in ['exit', 'quit']: break
        result = app.invoke({"question": query})
        print(f"\nAI: {result['answer']}")
        print(f"(Latency: {result['latency']:.2f}s)")